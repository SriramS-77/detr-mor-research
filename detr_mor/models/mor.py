"""Mixture-of-Recursions (MoR) encoder and decoder blocks for DETR.

Both stacks follow the paper's **Middle-Cycle / Middle-Sequence** layout: of
``num_blocks`` blocks, the first and last keep unique weights and run densely
over every token, and the ``num_blocks - 2`` blocks in between are shared and
re-applied ``num_recursions`` times. A router decides which tokens make each
extra pass, so effective depth exceeds parameter count::

    parameter cost  = num_blocks layers
    effective depth = 2 + (num_blocks - 2) * num_recursions

FIDELITY NOTE
-------------
The MoR stacks are no longer a line-by-line port of ``train_mor.ipynb``; they
were reworked to match the paper. Plain :class:`~detr_mor.models.detr.DETR` is
still verbatim. Remaining known deviations are listed in the README.
"""

import torch
import torch.nn as nn


def _gather_tokens(source, indices, embed_dim):
    """Pick ``indices`` (B, k) out of ``source`` (B, N, embed_dim)."""
    return torch.gather(
        source, 1, indices.unsqueeze(-1).expand(-1, -1, embed_dim))


class MoRExpertRouter(nn.Module):
    r"""
    Scores each token with a scalar in (0, 1) and keeps the top-k.

    :param embed_dim: token dimension
    """

    def __init__(self, embed_dim):
        super().__init__()
        # MoR uses a scalar routing score per token
        self.router_weights = nn.Linear(embed_dim, 1)
        self.router_func = nn.Sigmoid()

    def forward(self, x, topk):
        r"""
        :param x: (B, active_len, embed_dim)
        :param topk: how many tokens to keep
        :return: (weights, indices), both (B, topk); indices are relative to the
            input's own token axis and are returned in ascending order so that
            original token ordering is preserved.
        """
        # 1. Compute scalar scores g^r for each token
        scores = self.router_func(self.router_weights(x)).squeeze(-1)
        # scores -> (B, active_len)

        # 2. Select top-k tokens based on scores
        weights, rel_indices = torch.topk(scores, topk, dim=1)

        # 3. Sort indices to maintain original token position order
        sorted_indices, sort_idx = torch.sort(rel_indices, dim=1)
        sorted_weights = torch.gather(weights, 1, sort_idx)

        return sorted_weights, sorted_indices


class MoRRecursionBlock(nn.Module):
    r"""
    Reference implementation of a single MoR recursion block, kept from the
    notebook for documentation purposes.

    NOT USED by :class:`MoREncoder` / :class:`MoRDecoder` - those inline the same
    gather / route / recurse / scatter_add steps so they can interleave attention
    with the position embeddings. Kept here because it is the clearest statement
    of what the MoR step is meant to do.
    """

    def __init__(self, layers, router, embed_dim):
        super().__init__()
        self.layers = layers  # Shared stack of layers
        self.router = router
        self.embed_dim = embed_dim

    def forward(self, x, active_indices, topk):
        r"""
        :param x: full hidden states (B, seq_len, embed_dim)
        :param active_indices: indices of tokens eligible for this step
            (hierarchical filtering)
        :param topk: how many of the active tokens to keep
        :return: (updated x, new active indices)
        """
        # 1. Gather tokens eligible for this recursion
        active_x = torch.gather(
            x, 1, active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # 2. Routing: Select which tokens from the active set continue
        weights, rel_indices = self.router(active_x, topk)

        # 3. Map relative indices back to absolute sequence positions
        abs_indices = torch.gather(active_indices, 1, rel_indices)

        # 4. Extract selected patches for computation. This focuses computation
        #    only on tokens still active.
        selected_patches = torch.gather(
            active_x, 1, rel_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # 5. Pass through the SHARED layers (the recursion block)
        hidden_states = selected_patches
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 6. Apply router weights and prepare update
        weighted_updates = hidden_states * weights.unsqueeze(-1)

        # 7. RECOMBINE: scatter updates back into the full sequence. This
        #    preserves unselected tokens (the residual / skipped path).
        x = x.clone()  # Avoid in-place issues if needed
        x.scatter_add_(
            1, abs_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            weighted_updates)

        # Return updated sequence and the new active set (hierarchical filtering)
        return x, abs_indices


class _MoRMiddleStack(nn.Module):
    r"""
    Routing bookkeeping shared by the encoder and decoder middle stacks.

    Terminology used throughout this module:

    ``stage``
        One gated unit of work - the thing a router decides about. A token
        either enters a stage and is updated by it, or skips it and keeps its
        current value. What counts as a stage is schedule-dependent (see the
        Cyclic / Sequential subclasses); ``num_stages`` is the total, and there
        are ``num_stages - 1`` routers because stage 0 always runs densely.

    ``capacity schedule``
        At stage ``s`` the router keeps ``((num_stages - s) / num_stages) * N``
        tokens, where ``N`` is the *full* sequence length, so the active set
        shrinks linearly from all ``N`` tokens down to ``N / num_stages``.

    :param num_blocks: number of distinct middle weight sets (M)
    :param num_recursions: how many times the middle group is re-applied (R)
    :param num_stages: number of gated stages this schedule exposes
    """

    def __init__(self, num_blocks, num_recursions, d_model, num_stages):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_recursions = num_recursions
        self.num_stages = num_stages
        self.embed_dim = d_model

        # Stage 0 runs densely and needs no router, hence num_stages - 1.
        # Stage s (s >= 1) uses exp_routers[s - 1], so every router is reached.
        self.exp_routers = nn.ModuleList([
            MoRExpertRouter(d_model) for _ in range(max(num_stages - 1, 0))
        ])

    def route(self, stage, out, active_indices):
        r"""
        Run the stage-``stage`` router and gather the tokens that survive it.

        :param stage: gated stage index, must be >= 1
        :param out: (B, N, d_model) full hidden states
        :param active_indices: (B, k_prev) absolute positions still in play
        :return: ``(weights (B, k), active_indices (B, k), tokens (B, k, d))``
        """
        # Capacity is a fraction of the FULL sequence length, not of the
        # currently active set, so the schedule shrinks monotonically.
        topk = int(((self.num_stages - stage) / self.num_stages) * out.size(1))

        # Gather the still-eligible tokens (hierarchical filtering: the active
        # set can only ever shrink), score them, keep the top-k.
        active_out = _gather_tokens(out, active_indices, self.embed_dim)
        weights, rel_indices = self.exp_routers[stage - 1](active_out, topk)

        # Map the router's relative indices back to absolute positions.
        active_indices = torch.gather(active_indices, 1, rel_indices)
        tokens = _gather_tokens(active_out, rel_indices, self.embed_dim)
        return weights, active_indices, tokens

    def recombine(self, out, tokens, base, weights, active_indices):
        r"""
        Scatter a gated update back into the full sequence, implementing the MoR
        recurrence ``h <- h + g * f(h)`` where ``f(h) = tokens - base``.

        Unselected tokens are left untouched (the skipped path), and a selected
        token with ``g -> 0`` is also left untouched, which is what makes the
        gate meaningful.

        Scattering ``g * tokens`` instead would give ``(1 + g) * base +
        g * delta`` - the token's own value added back on top of itself at every
        stage, so activations grow by ~``(1 + g)`` per stage and blow up
        geometrically down a deep stack.

        :param tokens: (B, k, d_model) stage output for the selected tokens
        :param base: (B, k, d_model) the same tokens as they entered the stage
        :param weights: (B, k) router scores
        :return: (B, N, d_model)
        """
        update = (tokens - base) * weights.unsqueeze(-1)
        # scatter_add_ is in-place, so work on a copy and leave the caller's
        # tensor (and the autograd graph that reads it) alone.
        out = out.clone()
        out.scatter_add_(
            1,
            active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            update)
        return out


class _MoREncoderMiddleStack(_MoRMiddleStack):
    """Weights for the encoder's shared middle blocks (self-attn + MLP)."""

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob, num_stages):
        super().__init__(num_blocks, num_recursions, d_model, num_stages)
        self.dropout_prob = dropout_prob
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_blocks)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ff_inner_dim, d_model),
            )
            for _ in range(num_blocks)
        ])
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_blocks)])
        self.ff_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_blocks)])

    def apply_block(self, block_idx, tokens, pos_embeds):
        r"""
        One pre-norm encoder layer over the active tokens only.

        :param tokens: (B, k, d_model)
        :param pos_embeds: position embeddings for exactly those tokens
        :return: (B, k, d_model)
        """
        # Norm, Self Attention, Dropout and Residual. Position embeddings go on
        # q/k only, never on v (as in DETR).
        in_attn = self.attn_norms[block_idx](tokens)
        q = in_attn + pos_embeds
        k = in_attn + pos_embeds
        out_attn, _ = self.attns[block_idx](query=q, key=k, value=in_attn)
        tokens = tokens + self.attn_dropouts[block_idx](out_attn)

        # Norm, MLP, Dropout and Residual
        in_ff = self.ff_norms[block_idx](tokens)
        out_ff = self.ffs[block_idx](in_ff)
        tokens = tokens + self.ff_dropouts[block_idx](out_ff)
        return tokens


class MoREncoderCyclicRecursion(_MoREncoderMiddleStack):
    r"""
    Cycle-style recursion over the encoder's middle blocks - the body of the
    paper's "Middle-Cycle" strategy, its best-performing variant.

    With M middle blocks and R recursions the visit order is::

        recursion 0:  block 0 -> block 1 -> ... -> block M-1
        recursion 1:  block 0 -> block 1 -> ... -> block M-1
        ...           (R times in total)

    **A stage is one complete pass over all M blocks**, so there are R stages
    and R - 1 routing decisions. This is the granularity MoR is defined at: a
    token either makes a whole extra pass through the shared block or it stops
    recursing. Routing per *layer* instead would let a token be updated by
    block 0 and block 1 but not by blocks 2..M-1 within the same pass, which
    leaves "recursion depth" undefined for that token and prunes far harder
    than the intended schedule (M*R-1 decisions instead of R-1).
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0):
        super().__init__(num_blocks, num_recursions, num_heads, d_model,
                         ff_inner_dim, dropout_prob,
                         num_stages=num_recursions)

    def forward(self, x, spatial_position_embedding, active_indices):
        r"""
        :param x: (B, N, d_model)
        :param spatial_position_embedding: (N, d_model)
        :param active_indices: (B, N) int64, arange(N) on entry
        :return: (B, N, d_model)
        """
        out = x

        for recursion in range(self.num_recursions):
            if recursion == 0:
                # First pass is dense and ungated: every token recurses at
                # least once, so there is nothing to route yet.
                weights = None
                tokens = out
                pos_embeds = spatial_position_embedding
            else:
                # Routing happens HERE, once per recursion, before the cycle
                # starts - not between the blocks inside it.
                weights, active_indices, tokens = self.route(
                    recursion, out, active_indices)
                pos_embeds = spatial_position_embedding[active_indices]

            # The recursion body: ONE full cycle through every middle block.
            # `base` is the state the selected tokens entered this pass with;
            # `tokens - base` is the f(h) that the gate multiplies below.
            base = tokens
            for block_idx in range(self.num_blocks):
                tokens = self.apply_block(block_idx, tokens, pos_embeds)

            if weights is None:
                out = tokens
            else:
                out = self.recombine(out, tokens, base, weights, active_indices)

        return out


class MoREncoderSequentialRecursion(_MoREncoderMiddleStack):
    r"""
    Sequence-style recursion over the encoder's middle blocks - the body of the
    paper's "Middle-Sequence" strategy.

    With M middle blocks and R recursions the visit order is::

        block 0 -> block 0 -> ... (R times)
        block 1 -> block 1 -> ... (R times)
        ...                       (M blocks in total)

    **A stage is one application of one block**, so there are M * R stages and
    M * R - 1 routing decisions. Unlike Cycle there is no natural "full pass"
    boundary here - consecutive steps reuse the same weights - so the per-step
    granularity is the sensible one for this schedule.
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0):
        super().__init__(num_blocks, num_recursions, num_heads, d_model,
                         ff_inner_dim, dropout_prob,
                         num_stages=num_blocks * num_recursions)

    def forward(self, x, spatial_position_embedding, active_indices):
        r"""
        :param x: (B, N, d_model)
        :param spatial_position_embedding: (N, d_model)
        :param active_indices: (B, N) int64, arange(N) on entry
        :return: (B, N, d_model)
        """
        out = x
        stage = 0

        for block_idx in range(self.num_blocks):
            for _ in range(self.num_recursions):
                if stage == 0:
                    weights = None
                    tokens = out
                    pos_embeds = spatial_position_embedding
                else:
                    weights, active_indices, tokens = self.route(
                        stage, out, active_indices)
                    pos_embeds = spatial_position_embedding[active_indices]

                base = tokens
                tokens = self.apply_block(block_idx, tokens, pos_embeds)

                if weights is None:
                    out = tokens
                else:
                    out = self.recombine(out, tokens, base, weights,
                                         active_indices)
                stage += 1

        return out


class _MoRDecoderMiddleStack(_MoRMiddleStack):
    """Weights for the decoder's shared middle blocks (self + cross attn, MLP)."""

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob, num_stages):
        super().__init__(num_blocks, num_recursions, d_model, num_stages)
        self.dropout_prob = dropout_prob
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_blocks)
        ])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_blocks)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ff_inner_dim, d_model),
            )
            for _ in range(num_blocks)
        ])
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.cross_attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_blocks)])
        self.cross_attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_blocks)])
        self.ff_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_blocks)])

    def apply_block(self, block_idx, tokens, query_pos, encoder_output,
                    spatial_position_embedding):
        r"""
        One pre-norm decoder layer over the active queries only.

        :param tokens: (B, k, d_model) active object queries
        :param query_pos: (B, k, d_model) their learned query embeddings
        :param encoder_output: (B, num_tokens, d_model) - always attended in
            full, only the query side is routed
        :return: (B, k, d_model)
        """
        # Norm, Self Attention, Dropout and Residual
        in_attn = self.attn_norms[block_idx](tokens)
        q = in_attn + query_pos
        k = in_attn + query_pos
        out_attn, _ = self.attns[block_idx](query=q, key=k, value=in_attn)
        tokens = tokens + self.attn_dropouts[block_idx](out_attn)

        # Norm, Cross Attention, Dropout and Residual
        in_attn = self.cross_attn_norms[block_idx](tokens)
        q = in_attn + query_pos
        k = encoder_output + spatial_position_embedding
        out_attn, _ = self.cross_attns[block_idx](
            query=q, key=k, value=encoder_output)
        tokens = tokens + self.cross_attn_dropouts[block_idx](out_attn)

        # Norm, MLP, Dropout and Residual
        in_ff = self.ff_norms[block_idx](tokens)
        out_ff = self.ffs[block_idx](in_ff)
        tokens = tokens + self.ff_dropouts[block_idx](out_ff)
        return tokens


class MoRDecoderCyclicRecursion(_MoRDecoderMiddleStack):
    r"""
    Cycle-style recursion over the decoder's middle blocks. Stage semantics are
    identical to :class:`MoREncoderCyclicRecursion`: R stages, R - 1 routers,
    one routing decision per full pass over the M middle blocks.
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0):
        super().__init__(num_blocks, num_recursions, num_heads, d_model,
                         ff_inner_dim, dropout_prob,
                         num_stages=num_recursions)

    def forward(self, out, encoder_output, query_embedding,
                spatial_position_embedding, active_indices):
        r"""
        :param out: (B, num_queries, d_model) running query state
        :return: ``(out, intermediates)`` - ``intermediates`` holds the full
            sequence state after each stage, for deep supervision. There is one
            per stage rather than one per block application because a
            full-sequence state only exists at stage boundaries: mid-cycle, only
            the selected queries have been advanced.
        """
        intermediates = []

        for recursion in range(self.num_recursions):
            if recursion == 0:
                weights = None
                tokens = out
                query_pos = query_embedding
            else:
                weights, active_indices, tokens = self.route(
                    recursion, out, active_indices)
                query_pos = _gather_tokens(query_embedding, active_indices,
                                           self.embed_dim)

            base = tokens
            for block_idx in range(self.num_blocks):
                tokens = self.apply_block(block_idx, tokens, query_pos,
                                          encoder_output,
                                          spatial_position_embedding)

            if weights is None:
                out = tokens
            else:
                out = self.recombine(out, tokens, base, weights, active_indices)
            intermediates.append(out)

        return out, intermediates


class MoRDecoderSequentialRecursion(_MoRDecoderMiddleStack):
    r"""
    Sequence-style recursion over the decoder's middle blocks. Stage semantics
    are identical to :class:`MoREncoderSequentialRecursion`: M * R stages, one
    routing decision per block application.
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0):
        super().__init__(num_blocks, num_recursions, num_heads, d_model,
                         ff_inner_dim, dropout_prob,
                         num_stages=num_blocks * num_recursions)

    def forward(self, out, encoder_output, query_embedding,
                spatial_position_embedding, active_indices):
        r""":return: ``(out, intermediates)``, one intermediate per stage."""
        intermediates = []
        stage = 0

        for block_idx in range(self.num_blocks):
            for _ in range(self.num_recursions):
                if stage == 0:
                    weights = None
                    tokens = out
                    query_pos = query_embedding
                else:
                    weights, active_indices, tokens = self.route(
                        stage, out, active_indices)
                    query_pos = _gather_tokens(query_embedding, active_indices,
                                               self.embed_dim)

                base = tokens
                tokens = self.apply_block(block_idx, tokens, query_pos,
                                          encoder_output,
                                          spatial_position_embedding)

                if weights is None:
                    out = tokens
                else:
                    out = self.recombine(out, tokens, base, weights,
                                         active_indices)
                intermediates.append(out)
                stage += 1

        return out, intermediates


#: Middle-Cycle / Middle-Sequence both keep the first and last block unshared.
#: That count is structural, not a hyper-parameter.
NUM_UNIQUE_BLOCKS = 2

#: Recursion schedules accepted by both stacks.
RECURSION_TYPES = ('cyclic', 'sequential')

#: (cyclic, sequential) middle-stack classes, per stack.
_ENCODER_MIDDLE = (MoREncoderCyclicRecursion, MoREncoderSequentialRecursion)
_DECODER_MIDDLE = (MoRDecoderCyclicRecursion, MoRDecoderSequentialRecursion)


def _validate_middle_layout(num_blocks, recursion_type, config_key):
    """Reject layouts where the shared middle group would be empty."""
    if recursion_type not in RECURSION_TYPES:
        raise ValueError('recursion_type must be one of {}, got {!r}'.format(
            RECURSION_TYPES, recursion_type))
    if num_blocks < NUM_UNIQUE_BLOCKS + 1:
        # With num_blocks == 2 there are zero middle blocks, so the middle stack
        # is an empty ModuleList and its forward a silent no-op: the "MoR" stack
        # would degenerate into a plain 2-layer transformer while every smoke
        # test still passed. Fail loudly instead.
        raise ValueError(
            '{} must be >= {}: the first and last blocks are unshared, which '
            'leaves num_blocks - 2 = {} shared middle blocks and nothing to '
            'recurse over. Use 4 or more for a meaningful middle group.'.format(
                config_key, NUM_UNIQUE_BLOCKS + 1,
                num_blocks - NUM_UNIQUE_BLOCKS))


def _pick_middle_cls(candidates, recursion_type):
    return candidates[0] if recursion_type == 'cyclic' else candidates[1]


class MoREncoder(nn.Module):
    r"""
    MoR encoder for DETR, in the Middle-Cycle / Middle-Sequence layout::

        block 0              unique weights, dense over all tokens
        middle stack         num_blocks - 2 shared blocks, recursed and routed
        block num_blocks-1   unique weights, dense over all tokens

    Keeping the first and last blocks unshared is what makes this variant work:
    the first gives every token a full-resolution representation before any
    routing decision is made, and the last re-mixes the sequence after the
    routed tokens have been scattered back in, so tokens that exited early are
    not left stale.

    :param num_blocks: TOTAL blocks including the two unique ones. Must be >= 3.
    :param num_recursions: how many times the middle group is re-applied
    :param recursion_type: 'cyclic' (Middle-Cycle, the paper's best) or
        'sequential' (Middle-Sequence)
    :param if_middle_cycle: accepted but unused; superseded by ``recursion_type``
        and kept only so existing configs keep loading.
    """

    RECURSION_TYPES = RECURSION_TYPES

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0, if_middle_cycle=False,
                 recursion_type='cyclic'):
        super().__init__()
        _validate_middle_layout(num_blocks, recursion_type,
                                'encoder_num_blocks')

        self.num_blocks = num_blocks
        self.num_middle_blocks = num_blocks - NUM_UNIQUE_BLOCKS
        self.num_recursions = num_recursions
        self.recursion_type = recursion_type
        self.dropout_prob = dropout_prob
        self.embed_dim = d_model
        self.if_middle_cycle = if_middle_cycle
        self.effective_depth = (NUM_UNIQUE_BLOCKS +
                                self.num_middle_blocks * num_recursions)

        self.middle_recursion_blocks = _pick_middle_cls(
            _ENCODER_MIDDLE, recursion_type)(
                num_blocks=self.num_middle_blocks,
                num_recursions=num_recursions,
                num_heads=num_heads,
                d_model=d_model,
                ff_inner_dim=ff_inner_dim,
                dropout_prob=dropout_prob)

        # Weights for the two unique blocks: index 0 pre-recursion, 1 post.
        num_unique = NUM_UNIQUE_BLOCKS
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_unique)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ff_inner_dim, d_model),
            )
            for _ in range(num_unique)
        ])
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_unique)])
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_unique)])
        self.attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_unique)])
        self.ff_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_unique)])

        # Norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def _unique_block(self, block_idx, out, spatial_position_embedding):
        """One dense pre-norm encoder layer over the whole sequence."""
        # Norm, Self Attention, Dropout and Residual
        in_attn = self.attn_norms[block_idx](out)
        # Add spatial position embedding to q,k for self attention
        q = in_attn + spatial_position_embedding
        k = in_attn + spatial_position_embedding
        out_attn, _ = self.attns[block_idx](query=q, key=k, value=in_attn)
        out = out + self.attn_dropouts[block_idx](out_attn)

        # Norm, MLP, Dropout and Residual
        in_ff = self.ff_norms[block_idx](out)
        out_ff = self.ffs[block_idx](in_ff)
        out = out + self.ff_dropouts[block_idx](out_ff)
        return out

    def forward(self, x, spatial_position_embedding, active_indices):
        r"""
        :param x: (B, seq_len, d_model) flattened backbone features
        :param spatial_position_embedding: (seq_len, d_model)
        :param active_indices: (B, seq_len) int64, initially arange(seq_len)
        :return: (B, seq_len, d_model) encoder output

        NOTE(nb-fidelity): unlike :class:`TransformerEncoder`, no attention
        weights are returned, so ``MoRDETR`` omits 'enc_attn' from its output.
        """
        # 1. First unique block - dense, every token, no routing yet.
        out = self._unique_block(0, x, spatial_position_embedding)

        # 2. Shared middle group, recursed and routed.
        out = self.middle_recursion_blocks(out, spatial_position_embedding,
                                           active_indices)

        # 3. Last unique block - dense again, so tokens routed out early get
        #    re-mixed with the ones that recursed the whole way.
        out = self._unique_block(1, out, spatial_position_embedding)

        out = self.output_norm(out)
        return out


class MoRDecoder(nn.Module):
    r"""
    MoR decoder for DETR, in the same Middle-Cycle / Middle-Sequence layout as
    :class:`MoREncoder`, applied to the object queries. Only the query side is
    routed; cross-attention always sees the full encoder output.

    Each block is the usual pre-norm decoder layer (self attention, cross
    attention on the encoder output, MLP).

    Deep supervision: ``num_outputs = 2 + middle.num_stages`` states are
    returned - one after each unique block, plus one per middle stage. The
    middle stack emits per *stage* rather than per block application because a
    full-sequence query state only exists at stage boundaries; mid-cycle, only
    the selected queries have advanced. So ``num_outputs`` is
    ``2 + num_recursions`` under 'cyclic' and
    ``2 + (num_blocks - 2) * num_recursions`` under 'sequential'.

    :param num_blocks: TOTAL blocks including the two unique ones. Must be >= 3.
    :param if_middle_cycle: accepted but unused; superseded by ``recursion_type``
    """

    RECURSION_TYPES = RECURSION_TYPES

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0, if_middle_cycle=False,
                 recursion_type='cyclic'):
        super().__init__()
        _validate_middle_layout(num_blocks, recursion_type,
                                'decoder_num_blocks')

        self.num_blocks = num_blocks
        self.num_middle_blocks = num_blocks - NUM_UNIQUE_BLOCKS
        self.num_recursions = num_recursions
        self.recursion_type = recursion_type
        self.dropout_prob = dropout_prob
        self.embed_dim = d_model
        self.if_middle_cycle = if_middle_cycle
        self.effective_depth = (NUM_UNIQUE_BLOCKS +
                                self.num_middle_blocks * num_recursions)

        self.middle_recursion_blocks = _pick_middle_cls(
            _DECODER_MIDDLE, recursion_type)(
                num_blocks=self.num_middle_blocks,
                num_recursions=num_recursions,
                num_heads=num_heads,
                d_model=d_model,
                ff_inner_dim=ff_inner_dim,
                dropout_prob=dropout_prob)

        #: How many deeply-supervised outputs :meth:`forward` stacks.
        self.num_outputs = (NUM_UNIQUE_BLOCKS +
                            self.middle_recursion_blocks.num_stages)

        num_unique = NUM_UNIQUE_BLOCKS
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_unique)
        ])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                  batch_first=True)
            for _ in range(num_unique)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ff_inner_dim, d_model),
            )
            for _ in range(num_unique)
        ])
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_unique)])
        self.cross_attn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_unique)])
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_unique)])
        self.attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_unique)])
        self.cross_attn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_unique)])
        self.ff_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(num_unique)])

        # Shared Output norm for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def _unique_block(self, block_idx, out, encoder_output, query_embedding,
                      spatial_position_embedding):
        """One dense pre-norm decoder layer over every query."""
        # Norm, Self Attention, Dropout and Residual
        in_attn = self.attn_norms[block_idx](out)
        q = in_attn + query_embedding
        k = in_attn + query_embedding
        out_attn, _ = self.attns[block_idx](query=q, key=k, value=in_attn)
        out = out + self.attn_dropouts[block_idx](out_attn)

        # Norm, Cross Attention, Dropout and Residual
        in_attn = self.cross_attn_norms[block_idx](out)
        q = in_attn + query_embedding
        k = encoder_output + spatial_position_embedding
        out_attn, _ = self.cross_attns[block_idx](
            query=q, key=k, value=encoder_output)
        out = out + self.cross_attn_dropouts[block_idx](out_attn)

        # Norm, MLP, Dropout and Residual
        in_ff = self.ff_norms[block_idx](out)
        out_ff = self.ffs[block_idx](in_ff)
        out = out + self.ff_dropouts[block_idx](out_ff)
        return out

    def forward(self, query_objects, encoder_output,
                query_embedding, spatial_position_embedding, active_indices):
        r"""
        :param query_objects: (B, num_queries, d_model), zeros on entry
        :param encoder_output: (B, num_tokens, d_model)
        :param query_embedding: (B, num_queries, d_model) learned object queries
        :param spatial_position_embedding: (num_tokens, d_model)
        :param active_indices: (B, num_queries) int64, initially arange
        :return: (num_outputs, B, num_queries, d_model) for deep supervision

        NOTE(nb-fidelity): cross-attention weights are not returned, so
        ``MoRDETR`` omits 'dec_attn' from its output dict.
        """
        # 1. First unique block - dense, every query, no routing yet. This is
        #    also what lifts `query_objects` off its all-zeros initialisation
        #    before anything is gathered from it.
        out = self._unique_block(0, query_objects, encoder_output,
                                 query_embedding, spatial_position_embedding)
        decoder_outputs = [out]

        # 2. Shared middle group, recursed and routed.
        out, intermediates = self.middle_recursion_blocks(
            out, encoder_output, query_embedding, spatial_position_embedding,
            active_indices)
        decoder_outputs.extend(intermediates)

        # 3. Last unique block - dense again, so queries routed out early are
        #    re-mixed before the class/bbox heads read them.
        out = self._unique_block(1, out, encoder_output, query_embedding,
                                 spatial_position_embedding)
        decoder_outputs.append(out)

        return torch.stack([self.output_norm(state)
                            for state in decoder_outputs])
