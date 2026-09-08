"""Mixture-of-Recursions (MoR) encoder and decoder blocks for DETR.

A MoR stack replaces the usual "N distinct layers" with ``num_blocks`` recursion
blocks, each of which applies its own weights ``num_recursions`` times. Between
blocks an expert router scores every still-active token and keeps only the
top-k, so later blocks run on progressively fewer tokens - that is where the
compute saving comes from. Skipped tokens keep their previous value (the
residual / skipped path), and the selected tokens' updates are scattered back
into the full sequence, weighted by their router score.

FIDELITY NOTE
-------------
These forward passes are a deliberate line-by-line port of ``train_mor.ipynb``.
Several details look unintentional but are preserved so that training runs stay
comparable with the notebook's results. Each is flagged inline with
``NOTE(nb-fidelity)`` and summarised in the README. Do not "clean them up"
without re-running the baselines.
"""

import torch
import torch.nn as nn


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


class MoREncoder(nn.Module):
    r"""
    MoR encoder for DETR.

    ``num_blocks`` recursion blocks, each applied ``num_recursions`` times, so
    the effective depth is ``num_blocks * num_recursions`` while the parameter
    count is that of ``num_blocks`` layers. Each block after the first routes
    down to a shrinking top-k of tokens.

    Each block's layer stack is the usual pre-norm encoder layer:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for MLP
        4. MLP

    :param num_blocks: number of distinct weight sets
    :param num_recursions: how many times each block's weights are re-applied
    :param if_middle_cycle: accepted but unused; kept for config compatibility
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0, if_middle_cycle=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_recursions = num_recursions
        self.dropout_prob = dropout_prob
        self.embed_dim = d_model
        self.active_indices = None
        self.if_middle_cycle = if_middle_cycle

        # Expert Routers for each Recursion Block
        self.exp_routers = nn.ModuleList(
            [
                MoRExpertRouter(d_model)
                for _ in range(num_blocks)
            ])

        # Self Attention Module for all encoder blocks
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_blocks)
            ])

        # MLP Module for all encoder blocks
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_blocks)
            ])

        # Norm for Self Attention for all encoder blocks
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_blocks)
            ])

        # Norm for MLP for all encoder blocks
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_blocks)
            ])

        # Dropout for Self Attention for all encoder blocks
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_blocks)
            ])

        # Dropout for MLP for all encoder blocks
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_blocks)
            ])

        # Norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_position_embedding, active_indices):
        r"""
        :param x: (B, seq_len, d_model) flattened backbone features
        :param spatial_position_embedding: (seq_len, d_model)
        :param active_indices: (B, seq_len) int64, initially arange(seq_len)
        :return: (B, seq_len, d_model) encoder output

        NOTE(nb-fidelity): unlike :class:`TransformerEncoder`, no attention
        weights are returned. The notebook commented the stacking out, and
        ``MoRDETR`` correspondingly omits 'enc_attn' from its output dict.
        """
        selected_patches = None
        weights = None
        out = x
        for i in range(self.num_blocks):
            # x: full hidden states (B, seq_len, embed_dim)
            # active_indices: tokens eligible for this step (hierarchical filtering)
            topk = int(((self.num_blocks - i) / self.num_blocks) * out.size(1))

            if i > 0:
                # NOTE(nb-fidelity): from block 2 onwards this resets `out` to the
                # scatter_add-updated `x`. At block 1 it does not, and since block 0
                # performs no scatter_add (weights is still None), block 0's dense
                # output is read here but then discarded: the scatter_add below
                # writes into the *original* input `x`. Preserved as in the notebook.
                if i > 1:
                    out = x

                # 1. Gather tokens eligible for this recursion
                active_out = torch.gather(
                    out, 1,
                    active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

                # 2. Routing: select which tokens from the active set continue
                weights, rel_indices = self.exp_routers[i](active_out, topk)

                # 3. Map relative indices back to absolute sequence positions
                active_indices = torch.gather(active_indices, 1, rel_indices)

                # 4. Extract selected patches for computation. This focuses
                #    computation only on tokens still active.
                selected_patches = torch.gather(
                    active_out, 1,
                    rel_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

            # Running through the same layers more than once as per MoR
            for _ in range(self.num_recursions):
                if selected_patches is not None:
                    # Routed path: attend only over the selected tokens.
                    # Norm, Self Attention, Dropout and Residual
                    in_attn = self.attn_norms[i](selected_patches)
                    # NOTE(nb-fidelity): advanced indexing a (seq_len, d_model)
                    # embedding table with (B, topk) indices yields
                    # (B, topk, d_model) - the per-token position embeddings for
                    # the surviving tokens.
                    temp_pos_embeds = spatial_position_embedding[active_indices]
                    q = in_attn + temp_pos_embeds
                    k = in_attn + temp_pos_embeds
                    out_attn, _ = self.attns[i](
                        query=q,
                        key=k,
                        value=in_attn
                    )
                    out_attn = self.attn_dropouts[i](out_attn)
                    selected_patches = selected_patches + out_attn

                    # Norm, MLP, Dropout and Residual
                    in_ff = self.ff_norms[i](selected_patches)
                    out_ff = self.ffs[i](in_ff)
                    out_ff = self.ff_dropouts[i](out_ff)
                    selected_patches = selected_patches + out_ff
                else:
                    # Dense path: block 0 runs over every token.
                    # Norm, Self Attention, Dropout and Residual
                    in_attn = self.attn_norms[i](out)
                    # Add spatial position embedding to q,k for self attention
                    q = in_attn + spatial_position_embedding
                    k = in_attn + spatial_position_embedding
                    out_attn, _ = self.attns[i](
                        query=q,
                        key=k,
                        value=in_attn
                    )
                    out_attn = self.attn_dropouts[i](out_attn)
                    out = out + out_attn

                    # Norm, MLP, Dropout and Residual
                    in_ff = self.ff_norms[i](out)
                    out_ff = self.ffs[i](in_ff)
                    out_ff = self.ff_dropouts[i](out_ff)
                    out = out + out_ff

            if weights is not None:
                # 6. Apply router weights and prepare update
                weighted_out = selected_patches * weights.unsqueeze(-1)

                # 7. RECOMBINE: scatter updates back into the full sequence. This
                #    preserves unselected tokens (the residual / skipped path).
                x = x.clone()  # Avoid in-place issues if needed
                x.scatter_add_(
                    1,
                    active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                    weighted_out)

        # The loop above updates `x` after every recursion block, which is why the
        # next block re-reads `out = x`.
        # NOTE(nb-fidelity): the output is normed from `x`, not `out`. With a
        # single block (no routing ever happens) this returns the *unmodified*
        # input, so num_blocks must be >= 2 for the encoder to do anything.
        out = self.output_norm(x)
        return out


class MoRDecoder(nn.Module):
    r"""
    MoR decoder for DETR.

    Same recursion-block structure as :class:`MoREncoder`, applied to the object
    queries. Each block's layer stack is:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for Cross Attention on Encoder Outputs
        4. Cross Attention
        5. LayerNorm for MLP
        6. MLP

    :param if_middle_cycle: accepted but unused; kept for config compatibility
    """

    def __init__(self, num_blocks, num_recursions, num_heads, d_model,
                 ff_inner_dim, dropout_prob=0.0, if_middle_cycle=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_recursions = num_recursions
        self.dropout_prob = dropout_prob
        self.embed_dim = d_model
        self.active_indices = None
        self.if_middle_cycle = if_middle_cycle

        # Expert Routers for each Recursion Block
        self.exp_routers = nn.ModuleList(
            [
                MoRExpertRouter(d_model)
                for _ in range(num_blocks)
            ])

        # Self Attention module for all decoder blocks
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_blocks)
            ])

        # Cross Attention Module for all decoder blocks
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_blocks)
            ])

        # MLP Module for all decoder blocks
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_blocks)
            ])

        # Norm for Self Attention Module for all decoder blocks
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_blocks)
            ])

        # Norm for Cross Attention Module for all decoder blocks
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_blocks)
            ])

        # Norm for MLP Module for all decoder blocks
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_blocks)
            ])

        # Dropout for Attention Module for all decoder blocks
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_blocks)
            ])

        # Dropout for Cross Attention Module for all decoder blocks
        self.cross_attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_blocks)
            ])

        # Dropout for MLP Module for all decoder blocks
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_blocks)
            ])

        # Shared Output norm for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output,
                query_embedding, spatial_position_embedding, active_indices):
        r"""
        :param query_objects: (B, num_queries, d_model), zeros at the first block
        :param encoder_output: (B, num_tokens, d_model)
        :param query_embedding: (B, num_queries, d_model) learned object queries
        :param spatial_position_embedding: (num_tokens, d_model)
        :param active_indices: (B, num_queries) int64, initially arange(num_queries)
        :return: (num_blocks * num_recursions, B, num_queries, d_model) - one
            entry per recursion, for deep supervision.

        NOTE(nb-fidelity): cross-attention weights are computed but not returned
        (the notebook commented the stacking out), so ``MoRDETR`` omits
        'dec_attn' from its output dict.
        """
        decoder_outputs = []
        selected_patches = None
        out = query_objects
        for i in range(self.num_blocks):
            # out: full hidden states (B, num_queries, embed_dim)
            # active_indices: tokens eligible for this step (hierarchical filtering)
            topk = int(((self.num_blocks - i) / self.num_blocks) * out.size(1))

            if i > 0:
                # 1. Gather tokens eligible for this recursion
                active_out = torch.gather(
                    out, 1,
                    active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

                # 2. Routing: select which tokens from the active set continue
                _, rel_indices = self.exp_routers[i](active_out, topk)

                # 3. Map relative indices back to absolute sequence positions
                active_indices = torch.gather(active_indices, 1, rel_indices)

                # 4. Extract selected patches for computation. This focuses
                #    computation only on tokens still active.
                selected_patches = torch.gather(
                    active_out, 1,
                    rel_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

            # Running through the same layers more than once as per MoR
            for _ in range(self.num_recursions):
                if selected_patches is not None:
                    # Routed path: only the selected queries are updated.
                    # Norm, Self Attention, Dropout and Residual
                    in_attn = self.attn_norms[i](selected_patches)
                    temp_pos_embeds = torch.gather(
                        query_embedding, 1,
                        active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
                    q = in_attn + temp_pos_embeds
                    k = in_attn + temp_pos_embeds
                    out_attn, _ = self.attns[i](
                        query=q,
                        key=k,
                        value=in_attn
                    )
                    out_attn = self.attn_dropouts[i](out_attn)
                    selected_patches = selected_patches + out_attn

                    # Norm, Cross Attention, Dropout and Residual
                    in_attn = self.cross_attn_norms[i](selected_patches)
                    q = in_attn + temp_pos_embeds
                    k = encoder_output + spatial_position_embedding
                    out_attn, _ = self.cross_attns[i](
                        query=q,
                        key=k,
                        value=encoder_output
                    )
                    out_attn = self.cross_attn_dropouts[i](out_attn)
                    selected_patches = selected_patches + out_attn

                    # Norm, MLP, Dropout and Residual
                    in_ff = self.ff_norms[i](selected_patches)
                    out_ff = self.ffs[i](in_ff)
                    out_ff = self.ff_dropouts[i](out_ff)
                    selected_patches = selected_patches + out_ff

                    # 7. RECOMBINE: scatter updates back into the full sequence.
                    #    This preserves unselected tokens (the skipped path).
                    # NOTE(nb-fidelity): unlike the encoder, the decoder does NOT
                    # multiply by the router weights here, and it scatters into
                    # `query_objects` (which it then reassigns), so the scatter
                    # accumulates across recursions within a block.
                    query_objects = query_objects.clone()
                    query_objects.scatter_add_(
                        1,
                        active_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                        selected_patches)
                    out = query_objects
                    decoder_outputs.append(self.output_norm(out))
                else:
                    # Dense path: block 0 runs over every query.
                    # Norm, Self Attention, Dropout and Residual
                    in_attn = self.attn_norms[i](out)
                    q = in_attn + query_embedding
                    k = in_attn + query_embedding
                    out_attn, _ = self.attns[i](
                        query=q,
                        key=k,
                        value=in_attn
                    )
                    out_attn = self.attn_dropouts[i](out_attn)
                    out = out + out_attn

                    # Norm, Cross Attention, Dropout and Residual
                    in_attn = self.cross_attn_norms[i](out)
                    q = in_attn + query_embedding
                    k = encoder_output + spatial_position_embedding
                    out_attn, _ = self.cross_attns[i](
                        query=q,
                        key=k,
                        value=encoder_output
                    )
                    out_attn = self.cross_attn_dropouts[i](out_attn)
                    out = out + out_attn

                    # Norm, MLP, Dropout and Residual
                    in_ff = self.ff_norms[i](out)
                    out_ff = self.ffs[i](in_ff)
                    out_ff = self.ff_dropouts[i](out_ff)
                    out = out + out_ff
                    decoder_outputs.append(self.output_norm(out))

        # One output per recursion of every block, i.e. num_blocks * num_recursions.
        output = torch.stack(decoder_outputs)
        return output
