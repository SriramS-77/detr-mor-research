import torchvision.datasets as datasets

import os
import torch
import torch.nn as nn
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image
import yaml
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models
from torchvision.models import resnet34
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import numpy as np
import random

from dataset_loader import *


# print(torch.cuda.is_available())

# print(os.environ.get('CUDA_VISIBLE_DEVICES'))

# print(torch.__version__)

# print(torch.version.cuda)

# ---------------------------------------------------------------------------
# Shared utilities + multi-scale deformable attention.
#
# `MultiScaleDeformableAttention` is NOT part of torchvision (any version) -- it
# lives in mmcv.ops / the original Deformable-DETR CUDA extension. So we own the
# module here and only swap the inner sampling kernel:
#
#   * default: a pure-PyTorch F.grid_sample core. Runs everywhere (Windows CPU,
#     Windows CUDA, Kaggle/Linux). ~2-3x slower than the fused kernel.
#   * fast path: if mmcv or the Deformable-DETR CUDA extension happens to be
#     importable AND the tensors are on CUDA, the fused kernel is used instead.
#     Both expose the identical signature
#         fn(value, spatial_shapes, level_start_index,
#            sampling_locations, attention_weights, im2col_step)
#     so the module, its parameters and its checkpoints are backend-independent.
#
# To get the fast path on Kaggle:  pip install mmcv
# (the 2021 `MultiScaleDeformableAttention` PyPI wheel is ABI-stale against
#  torch 2.x -- prefer mmcv or a fresh build of the Deformable-DETR op.)
# ---------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_MSDA_CUDA_FN = None
_MSDA_BACKEND = 'pytorch (grid_sample)'
try:
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction as _MSDA_CUDA_FN,
    )
    _MSDA_BACKEND = 'mmcv fused CUDA kernel'
except Exception:
    try:
        from MultiScaleDeformableAttention import (
            MSDeformAttnFunction as _MSDA_CUDA_FN,
        )
        _MSDA_BACKEND = 'Deformable-DETR fused CUDA kernel'
    except Exception:
        pass
# print('MSDeformAttn backend: {}'.format(_MSDA_BACKEND))


def inverse_sigmoid(x, eps=1e-5):
    r"""
    Logit of x, with clamping so the log never sees 0 or a negative.

    :param x: tensor expected to lie in [0, 1]
    :param eps: floor applied to both x and 1-x
    :return: tensor of the same shape in unbounded space
    """
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1. - x).clamp(min=eps))


def get_spatial_position_embedding(pos_emb_dim, feat_map):
    r"""
    2D sine/cosine position embedding, one vector per feature-map cell.

    Half the dimensions encode the row index and half the column index, each
    split again into sin and cos - hence the divisible-by-4 requirement.

    :param pos_emb_dim: embedding dimension (must be divisible by 4)
    :param feat_map: (B, C, H, W) tensor; only its shape and device are used
    :return: (H * W, pos_emb_dim) tensor
    """
    assert pos_emb_dim % 4 == 0, ('Position embedding dimension '
                                  'must be divisible by 4')
    grid_size_h, grid_size_w = feat_map.shape[2], feat_map.shape[3]
    grid_h = torch.arange(grid_size_h,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid_w = torch.arange(grid_size_w,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of grid cell tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=feat_map.device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1)
    # grid_h_emb -> (Number of grid cell tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of grid cell tokens, pos_emb_dim)
    return pos_emb


def generate_anchors(spatial_shapes, grid_size=0.05, eps=1e-2,
                     dtype=torch.float32, device='cpu'):
    r"""
    RT-DETR per-level anchor grid, used to turn the encoder bbox head into a
    *delta* predictor instead of an absolute one.

    One anchor per feature-map cell: centre at the cell centre, width/height a
    fixed fraction of the image that doubles with every level. Returned in
    inverse-sigmoid space so the head can simply add its delta before the
    sigmoid. Anchors whose centre or size falls outside (eps, 1-eps) are marked
    invalid and pushed to a large value, so sigmoid saturates them out.

    :param spatial_shapes: (num_levels, 2) tensor of (H, W) per level
    :param grid_size: base anchor side length at level 0, as a fraction of the image
    :param eps: validity margin
    :return: (anchors, valid_mask) of shapes (1, sum(H*W), 4) and (1, sum(H*W), 1)
    """
    anchors = []
    for level, (h, w) in enumerate(spatial_shapes.tolist()):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=dtype, device=device),
            torch.arange(w, dtype=dtype, device=device),
            indexing='ij')
        # Cell centres, normalised to [0, 1]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = (grid_xy + 0.5) / torch.tensor([w, h], dtype=dtype,
                                                 device=device)
        wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** level)
        anchors.append(torch.cat([grid_xy, wh], dim=-1).reshape(-1, 4))

    anchors = torch.cat(anchors, dim=0).unsqueeze(0)
    # anchors -> (1, sum(H*W), 4) in cxcywh, [0, 1]

    valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(dim=-1,
                                                             keepdim=True)
    # valid_mask -> (1, sum(H*W), 1)

    anchors = inverse_sigmoid(anchors)
    # Large-but-finite instead of inf: sigmoid still saturates to 1 and the
    # gradient is 0, but no inf ever enters the arithmetic.
    anchors = torch.where(valid_mask, anchors,
                          torch.full_like(anchors, 1e4))
    return anchors, valid_mask


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    r"""
    Portable reference implementation of multi-scale deformable attention.

    Each level is grid_sample'd independently at the predicted sampling
    locations, then all (level, point) samples are combined by the predicted
    attention weights. Numerically equivalent to the fused CUDA kernel.

    :param value: (B, sum(H*W), num_heads, head_dim)
    :param value_spatial_shapes: (num_levels, 2) tensor of (H, W)
    :param sampling_locations: (B, num_queries, num_heads, num_levels, num_points, 2)
        in [0, 1] normalised (x, y)
    :param attention_weights: (B, num_queries, num_heads, num_levels, num_points)
    :return: (B, num_queries, num_heads * head_dim)
    """
    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    shapes = value_spatial_shapes.tolist()
    value_list = value.split([h * w for h, w in shapes], dim=1)

    # grid_sample wants [-1, 1] coordinates
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(shapes):
        # (B, H*W, heads, head_dim) -> (B*heads, head_dim, H, W)
        value_l = (value_list[level]
                   .flatten(2)
                   .transpose(1, 2)
                   .reshape(batch_size * num_heads, head_dim, h, w))
        # (B, queries, heads, points, 2) -> (B*heads, queries, points, 2)
        grid_l = (sampling_grids[:, :, :, level]
                  .transpose(1, 2)
                  .flatten(0, 1))
        sampling_value_list.append(
            F.grid_sample(value_l, grid_l,
                          mode='bilinear',
                          padding_mode='zeros',
                          align_corners=False)
        )
        # each -> (B*heads, head_dim, queries, points)

    attention_weights = (attention_weights
                         .transpose(1, 2)
                         .reshape(batch_size * num_heads, 1, num_queries,
                                  num_levels * num_points))
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
              * attention_weights).sum(-1)
    # output -> (B*heads, head_dim, queries)
    output = output.view(batch_size, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    r"""
    Multi-scale deformable attention (Deformable-DETR / RT-DETR).

    Instead of attending to every one of the ~8400 encoder tokens, each query
    predicts ``num_points`` sampling offsets per head per level around its
    reference point and attends only to those. Cost is independent of the
    feature-map size, which is what makes the decoder real-time.

    :param d_model: model dimension
    :param num_heads: attention heads
    :param num_levels: number of feature-map levels in the flattened memory
    :param num_points: sampling points per head, per level
    """

    def __init__(self, d_model=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = d_model // num_heads
        self.im2col_step = 64

        self.sampling_offsets = nn.Linear(
            d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""
        The published init, which deformable attention genuinely needs: offsets
        start as a ring of ``num_heads`` directions (growing outwards with the
        point index) so the heads immediately look in different directions, and
        the attention weights start uniform.
        """
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = (torch.arange(self.num_heads, dtype=torch.float32)
                  * (2.0 * math.pi / self.num_heads))
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = (grid_init.view(self.num_heads, 1, 1, 2)
                     .repeat(1, self.num_levels, self.num_points, 1))
        for point in range(self.num_points):
            grid_init[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index):
        r"""
        :param query: (B, num_queries, d_model)
        :param reference_points: (B, num_queries, num_levels, 2 or 4),
            normalised to [0, 1]. 2 -> (cx, cy); 4 -> (cx, cy, w, h), where the
            w/h additionally scale the sampling offsets.
        :param input_flatten: (B, sum(H*W), d_model) flattened multi-scale memory
        :param input_spatial_shapes: (num_levels, 2) tensor of (H, W)
        :param input_level_start_index: (num_levels,) offsets into input_flatten
        :return: (B, num_queries, d_model)
        """
        batch_size, num_queries, _ = query.shape
        _, num_value, _ = input_flatten.shape
        assert int(input_spatial_shapes.prod(dim=1).sum()) == num_value, (
            'spatial_shapes do not account for every token in the memory')

        value = self.value_proj(input_flatten)
        value = value.view(batch_size, num_value, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads,
            self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            batch_size, num_queries, self.num_heads,
            self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            # Offsets are in units of one feature-map cell of their own level
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                dim=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            # Offsets are in units of half the reference box own size
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                'reference_points last dim must be 2 or 4, got {}'.format(
                    reference_points.shape[-1]))

        if _MSDA_CUDA_FN is not None and value.is_cuda:
            output = _MSDA_CUDA_FN.apply(
                value, input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights, self.im2col_step)
        else:
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes,
                sampling_locations, attention_weights)

        return self.output_proj(output)


class TransformerEncoder(nn.Module):
    r"""
    Encoder for transformer of DETR.
    This has sequence of encoder layers.
    Each layer has the following modules:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for MLP
        4. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention Module for all encoder layers
        self.attns = nn.ModuleList(
                [
                    nn.MultiheadAttention(d_model, num_heads,
                                          dropout=self.dropout_prob,
                                          batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        # MLP Module for all encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention for all encoder layers
        self.attn_norms = nn.ModuleList(
                [
                    nn.LayerNorm(d_model)
                    for _ in range(num_layers)
                ])

        # Norm for MLP for all encoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Self Attention for all encoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP for all encoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_position_embedding, return_attn=False):
        r"""
        :param return_attn: also return the per-layer attention maps. Off by
            default: ``need_weights=True`` forces ``nn.MultiheadAttention`` off
            PyTorch's fused scaled-dot-product path and materialises the full
            (B, L, L) matrix on every forward. Nothing in training reads these,
            so they are now computed only when explicitly asked for.
        :return: ``(output, attn_weights)``; ``attn_weights`` is None unless
            ``return_attn``.
        """
        out = x
        attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            # Add spatial position embedding
            # to q,k for self attention
            q = in_attn + spatial_position_embedding
            k = in_attn + spatial_position_embedding
            out_attn, attn_weight = self.attns[i](
                query=q,
                key=k,
                value=in_attn,
                need_weights=return_attn
            )
            if return_attn:
                attn_weights.append(attn_weight)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff

        # Output Normalization
        out = self.output_norm(out)
        return out, (torch.stack(attn_weights) if return_attn else None)



class DeformableCrossAttnBlock(nn.Module):
    r"""
    Thin wrapper around :class:`MSDeformAttn` that broadcasts a single reference
    box per query across all feature levels and applies output dropout.

    :param d_model: model dimension
    :param num_heads: attention heads
    :param dropout: dropout applied to the attention output
    :param num_levels: number of feature-map levels in the flattened memory
    :param num_points: sampling points per head, per level
    """

    def __init__(self, d_model, num_heads, dropout=0.0,
                 num_levels=3, num_points=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.ms_deform_attn = MSDeformAttn(
            d_model=d_model,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index):
        r"""
        :param query: (B, num_queries, d_model)
        :param reference_points: (B, num_queries, 4) cxcywh in [0, 1]
        :param input_flatten: (B, sum(H*W), d_model)
        :param input_spatial_shapes: (num_levels, 2) tensor of (H, W)
        :param input_level_start_index: (num_levels,) offsets into input_flatten
        :return: (B, num_queries, d_model)
        """
        # Deformable attention wants one reference point *per level*. Every
        # image is resized to a fixed square with no padding, so all valid
        # ratios are 1 and the same box can simply be reused at every level.
        reference_points = (reference_points.unsqueeze(2)
                            .expand(-1, -1, self.num_levels, -1))
        # reference_points -> (B, num_queries, num_levels, 4)

        output = self.ms_deform_attn(
            query=query,
            reference_points=reference_points,
            input_flatten=input_flatten,
            input_spatial_shapes=input_spatial_shapes,
            input_level_start_index=input_level_start_index
        )
        return self.dropout(output)


class DeformableTransformerDecoder(nn.Module):
    r"""
    Decoder for the deformable transformer of RT-DETR.
    This has a sequence of decoder layers.
    Each layer has the following modules:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for Deformable Cross Attention on the encoder output
        4. Deformable Cross Attention
        5. LayerNorm for MLP
        6. MLP
        7. A bbox head predicting a refinement of the reference box

    Every layer returns its own output and its own refined boxes, so the caller
    can apply deep supervision across all of them.
    """

    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0, num_levels=3, num_points=4):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout_prob = dropout_prob
        self.num_levels = num_levels

        # Self Attention module for all decoder layers
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # Cross Attention Module for all decoder layers
        self.cross_attns = nn.ModuleList([
            DeformableCrossAttnBlock(d_model, num_heads,
                                     dropout=self.dropout_prob,
                                     num_levels=num_levels,
                                     num_points=num_points)
            for _ in range(num_layers)
        ])

        # MLP Module for all decoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention Module for all decoder layers
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for Cross Attention Module for all decoder layers
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for MLP Module for all decoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Attention Module for all decoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP Module for all decoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Per-layer (not shared) bbox heads, as in Deformable-DETR / RT-DETR
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 4)
            )
            for _ in range(num_layers)
        ])

        # Shared Output norm for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Start every layer as an identity refinement: with a zeroed last layer
        # the predicted delta is 0, so layer 0 reproduces the encoder boxes and
        # the refinement chain has to earn any movement it makes.
        for bbox_head in self.bbox_heads:
            nn.init.constant_(bbox_head[-1].weight, 0.)
            nn.init.constant_(bbox_head[-1].bias, 0.)

    def forward(self, query_objects, query_pos_head, reference_points,
                encoder_output, spatial_shapes, level_start_index,
                attn_mask=None):
        r"""
        :param query_objects: (B, num_queries, d_model) content queries
        :param query_pos_head: module mapping (B, num_queries, 4) reference
            boxes to (B, num_queries, d_model) positional queries. A *module*,
            not a precomputed tensor: RT-DETR / DAB-DETR regenerate the
            positional query from each layer's own refined boxes, so it has to
            be callable inside the loop (see the call site below).
        :param reference_points: (B, num_queries, 4) cxcywh in [0, 1], the
            top-k boxes selected from the encoder
        :param encoder_output: (B, sum(H*W), d_model) flattened multi-scale memory
        :param spatial_shapes: (num_levels, 2) tensor of (H, W) per level
        :param level_start_index: (num_levels,) offsets into encoder_output
        :param attn_mask: optional (L, L) bool self-attention mask, where True
            means "may not attend". Used by CDN to stop the ordinary matching
            queries reading the answer off the denoising queries. Only the
            self-attention needs it - queries do not attend to one another in
            the deformable cross-attention.
        :return: ``(outputs, reference_points)`` stacked over layers, of shapes
            (num_layers, B, num_queries, d_model) and (num_layers, B, num_queries, 4)
        """
        out = query_objects
        decoder_outputs = []
        updated_reference_points = []

        for i in range(self.num_layers):
            # The positional query is regenerated from the reference boxes this
            # layer is actually starting from. Computing it once before the loop
            # would describe the ENCODER's boxes at every layer, so layers 2..N
            # would be positioned by boxes they have already refined away from -
            # which is precisely the signal iterative refinement depends on.
            query_pos_embed = query_pos_head(reference_points)

            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_pos_embed
            k = in_attn + query_pos_embed
            # need_weights=False: nothing here consumes the decoder's attention
            # maps, and leaving it True blocks the fused attention kernel.
            out_attn, _ = self.attns[i](
                query=q,
                key=k,
                value=in_attn,
                attn_mask=attn_mask,
                need_weights=False
            )
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, Deformable Cross Attention, Dropout and Residual.
            # The cross-attention block owns its own output dropout.
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_pos_embed
            out_attn = self.cross_attns[i](
                query=q,
                reference_points=reference_points,
                input_flatten=encoder_output,
                input_spatial_shapes=spatial_shapes,
                input_level_start_index=level_start_index
            )
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff
            layer_out = self.output_norm(out)
            decoder_outputs.append(layer_out)

            # ITERATIVE REFERENCE POINT REFINEMENT
            # Predict the offset (delta) from the current layer output
            delta_unsig = self.bbox_heads[i](layer_out)

            # Convert current reference points to inverse sigmoid space, add
            # delta, and sigmoid back to [0, 1] coordinate space.
            ref_unsig = inverse_sigmoid(reference_points)
            new_reference_points = (ref_unsig + delta_unsig).sigmoid()

            # Keep the *undetached* boxes for the loss...
            updated_reference_points.append(new_reference_points)
            # ...but detach the ones handed to the next layer, so gradients do
            # not flow back through the whole iterative update chain.
            reference_points = new_reference_points.detach()

        # Stack outputs for deep supervision loss computation
        output = torch.stack(decoder_outputs)
        ref_points_out = torch.stack(updated_reference_points)
        return output, ref_points_out



# ---------------------------------------------------------------------------
# CDN - Contrastive DeNoising (DINO / RT-DETR).
#
# Why: Hungarian matching is recomputed every forward pass, so early in training
# the same GT box gets assigned to a different query each step and the gradients
# partly cancel. CDN sidesteps matching entirely for an extra set of queries: it
# feeds in *noised copies of the ground truth* whose assignment is known by
# construction, giving a dense, stable regression signal alongside the matching
# branch.
#
# The "contrastive" part: every GT produces TWO noised queries -
#   positive (small noise)  -> target = true class + true box
#   negative (large noise)  -> target = background
# Plain denoising only ever says "pull toward the GT", so the model learns to
# fire on anything nearby and emits duplicates. The negatives teach an explicit
# reject boundary, which is what lets RT-DETR drop NMS.
#
# Training only - at inference no DN queries are built, so the cost is zero.
# ---------------------------------------------------------------------------


def get_contrastive_denoising_training_group(targets, num_classes, num_queries,
                                             class_embed, num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,
                                             device='cpu'):
    r"""
    Build the contrastive denoising query group for one batch.

    :param targets: list of B dicts with 'boxes' (N_i, 4) xyxy normalised and
        'labels' (N_i,)
    :param num_classes: foreground classes (VFL has no background class);
        index ``num_classes`` is used as the padding class, hence the embedding
        has ``num_classes + 1`` rows
    :param num_queries: number of ordinary matching queries (for the attn mask)
    :param class_embed: nn.Embedding(num_classes + 1, d_model)
    :param num_denoising: target DN query budget; the realised count is
        ``2 * num_group * max_gt_num``, which lands near ``2 * num_denoising``
    :param label_noise_ratio: fraction of DN labels flipped to a random class
    :param box_noise_scale: box jitter, in units of half the box's own w/h
    :return: ``(dn_content, dn_ref_points, attn_mask, dn_meta)``, or four Nones
        when there is nothing to denoise. ``dn_ref_points`` is cxcywh in [0, 1].
    """
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(target['labels']) for target in targets]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        # Nothing to denoise in this batch (every image is empty).
        return None, None, None, None

    batch_size = len(targets)
    # Each group reconstructs every GT once. More groups = more denoising
    # samples per image, without them being able to collude (see attn mask).
    num_group = max(1, num_denoising // max_gt_num)

    # --- 1. Pad the ragged GT into dense (B, max_gt_num) tensors -------------
    # Slots past an image's own GT count hold the padding class and a zero box;
    # they end up unmatched, so they simply train toward background.
    input_query_class = torch.full([batch_size, max_gt_num], num_classes,
                                   dtype=torch.long, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool,
                              device=device)
    for i, target in enumerate(targets):
        n = num_gts[i]
        if n > 0:
            input_query_class[i, :n] = target['labels'].to(device)
            # CDN operates in cxcywh; our targets are stored as xyxy.
            input_query_bbox[i, :n] = torchvision.ops.box_convert(
                target['boxes'].to(device), 'xyxy', 'cxcywh')
            pad_gt_mask[i, :n] = True

    # --- 2. Tile into groups, each [positives | negatives] ------------------
    # Layout along the query axis: [pos(max_gt) neg(max_gt)] repeated num_group
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    # 1.0 marks the negative half of each group, 0.0 the positive half
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1],
                                   device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])

    # Positive AND a real (non-padding) GT -> these are the queries the loss
    # supervises against an actual target.
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * pad_gt_mask

    # Query-axis indices of those positives, split back out per image.
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx,
                                  [n * num_group for n in num_gts])

    num_denoising_total = int(max_gt_num * 2 * num_group)

    # --- 3. Label noise -----------------------------------------------------
    # Flip some INPUT labels to a random class while the target stays the true
    # class, so the decoder learns to correct a wrong class hypothesis.
    # VFL: every index 0..num_classes-1 is a foreground class (no background
    # column), so the full range is sampled. Under the old softmax head this
    # started at 1 to skip background at index 0.
    if label_noise_ratio > 0:
        mask = torch.rand(input_query_class.shape, device=device) < (
            label_noise_ratio * 0.5)
        new_label = torch.randint_like(input_query_class, 0, num_classes)
        input_query_class = torch.where(mask & pad_gt_mask, new_label,
                                        input_query_class)

    # --- 4. Box noise - the contrastive part --------------------------------
    if box_noise_scale > 0:
        known_bbox = torchvision.ops.box_convert(input_query_bbox,
                                                 'cxcywh', 'xyxy')
        # Jitter budget: half the box's own width/height, times the scale.
        diff = (input_query_bbox[..., 2:] * 0.5).tile([1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0

        rand_part = torch.rand_like(input_query_bbox)          # [0, 1)
        # Negatives get pushed into [1, 2) - strictly outside the positives'
        # range. That hard gap IS the decision boundary the model is taught:
        # inside -> report the object, outside -> report background.
        rand_part = (rand_part + 1.0) * negative_gt_mask + \
                    rand_part * (1 - negative_gt_mask)
        rand_part = rand_part * rand_sign

        known_bbox = (known_bbox + rand_part * diff).clip(min=0.0, max=1.0)
        input_query_bbox = torchvision.ops.box_convert(known_bbox,
                                                       'xyxy', 'cxcywh')
        # A negative can be displaced far enough to flip (x2 < x1), giving a
        # negative width. Upstream RT-DETR masks this by inverse_sigmoid'ing
        # before returning (which clamps to [0,1]); we return raw cxcywh, so
        # clamp here instead. Only negatives can flip - at box_noise_scale 1.0
        # a positive bottoms out at exactly zero width - and a collapsed
        # negative box is still a perfectly good "reject this" sample.
        input_query_bbox = input_query_bbox.clip(min=0.0, max=1.0)

    # --- 5. Content queries from the (noised) class labels ------------------
    dn_content = class_embed(input_query_class)
    # dn_content -> (B, num_denoising_total, d_model)
    # dn_ref_points are already cxcywh in [0, 1]; our decoder inverse-sigmoids
    # them itself, so unlike upstream RT-DETR we do NOT pre-apply it here.
    dn_ref_points = input_query_bbox

    # --- 6. Attention mask --------------------------------------------------
    # DN queries carry the answer, so leakage has to be blocked. True = masked.
    #
    #                 dn grp 0   dn grp 1   matching
    #   dn group 0  |    .     |  BLOCK   |    .     |
    #   dn group 1  |  BLOCK   |    .     |    .     |
    #   matching    |  BLOCK   |  BLOCK   |    .     |
    #
    # Matching queries must not see any DN query (they would read off the GT).
    # DN groups must not see each other (each is an independent reconstruction).
    tgt_size = num_denoising_total + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool,
                           device=device)
    # matching queries cannot attend to the reconstruction
    attn_mask[num_denoising_total:, :num_denoising_total] = True
    # groups cannot attend to one another
    for i in range(num_group):
        lo, hi = max_gt_num * 2 * i, max_gt_num * 2 * (i + 1)
        attn_mask[lo:hi, :lo] = True
        attn_mask[lo:hi, hi:num_denoising_total] = True

    dn_meta = {
        'dn_positive_idx': dn_positive_idx,
        'dn_num_group': num_group,
        # split sizes used to peel the DN part back off the decoder output
        'dn_num_split': [num_denoising_total, num_queries],
    }
    return dn_content, dn_ref_points, attn_mask, dn_meta


def build_dn_match_indices(dn_meta, targets):
    r"""
    The known DN assignment, in the same ``(pred_idx, target_idx)`` form the
    Hungarian matcher returns - so the loss code is shared verbatim.

    No matching is involved: within group g, positive slot j simply *is* GT j.
    Negatives and padding slots are deliberately left out, so they fall through
    to the loss's background default, which is exactly the target CDN wants.

    :return: list of B ``(pred_idx, target_idx)`` int64 tensor pairs
    """
    num_group = dn_meta['dn_num_group']
    match_indices = []
    for i, target in enumerate(targets):
        n = len(target['labels'])
        pred_idx = dn_meta['dn_positive_idx'][i].cpu()
        # Positives are emitted group-major, so GT indices just repeat per group
        target_idx = torch.arange(n, dtype=torch.int64).repeat(num_group)
        match_indices.append((pred_idx.to(torch.int64), target_idx))
    return match_indices



# Backbones the multi-scale forward below understands. Any torchvision resnet
# works: the forward only needs .conv1/.bn1/.relu/.maxpool/.layer1..4, and the
# per-stage channel widths are probed at build time rather than hardcoded.
SUPPORTED_BACKBONES = ('resnet18', 'resnet34', 'resnet50', 'resnet101',
                       'resnet152')


def fuse_conv_bn_weights(conv_weight, bn):
    r"""
    Fold a BatchNorm into the conv that precedes it.

    At inference BN is affine: ``y = gamma * (x - mu) / sqrt(var + eps) + beta``.
    The conv before it is affine too, so the pair collapses into a single conv
    with a bias: scale each output filter by ``gamma / sqrt(var + eps)`` and set
    ``bias = beta - mu * gamma / sqrt(var + eps)``.

    :param conv_weight: (out_ch, in_ch, k, k) conv kernel, already 3x3-padded
    :param bn: the BatchNorm2d to fold in
    :return: ``(weight, bias)``
    """
    std = (bn.running_var + bn.eps).sqrt()
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    return conv_weight * scale, bn.bias - bn.running_mean * bn.weight / std


class RepBlock(nn.Module):
    r"""
    RepVGG block: a 3x3 and a 1x1 branch (plus an identity branch when the
    channel count allows), summed and activated. At inference the three branches
    re-parameterise into a single 3x3 conv - see :meth:`convert_to_deploy`.

    Only used inside :class:`CSPRepLayer`, which always instantiates it square
    (in == out), so the identity branch is live. SiLU to match RT-DETR.

    :param in_channels: input channels
    :param out_channels: output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_conn = (nn.BatchNorm2d(out_channels)
                          if in_channels == out_channels else None)

        self.act = nn.SiLU()
        self.deployed = False

    def forward(self, x):
        if self.deployed:
            # A single fused 3x3 - the whole point of the RepVGG parameterisation
            return self.act(self.conv_fused(x))

        out1 = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(x))
        out3 = self.skip_conn(x) if self.skip_conn is not None else 0

        out = self.act(out1 + out2 + out3)
        return out

    def _branch_kernel_bias(self, conv, bn):
        r"""
        One branch, folded to an equivalent 3x3 kernel + bias.

        ``conv=None`` means the identity branch: a pass-through IS a 3x3 kernel,
        one that is 1 at the centre tap of each channel's own input channel and
        0 everywhere else, so it folds like any other conv.
        """
        if conv is None:
            channels = bn.num_features
            kernel = torch.zeros(channels, channels, 3, 3,
                                 device=bn.weight.device, dtype=bn.weight.dtype)
            idx = torch.arange(channels)
            kernel[idx, idx, 1, 1] = 1.0
        else:
            kernel = conv.weight
            if kernel.shape[-1] == 1:
                # Zero-pad the 1x1 out to 3x3 so the branches can be summed
                kernel = torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        return fuse_conv_bn_weights(kernel, bn)

    @torch.no_grad()
    def convert_to_deploy(self):
        r"""
        Collapse the three training-time branches into one 3x3 conv.

        All three branches are linear and see the same input, so their sum is
        itself a single convolution: pad the 1x1 out to 3x3, express the
        identity as a centre-tap 3x3, fold each branch's BatchNorm in, and add
        the kernels.

        Numerically identical in eval mode, but a third of the convolutions and
        no BatchNorm reads. Without this the notebook paid the full training
        cost of CSPRepLayer and got none of the deployment benefit - a fused
        RepBlock is what makes the block cheaper than the plain conv stack it
        replaced, not more expensive.

        **Irreversible and inference-only.** It destroys the training branches
        and changes the state_dict, so fuse a freshly loaded copy for
        evaluation, never a model you intend to keep training.
        """
        if self.deployed:
            return

        kernel, bias = self._branch_kernel_bias(self.conv1, self.bn1)
        kernel_1x1, bias_1x1 = self._branch_kernel_bias(self.conv2, self.bn2)
        kernel, bias = kernel + kernel_1x1, bias + bias_1x1

        if self.skip_conn is not None:
            kernel_id, bias_id = self._branch_kernel_bias(None, self.skip_conn)
            kernel, bias = kernel + kernel_id, bias + bias_id

        fused = nn.Conv2d(self.conv1.in_channels, self.conv1.out_channels,
                          kernel_size=3, stride=1, padding=1, bias=True)
        fused.weight.copy_(kernel)
        fused.bias.copy_(bias)

        for name in ('conv1', 'bn1', 'conv2', 'bn2', 'skip_conn'):
            delattr(self, name)
        self.conv_fused = fused.to(device=kernel.device, dtype=kernel.dtype)
        self.deployed = True


class ConvNormLayer(nn.Module):
    r"""Conv -> BatchNorm -> optional activation, as in RT-DETR."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None,
                 act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else act
        self.deployed = False

    def forward(self, x):
        if self.deployed:
            return self.act(self.conv(x))
        return self.act(self.norm(self.conv(x)))

    @torch.no_grad()
    def convert_to_deploy(self):
        r"""Fold the BatchNorm into the conv. Same caveats as RepBlock's."""
        if self.deployed:
            return
        weight, bias = fuse_conv_bn_weights(self.conv.weight, self.norm)
        fused = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                          self.conv.kernel_size, self.conv.stride,
                          self.conv.padding, bias=True)
        fused.weight.copy_(weight)
        fused.bias.copy_(bias)
        delattr(self, 'norm')
        self.conv = fused.to(device=weight.device, dtype=weight.dtype)
        self.deployed = True


class CSPRepLayer(nn.Module):
    r"""
    RT-DETR's CCFM fusion block, replacing a single wide RepBlock.

    Note this is NOT the classic CSPNet channel split: both branches project the
    *full* input through their own 1x1 conv, and the two are **summed** (not
    concatenated). conv1/conv2 do the ``2*d_model -> d_model`` reduction up
    front, which is what lets the inner RepVGG blocks all be square - and so
    lets their identity branch actually fire, unlike a lone RepBlock(512, 256).

    :param in_channels: input channels (the concatenated fusion input)
    :param out_channels: output channels
    :param num_blocks: RepVGG blocks on the refinement branch
    :param expansion: hidden width as a fraction of out_channels
    """

    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1,
                                   act=nn.SiLU())
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1,
                                   act=nn.SiLU())
        self.bottlenecks = nn.Sequential(*[
            RepBlock(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])
        self.conv3 = (ConvNormLayer(hidden_channels, out_channels, 1, 1,
                                    act=nn.SiLU())
                      if hidden_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class PositionEmbedding(nn.Module):
    r"""
    Sine/cosine embedding of a box, applied independently to each of its four
    normalised coordinates.

    :param num_pos_feats: features generated per coordinate
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        # x -> (B, num_queries, 4) with (cx, cy, w, h) normalised to [0, 1]
        scale = 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,
                             device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        x_sine = x.unsqueeze(-1) * scale / dim_t

        # Apply sin to even indices and cos to odd indices
        pos = torch.stack((x_sine[..., 0::2].sin(),
                           x_sine[..., 1::2].cos()), dim=-1).flatten(2)
        return pos  # -> (B, num_queries, 4 * num_pos_feats)


class RefPointsToQueryEmbed(nn.Module):
    r"""
    Turns reference boxes into positional queries, DAB-DETR style: sinusoidal
    encoding of (cx, cy, w, h) followed by a small MLP.

    :param embed_dim: output dimension (d_model)
    :param num_pos_feats: sine features per coordinate
    """

    def __init__(self, embed_dim=256, num_pos_feats=None):
        super().__init__()
        # Default to embed_dim // 4 so the four coordinates together produce
        # exactly embed_dim sine features - keeps the sine resolution tied to
        # d_model instead of being fixed at 64 whatever d_model is set to.
        if num_pos_feats is None:
            num_pos_feats = embed_dim // 4
        self.pos_emb = PositionEmbedding(num_pos_feats=num_pos_feats)
        # 4 coordinates x num_pos_feats each
        self.mlp = nn.Sequential(
            nn.Linear(4 * num_pos_feats, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, reference_points):
        # 1. Generate sine/cosine positional features from reference points
        pos_embed = self.pos_emb(reference_points)
        # 2. Refine through an MLP
        return self.mlp(pos_embed)


class DETR(nn.Module):
    r"""
    RT-DETR. A forward pass goes through the following stages:
        1. Backbone (ResNet-34), taking S3 / S4 / S5
        2. 1x1 projection of each level to d_model
        3. AIFI - self attention on S5 only
        4. CCFM - top-down then bottom-up cross-scale fusion
        5. IoU-aware query selection - top-k encoder tokens become queries
        6. Deformable decoder with iterative box refinement
        7. Hungarian matching + set-prediction loss, per decoder layer plus the
           encoder selection head

    :param config: the ``model_params`` block of the config
    :param num_classes: foreground classes only - VFL is a per-class sigmoid
        loss, so there is no background class
    """

    def __init__(self, config, num_classes):
        super().__init__()
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_decoder_layers = config['decoder_num_blocks']
        # Matcher costs and loss weights are NOT the same numbers in
        # RT-DETR: the matcher uses cls 2.0 to make the assignment decisive,
        # while the loss uses 1.0 so classification does not swamp the box
        # terms. They were previously sharing one value.
        self.cls_cost_weight = config['cls_cost_weight']
        self.cls_loss_weight = config.get('cls_loss_weight', 1.0)
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.nms_threshold = config['nms_threshold']
        # VFL hyperparameters (see self.varifocal_loss)
        self.vfl_alpha = config.get('vfl_alpha', 0.75)
        self.vfl_gamma = config.get('vfl_gamma', 2.0)
        self.num_levels = 3

        # Backbone, selected by name from the config. Keep the resnet module
        # itself (not an nn.Sequential wrapper) so the named stages stay
        # reachable for the multi-scale forward below.
        self.backbone_name = config.get('backbone', 'resnet34')
        assert self.backbone_name in SUPPORTED_BACKBONES, (
            'backbone must be one of {}, got {}'.format(
                sorted(SUPPORTED_BACKBONES), self.backbone_name))
        backbone = torchvision.models.get_model(
            self.backbone_name,
            # 'DEFAULT' picks the best available ImageNet weights for each
            # model (V2 where torchvision ships one, e.g. resnet50).
            weights='DEFAULT' if config.get('pretrained_backbone', True) else None,
            norm_layer=torchvision.ops.FrozenBatchNorm2d,
        )
        del backbone.avgpool
        del backbone.fc
        self.backbone = backbone

        if config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Stage widths differ by block type: the BasicBlock nets (18/34) give
        # 128/256/512, the Bottleneck nets (50/101/152) give 512/1024/2048.
        # Probe once with a dummy input rather than hardcoding a lookup table -
        # it cannot fall out of sync with whatever torchvision actually builds.
        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64)
            backbone_channels = [feat.shape[1]
                                 for feat in self.backbone_forward(probe)]

        # RT-DETR's input_proj is Conv(bias=False) + BatchNorm, not a bare
        # conv. The three backbone stages come out with very different
        # activation scales, and the neck concatenates them; without the norm
        # the CCFM sees S3/S4/S5 on incomparable scales and the fusion blocks
        # have to spend capacity undoing that.
        self.backbone_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.d_model),
            )
            for channels in backbone_channels  # S3 /8, S4 /16, S5 /32
        ])

        # AIFI: intra-scale attention, on the highest level only
        self.s5_encoder = TransformerEncoder(
            num_layers=config['encoder_num_blocks'],
            num_heads=config['encoder_attn_heads'],
            d_model=config['d_model'],
            ff_inner_dim=config['ff_inner_dim'],
            dropout_prob=config['dropout_prob'])

        # ---- CCFM. These MUST be ModuleLists: as plain python lists their
        # parameters are invisible to .parameters(), .to(device) and state_dict.
        self.up_sampling = nn.ModuleList([
            nn.ConvTranspose2d(self.d_model, self.d_model, kernel_size=2,
                               stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(self.d_model, self.d_model, kernel_size=2,
                               stride=2, padding=0, bias=False),
        ])

        self.down_sampling = nn.ModuleList([
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3,
                      stride=2, padding=1, bias=False),
        ])

        # RT-DETR's fpn_blocks / pan_blocks. The concat feeds 2*d_model in; the
        # CSP layer's two 1x1 convs do the reduction back to d_model.
        csp_blocks = config.get('csp_num_blocks', 3)
        csp_expansion = config.get('csp_expansion', 1.0)
        csp = lambda: CSPRepLayer(in_channels=2 * self.d_model,
                                  out_channels=self.d_model,
                                  num_blocks=csp_blocks,
                                  expansion=csp_expansion)

        self.rep_blocks_up = nn.ModuleList([csp(), csp()])
        self.rep_blocks_down = nn.ModuleList([csp(), csp()])

        # ---- Query selection heads, run over every encoder token.
        # RT-DETR projects + LayerNorms the memory before the selection heads.
        # Not cosmetic: the decoder's logits come off a LayerNorm, but the CCFM
        # output is raw conv activations. Without this the encoder logits have
        # an uncontrolled scale, and since top-k selects the MAX-logit tokens it
        # picks exactly the saturated ones - making the VFL encoder term dwarf
        # every decoder term and dominate the gradient.
        self.enc_output = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.enc_class_head = nn.Linear(self.d_model, self.num_classes)
        self.enc_bbox_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4),
        )
        # Zeroed last layer => the encoder starts by predicting the anchors
        # themselves, and has to learn any deviation from them.
        nn.init.constant_(self.enc_bbox_head[-1].weight, 0.)
        nn.init.constant_(self.enc_bbox_head[-1].bias, 0.)

        # Detections reported per image at inference. RT-DETR takes the top-k
        # (query, class) PAIRS, so k is decoupled from num_queries - see the
        # decode block in forward().
        self.num_top_queries = config.get('num_top_queries', 100)

        # Positional queries derived from the reference boxes. Owned here, but
        # handed to the decoder as a module so it can be re-applied to each
        # layer's own refined boxes rather than only to the encoder's.
        self.ref_to_query_embed = RefPointsToQueryEmbed(embed_dim=self.d_model)

        self.decoder = DeformableTransformerDecoder(
            num_layers=config['decoder_num_blocks'],
            num_heads=config['decoder_attn_heads'],
            d_model=config['d_model'],
            ff_inner_dim=config['ff_inner_dim'],
            dropout_prob=config['dropout_prob'],
            num_levels=self.num_levels,
            num_points=config.get('decoder_num_points', 4)
        )

        # One classification head PER decoder layer, as in RT-DETR
        # (dec_score_head), matching the per-layer box heads that already live
        # inside the decoder. This matters more under VFL than it did under
        # cross-entropy: a matched query's VFL target is *that layer's own*
        # achieved IoU, so layer 1 is trained toward low scores and layer 6
        # toward high ones. A single shared head would have to be calibrated
        # for both target distributions simultaneously.
        self.class_heads = nn.ModuleList([
            nn.Linear(self.d_model, self.num_classes)
            for _ in range(self.num_decoder_layers)
        ])

        # ---- CDN (contrastive denoising), training only.
        self.num_denoising = config.get('num_denoising', 0)
        self.label_noise_ratio = config.get('label_noise_ratio', 0.5)
        self.box_noise_scale = config.get('box_noise_scale', 1.0)
        if self.num_denoising > 0:
            # One row per class, plus a final row used for the padding slots
            # that appear when images in a batch have different GT counts.
            self.denoising_class_embed = nn.Embedding(self.num_classes + 1,
                                                      self.d_model)
        else:
            self.denoising_class_embed = None

        # VFL: sigmoid classification heads must start pessimistic. With a
        # zero bias every one of the ~8400 encoder tokens starts at sigmoid(0)
        # = 0.5 for all 20 classes, and since almost all of them are negatives
        # the initial loss is enormous and the first steps destabilise. RetinaNet's
        # fix, which RT-DETR inherits: bias = -log((1 - p) / p) for a prior
        # p = 0.01, so every class starts at ~1% confidence.
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        for class_head in self.class_heads:
            nn.init.constant_(class_head.bias, bias_init)
        nn.init.constant_(self.enc_class_head.bias, bias_init)

        # Anchors depend only on the feature-map shapes, so cache them.
        self._anchor_cache = {}

    @torch.no_grad()
    def convert_to_deploy(self):
        r"""
        Re-parameterise every RepVGG block and fold every ConvNormLayer's
        BatchNorm away, for inference.

        This is the half of the RepVGG bargain that makes CSPRepLayer worth
        having: multi-branch while training for the gradient flow, a single
        fused 3x3 at deployment for the speed. Skipping it leaves the block
        strictly *slower* than the plain conv stack it replaced.

        **Irreversible.** The fused model cannot be trained further and its
        state_dict no longer matches a freshly constructed one, so call this on
        a copy loaded for evaluation and never checkpoint the result.

        :return: number of modules fused
        """
        fused = 0
        # list() first: convert_to_deploy rewrites each module's children while
        # we walk them, and .modules() is a lazy generator over those dicts.
        for module in list(self.modules()):
            if isinstance(module, (RepBlock, ConvNormLayer)):
                module.convert_to_deploy()
                fused += 1
        return fused

    def backbone_forward(self, x):
        r"""
        :param x: (B, 3, H, W)
        :return: S3 (stride 8, 128ch), S4 (stride 16, 256ch), S5 (stride 32, 512ch)
        """
        backbone = self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        s3 = backbone.layer2(x)   # stride 8,  128 channels
        s4 = backbone.layer3(s3)  # stride 16, 256 channels
        s5 = backbone.layer4(s4)  # stride 32, 512 channels

        return s3, s4, s5

    def get_anchors(self, spatial_shapes, device, dtype):
        r"""Cached :func:`generate_anchors` keyed on shape + device."""
        key = (tuple(spatial_shapes.flatten().tolist()), str(device), dtype)
        if key not in self._anchor_cache:
            self._anchor_cache[key] = generate_anchors(
                spatial_shapes, dtype=dtype, device=device)
        return self._anchor_cache[key]

    @torch.no_grad()
    def match(self, cls_output, bbox_output, targets):
        r"""
        Hungarian matching: the minimum-cost one-to-one assignment between
        queries and ground-truth objects, per image.

        Split out from :meth:`set_loss` so CDN can reuse the loss half below
        with its own known assignment instead of running the matcher.

        :param cls_output: (B, num_queries, num_classes) logits
        :param bbox_output: (B, num_queries, 4) cxcywh in [0, 1]
        :param targets: list of B dicts with 'boxes' (N_i, 4) xyxy normalised
            and 'labels' (N_i,)
        :return: list of B ``(pred_idx, target_idx)`` int64 tensor pairs
        """
        batch_size, num_queries, num_classes = cls_output.shape

        # VFL: sigmoid per class, not softmax over classes.
        class_prob = cls_output.reshape((-1, num_classes)).sigmoid()
        # class_prob -> (B*num_queries, num_classes)

        pred_boxes = bbox_output.reshape((-1, 4))
        # pred_boxes -> (B*num_queries, 4)

        # Concat all target boxes and labels also together
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])
        # len(target_labels) -> num_targets_for_entire_batch
        # target_boxes -> (num_targets_for_entire_batch, 4)

        # VFL: focal-style classification cost, replacing the softmax matcher's
        # plain -p. The cost of calling a query class c is the focal loss it
        # would incur as a positive, minus what it would incur as a negative -
        # so the matcher scores candidates the same way the loss will. Leaving
        # the old -softmax_prob here would have the matcher and the loss
        # disagree about what a good classification looks like.
        neg_cost = ((1 - self.vfl_alpha) * class_prob.pow(self.vfl_gamma)
                    * -(1 - class_prob + 1e-8).log())
        pos_cost = (self.vfl_alpha * (1 - class_prob).pow(self.vfl_gamma)
                    * -(class_prob + 1e-8).log())
        cost_classification = (pos_cost[:, target_labels]
                               - neg_cost[:, target_labels])
        # cost_cls -> (B*num_queries, num_targets_for_entire_batch)

        # We predict cx,cy,w,h , we need to convert to x1y1x2y2 for giou.
        # Don't need to convert targets as they are already in x1y1x2y2
        pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
            pred_boxes, 'cxcywh', 'xyxy')

        # L1 in cxcywh, as in the paper - NOT xyxy. In xyxy a shift of the box
        # centre moves both corners, so the same displacement is counted twice
        # and the paper's l1 weight of 5.0 ends up worth roughly double what it
        # is meant to be relative to GIoU. GIoU below still needs xyxy.
        target_boxes_cxcywh = torchvision.ops.box_convert(
            target_boxes, 'xyxy', 'cxcywh')

        cost_localization_l1 = torch.cdist(
            pred_boxes,
            target_boxes_cxcywh,
            p=1
        )
        # cost_l1 -> (B*num_queries, num_targets_for_entire_batch)

        cost_localization_giou = -torchvision.ops.generalized_box_iou(
            pred_boxes_x1y1x2y2,
            target_boxes
        )
        # cost_giou -> (B*num_queries, num_targets_for_entire_batch)

        total_cost = (self.l1_cost_weight * cost_localization_l1
                      + self.cls_cost_weight * cost_classification
                      + self.giou_cost_weight * cost_localization_giou)

        # linear_sum_assignment is scipy, so the cost has to come back to cpu
        total_cost = total_cost.reshape(batch_size, num_queries, -1).cpu()
        # total_cost -> (B, num_queries, num_targets_for_entire_batch)

        num_targets_per_image = [len(target["labels"]) for target in targets]
        total_cost_per_batch_image = total_cost.split(
            num_targets_per_image,
            dim=-1
        )
        # total_cost_per_batch_image[i] = (B,num_queries,num_targets_ith_image)

        match_indices = []
        for batch_idx in range(batch_size):
            batch_idx_assignments = linear_sum_assignment(
                total_cost_per_batch_image[batch_idx][batch_idx]
            )
            batch_idx_pred, batch_idx_target = batch_idx_assignments
            # len(batch_idx_pred) = num_targets_ith_image
            match_indices.append((torch.as_tensor(batch_idx_pred,
                                                  dtype=torch.int64),
                                  torch.as_tensor(batch_idx_target,
                                                  dtype=torch.int64)))
            # match_indices -> [
            #   ([pred_box_a1, ...],[target_box_i1, ...]),
            #   ([pred_box_a2, ...],[target_box_i2, ...]),
            #   ... assignment pairs for ith batch image
            #   ]
        return match_indices

    def varifocal_loss(self, cls_output, target_score, target_onehot,
                       num_boxes):
        r"""
        VariFocal Loss (VarifocalNet), the classification loss RT-DETR uses.

        ``VFL(p, q) = -q * (q*log(p) + (1-q)*log(1-p))``   for a positive (q>0)
        ``VFL(p, q) = -alpha * p^gamma * log(1-p)``        for a negative (q=0)

        The asymmetry is the whole point. Negatives get the usual focal
        down-weighting, which handles the fact that almost every query is
        background. Positives are instead weighted by ``q`` - their own IoU -
        and are deliberately NOT focal-attenuated: they are rare and valuable,
        so rather than damping the easy ones, VFL makes a well-localised
        positive contribute more gradient than a marginal one. The model
        therefore learns to reserve high scores for well-localised boxes, which
        is precisely the signal IoU-aware query selection needs.

        Compare: plain BCE asks "is it a car?". VFL asks "is it a car, and how
        well have I boxed it?".

        :param cls_output: (B, num_queries, num_classes) raw logits
        :param target_score: (B, num_queries, num_classes), IoU at the matched
            (query, class) cells and 0 everywhere else
        :param target_onehot: same shape, 1.0 at the matched cells
        :param num_boxes: normalizer (matched box count, clamped to >= 1)
        :return: scalar
        """
        # Detached: the focal weight modulates the gradient, it is not itself
        # something to backprop through.
        pred_score = cls_output.sigmoid().detach()
        # Negatives (and the non-target classes of a matched query) get the
        # focal term; the matched class gets its IoU as the weight instead.
        weight = (self.vfl_alpha * pred_score.pow(self.vfl_gamma)
                  * (1 - target_onehot) + target_score)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cls_output, target_score, weight=weight, reduction='none')
        return loss.sum() / num_boxes

    def losses_from_indices(self, cls_output, bbox_output, targets,
                            match_indices):
        r"""
        The DETR set-prediction loss, given an assignment.

        Every query not named in ``match_indices`` is trained toward the
        background class. For CDN that is exactly the right default: its
        negative and padding queries are simply left out of the indices.

        :param cls_output: (B, num_queries, num_classes) logits
        :param bbox_output: (B, num_queries, 4) cxcywh in [0, 1]
        :param targets: list of B target dicts
        :param match_indices: list of B ``(pred_idx, target_idx)`` pairs
        :return: ``(loss_cls, loss_bbox, loss_giou)`` scalars
        """
        # pred_batch_idxs are batch indexes for each assignment pair
        pred_batch_idxs = torch.cat([
            torch.ones_like(pred_idx) * i
            for i, (pred_idx, _) in enumerate(match_indices)
        ])
        # pred_batch_idxs -> (num_targets_for_entire_batch, )

        # pred_query_idx are prediction box indexes(out of num_queries)
        # for each assignment pair
        pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])
        # pred_query_idx -> (num_targets_for_entire_batch, )

        # For all assigned prediction boxes, get the target label
        valid_obj_target_cls = torch.cat([
            target["labels"][target_obj_idx]
            for target, (_, target_obj_idx) in zip(targets, match_indices)
        ])
        # valid_obj_target_cls -> (num_targets_for_entire_batch, )

        # Get pred box coordinates for all matched pred boxes
        matched_pred_boxes = bbox_output[pred_batch_idxs, pred_query_idx]
        # matched_pred_boxes -> (num_targets_for_entire_batch, 4)

        # Get target box coordinates for all matched target boxes
        matched_target_boxes = torch.cat([
            target['boxes'][target_obj_idx]
            for target, (_, target_obj_idx) in zip(targets, match_indices)],
            dim=0
        )
        # matched_target_boxes -> (num_targets_for_entire_batch, 4)

        # Convert matched pred boxes to x1y1x2y2 format (for IoU / GIoU)
        matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
            matched_pred_boxes, 'cxcywh', 'xyxy')
        # ...and the targets the other way, for the L1 term (see match()).
        matched_target_boxes_cxcywh = torchvision.ops.box_convert(
            matched_target_boxes, 'xyxy', 'cxcywh')

        # An image batch can legitimately contain no boxes at all; clamp the
        # normalizer so the loss is 0 rather than NaN.
        num_boxes = max(matched_pred_boxes.shape[0], 1)

        # ---- VFL classification loss ----
        # The IoU-aware part: a matched query's target is not a hard 1.0, it is
        # the IoU its own box achieves against the GT. The single score then
        # encodes "this is class c AND my box is this good", which is what makes
        # top-k query selection in the encoder IoU-aware rather than
        # classification-only. Detached - this is a target, not a path for
        # box gradients (those come from the L1 / GIoU terms below).
        target_onehot = torch.zeros_like(cls_output)
        target_score = torch.zeros_like(cls_output)
        if matched_pred_boxes.shape[0] > 0:
            ious = torchvision.ops.box_iou(
                matched_pred_boxes_x1y1x2y2, matched_target_boxes
            ).diagonal().detach()
            cls_idx = valid_obj_target_cls.to(cls_output.device)
            target_onehot[pred_batch_idxs, pred_query_idx, cls_idx] = 1.0
            target_score[pred_batch_idxs, pred_query_idx, cls_idx] = ious
        # Everything not named in match_indices keeps target_score 0, i.e. is a
        # negative. That covers unmatched queries AND every CDN negative and
        # padding query - exactly the target contrastive denoising wants.

        loss_cls = self.varifocal_loss(cls_output, target_score, target_onehot,
                                       num_boxes)

        # L1 in cxcywh so a centre shift is counted once, not twice.
        loss_bbox = torch.nn.functional.l1_loss(
            matched_pred_boxes,
            matched_target_boxes_cxcywh,
            reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        # Compute GIoU loss
        loss_giou = torchvision.ops.generalized_box_iou_loss(
            matched_pred_boxes_x1y1x2y2,
            matched_target_boxes
        )
        loss_giou = loss_giou.sum() / num_boxes

        return loss_cls, loss_bbox, loss_giou

    def set_loss(self, cls_output, bbox_output, targets):
        r"""
        Hungarian matching plus the set-prediction loss, for one set of
        predictions. Called once per decoder layer and once for the encoder
        query-selection head.

        :return: ``(loss_cls, loss_bbox, loss_giou)`` scalars
        """
        match_indices = self.match(cls_output, bbox_output, targets)
        return self.losses_from_indices(cls_output, bbox_output, targets,
                                        match_indices)

    def weighted_losses(self, cls_output, bbox_output, targets, match_indices):
        r"""
        :meth:`losses_from_indices` with the config's loss weights applied,
        returned in the two buckets the training loop sums.

        :return: ``(classification_term, bbox_regression_term)``
        """
        loss_cls, loss_bbox, loss_giou = self.losses_from_indices(
            cls_output, bbox_output, targets, match_indices)
        return (loss_cls * self.cls_loss_weight,
                loss_bbox * self.l1_cost_weight + loss_giou * self.giou_cost_weight)
    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        # x -> (B, C, H, W)
        # default d_model - 256
        # default C - 3
        # default H,W - 640,640
        # S3,S4,S5 feature maps -> 80x80, 40x40, 20x20

        s3, s4, s5 = self.backbone_forward(x)

        s3, s4, s5 = [
            proj(feat)
            for proj, feat in zip(self.backbone_proj, [s3, s4, s5])
        ]
        # each -> (B, d_model, feat_h, feat_w)

        # ---------------- AIFI (S5 only) ----------------
        batch_size, d_model, feat_h, feat_w = s5.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, s5)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        s5_flat = s5.reshape(batch_size, d_model,
                             feat_h * feat_w).transpose(1, 2)
        # s5_flat -> (B, feat_h*feat_w, d_model)

        # return_attn=False keeps nn.MultiheadAttention on its fused path and
        # stops it materialising a (B, 400, 400) map per layer on every forward.
        # Nothing consumed them; pass return_attn=True to inspect them.
        enc_output, _ = self.s5_encoder(s5_flat, spatial_pos_embed)
        # enc_output -> (B, feat_h*feat_w, d_model)

        # Back to a 2D map so the convolutional neck can consume it
        f5 = enc_output.transpose(1, 2).reshape(batch_size, d_model,
                                                feat_h, feat_w)
        # ---------------- AIFI end ----------------

        # ---------------- CCFM ----------------
        # Top-down: F5 -> P4 -> P3
        up2 = self.up_sampling[0](f5)
        up2 = torch.cat((up2, s4), dim=1)
        up2 = self.rep_blocks_up[0](up2)

        up3 = self.up_sampling[1](up2)
        up3 = torch.cat((up3, s3), dim=1)
        up3 = self.rep_blocks_up[1](up3)

        # Bottom-up: P3 -> N4 -> N5
        down2 = self.down_sampling[0](up3)
        down2 = torch.cat((down2, up2), dim=1)
        down2 = self.rep_blocks_down[0](down2)

        down1 = self.down_sampling[1](down2)
        down1 = torch.cat((down1, f5), dim=1)
        down1 = self.rep_blocks_down[1](down1)
        # ---------------- CCFM end ----------------

        # Flatten the three levels into one token sequence, finest level first
        levels = [up3, down2, down1]  # strides 8, 16, 32
        encoder_output = torch.cat(
            [level.flatten(2).transpose(1, 2) for level in levels], dim=1)
        # encoder_output -> (B, sum(H*W), d_model), e.g. (B, 8400, 256)

        spatial_shapes = torch.as_tensor(
            [[level.shape[2], level.shape[3]] for level in levels],
            dtype=torch.long, device=encoder_output.device)
        # spatial_shapes -> (num_levels, 2), e.g. [[80,80],[40,40],[20,20]]
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),
            spatial_shapes.prod(dim=1).cumsum(0)[:-1]
        ])
        # level_start_index -> (num_levels,), e.g. [0, 6400, 8000]

        # ---------------- IoU-aware query selection ----------------
        anchors, valid_mask = self.get_anchors(spatial_shapes,
                                               encoder_output.device,
                                               encoder_output.dtype)
        # Border cells whose anchor falls outside the image are zeroed out,
        # then projected + LayerNormed so the selection heads see a controlled
        # scale (see self.enc_output).
        memory = self.enc_output(encoder_output * valid_mask)

        enc_logits = self.enc_class_head(memory)
        # enc_logits -> (B, sum(H*W), num_classes)
        enc_boxes = (self.enc_bbox_head(memory) + anchors).sigmoid()
        # enc_boxes -> (B, sum(H*W), 4) cxcywh in [0, 1]

        # Score by the best class. VFL: every channel is a foreground class
        # now, so there is no background column to slice off. sigmoid is
        # monotonic, so ranking the raw logits gives the same top-k.
        #
        # This is where VFL pays for itself: because the loss trained these
        # logits toward IoU rather than a hard 1.0, a high score here means
        # "right class AND well localised". Under the old softmax/CE head the
        # score knew nothing about box quality, so top-k could hand the decoder
        # confident tokens with poor reference boxes.
        scores = enc_logits.max(dim=-1).values
        # scores -> (B, sum(H*W))

        topk_indices = torch.topk(scores, k=self.num_queries, dim=1).indices
        # topk_indices -> (B, num_queries)

        gather_d = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        gather_c = topk_indices.unsqueeze(-1).expand(-1, -1, self.num_classes)
        gather_4 = topk_indices.unsqueeze(-1).expand(-1, -1, 4)

        # Content queries are the projected encoder features of the selected
        # tokens (RT-DETR gathers from the same enc_output-processed memory the
        # heads scored), detached so the decoder does not backprop into the
        # memory through them.
        query_objects = torch.gather(memory, 1, gather_d).detach()

        # These two stay attached: they carry the gradient for the encoder
        # auxiliary loss, which is the only thing that trains the selection head
        enc_topk_logits = torch.gather(enc_logits, 1, gather_c)
        enc_topk_boxes = torch.gather(enc_boxes, 1, gather_4)

        # The decoder refines a detached copy
        reference_points = enc_topk_boxes.detach()

        # ---------------- CDN (training only) ----------------
        # Build noised copies of the GT and prepend them as extra queries. Their
        # assignment is known, so they give a stable signal that does not depend
        # on Hungarian matching converging. attn_mask stops the ordinary queries
        # from reading the answer off them.
        dn_meta, attn_mask = None, None
        if (self.training and targets is not None
                and self.denoising_class_embed is not None):
            dn_content, dn_ref_points, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    num_classes=self.num_classes,
                    num_queries=self.num_queries,
                    class_embed=self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    device=encoder_output.device))
            if dn_meta is not None:
                # DN queries go in FRONT; dn_num_split peels them back off later
                query_objects = torch.cat([dn_content, query_objects], dim=1)
                reference_points = torch.cat([dn_ref_points, reference_points],
                                             dim=1)
        # ---------------- CDN end ----------------

        # The positional-query module is passed in rather than a precomputed
        # embedding: the decoder re-applies it to each layer's own refined
        # boxes. Computing it once here would leave every layer positioned by
        # the encoder's boxes. It covers the DN queries for free, since they
        # are already concatenated into reference_points.
        decoder_outputs, updated_reference_points = self.decoder(
            query_objects,
            self.ref_to_query_embed,
            reference_points,
            encoder_output,
            spatial_shapes,
            level_start_index,
            attn_mask=attn_mask,
        )
        # decoder_outputs -> (num_decoder_layers, B, num_dn + num_queries, d_model)
        # updated_reference_points -> (num_decoder_layers, B, num_dn + num_queries, 4)

        # Each layer is scored by its own head (A4), so this cannot be a
        # single call over the stacked tensor any more.
        cls_output = torch.stack([
            self.class_heads[layer_idx](decoder_outputs[layer_idx])
            for layer_idx in range(self.num_decoder_layers)
        ])
        # cls_output -> (num_decoder_layers, B, num_dn + num_queries, num_classes)
        bbox_output = updated_reference_points
        # bbox_output -> (num_decoder_layers, B, num_dn + num_queries, 4), cxcywh

        # Split the DN part back off. NOTE dim=2: these tensors are stacked over
        # decoder layers, so the query axis is 2, not 1.
        dn_cls_output, dn_bbox_output = None, None
        if dn_meta is not None:
            dn_cls_output, cls_output = cls_output.split(
                dn_meta['dn_num_split'], dim=2)
            dn_bbox_output, bbox_output = bbox_output.split(
                dn_meta['dn_num_split'], dim=2)

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if targets is not None:
            # Deep supervision: match and score every decoder layer...
            for decoder_idx in range(self.num_decoder_layers):
                match_indices = self.match(cls_output[decoder_idx],
                                           bbox_output[decoder_idx], targets)
                cls_term, box_term = self.weighted_losses(
                    cls_output[decoder_idx], bbox_output[decoder_idx],
                    targets, match_indices)
                losses['classification'].append(cls_term)
                losses['bbox_regression'].append(box_term)

            # ...plus one extra term on the encoder query-selection head.
            # Without this, enc_class_head feeds only a non-differentiable
            # topk and receives no gradient at all, so query selection would
            # stay frozen at its random initialisation forever.
            # NOTE: appended to the same two lists the training loop already
            # sums, so the trailing entries are the encoder head and the CDN
            # terms rather than decoder layers.
            enc_match_indices = self.match(enc_topk_logits, enc_topk_boxes,
                                           targets)
            cls_term, box_term = self.weighted_losses(
                enc_topk_logits, enc_topk_boxes, targets, enc_match_indices)
            losses['classification'].append(cls_term)
            losses['bbox_regression'].append(box_term)

            # ...plus the CDN terms, one per decoder layer. No matching here:
            # the assignment is known by construction. Negative and padding DN
            # queries are simply absent from the indices, so they fall through
            # to the loss's background default - which is exactly the target
            # contrastive denoising wants for them.
            if dn_meta is not None:
                dn_match_indices = build_dn_match_indices(dn_meta, targets)
                for decoder_idx in range(self.num_decoder_layers):
                    cls_term, box_term = self.weighted_losses(
                        dn_cls_output[decoder_idx],
                        dn_bbox_output[decoder_idx],
                        targets, dn_match_indices)
                    losses['classification'].append(cls_term)
                    losses['bbox_regression'].append(box_term)

            detr_output['loss'] = losses

        if not self.training:
            # For inference we are only interested in last layer outputs
            final_cls_output = cls_output[-1]
            final_bbox_output = bbox_output[-1]
            # final_cls_output -> (B, num_queries, num_classes)
            # final_bbox_output -> (B, num_queries, 4)

            # VFL: per-class sigmoid, not a softmax over classes. Scores are
            # therefore independent per class and do not sum to 1 - and each
            # one is IoU-aware, so it doubles as a localisation-quality
            # estimate. That is what makes score_thresh meaningful and, in
            # principle, lets NMS be dropped: a duplicate with a worse box now
            # scores lower rather than merely tying.
            prob = final_cls_output.sigmoid()

            # convert to x1y1x2y2 format
            boxes = torchvision.ops.box_convert(final_bbox_output,
                                                'cxcywh',
                                                'xyxy')

            # RT-DETR decodes by taking the top-k entries of the FLATTENED
            # (num_queries x num_classes) score matrix, not one argmax label per
            # query. Two consequences, both of which matter for mAP:
            #   * a query may be reported under more than one class, which is
            #     how a genuinely ambiguous box earns credit on the class it
            #     got right instead of being silently dropped;
            #   * the detection budget is num_top_queries, decoupled from
            #     num_queries. Under the old prob.max(-1) this model could never
            #     emit more than num_queries (25) boxes per image, which
            #     truncates the low-precision tail of the PR curve - exactly the
            #     region an mAP sweep at score_thresh 0 is there to measure.
            # Scores are per-class sigmoids, so they are directly comparable
            # across classes; a softmax head could not be decoded this way.
            flat_scores = prob.flatten(1)
            # flat_scores -> (B, num_queries * num_classes)
            topk = min(self.num_top_queries, flat_scores.shape[1])
            scores, flat_idx = torch.topk(flat_scores, topk, dim=1)
            # Undo the flatten: which query, and which class of it.
            query_idx = torch.div(flat_idx, self.num_classes,
                                  rounding_mode='floor')
            labels = flat_idx % self.num_classes
            boxes = boxes.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, 4))
            # boxes/scores/labels -> (B, topk, 4) / (B, topk) / (B, topk)

            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

                # Low score filtering
                keep_idxs = scores_idx >= score_thresh
                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]

                # NMS filtering
                if use_nms:
                    keep_idxs = torchvision.ops.batched_nms(
                        boxes_idx,
                        scores_idx,
                        labels_idx,
                        iou_threshold=self.nms_threshold)
                    scores_idx = scores_idx[keep_idxs]
                    boxes_idx = boxes_idx[keep_idxs]
                    labels_idx = labels_idx[keep_idxs]
                detections.append(
                    {
                        "boxes": boxes_idx,
                        "scores": scores_idx,
                        "labels": labels_idx,
                    }
                )

            detr_output['detections'] = detections
        return detr_output



# ---------------------------------------------------------------------------
# Shared run helpers: used by BOTH the training cell and the test-evaluation
# cell below.
#
# They live in their own cell on purpose. The training cell starts a full run
# when executed, so if these lived there you could not evaluate a checkpoint
# without first kicking off training. Run: model cells -> this cell -> eval.
# ---------------------------------------------------------------------------
import json
import platform


def collate_function(data):
    r"""
    Detection targets are ragged (a different number of objects per image), so we
    cannot stack them. Transpose the batch into tuples instead:
    ``(images, targets, filenames)``.
    """
    return tuple(zip(*data))


def set_seed(seed):
    r"""Seed torch, numpy and python's rng."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_num_workers(train_config):
    r"""
    Worker count for the loaders.

    Augmentation (zoom-out, IoU crop, photometric distort, 640px resize) runs on
    the CPU per image, so with ``num_workers=0`` it blocks the training step and
    the GPU starves. Workers only pay off when something else can use the time,
    hence the CUDA-dependent default.

    Forced to 0 in a Windows notebook: DataLoader workers use *spawn* there, and
    the child process cannot unpickle ``VOCDataset`` because the class lives in
    the notebook's ``__main__`` rather than an importable module. The symptom is
    a hang or ``Can't get attribute 'VOCDataset'``. On Linux / Kaggle the fork
    start method has no such problem and the config value is used as written.

    :param train_config: config['train_params']
    """
    default_workers = 4 if torch.cuda.is_available() else 0
    num_workers = train_config.get('num_workers', default_workers)
    # if num_workers > 0 and platform.system() == 'Windows':
    #     print('Windows notebook: forcing num_workers {} -> 0 (spawn cannot '
    #           'pickle VOCDataset out of __main__). The config value is honoured '
    #           'on Linux / Kaggle.'.format(num_workers))
    #     num_workers = 0
    return num_workers


def save_checkpoint(model, optimizer, scheduler, epoch, loss, steps, path,
                    extra=None):
    r"""
    Write a resumable checkpoint, in the same format as
    ``detr_mor.engine.checkpoint.save_checkpoint``.

    :param epoch: index of the epoch that just finished
    :param steps: global step counter
    :param extra: optional dict of extra fields (e.g. the losses that made this
        the best epoch)
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item() if torch.is_tensor(loss) else loss,
        'steps': steps,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print('Checkpoint saved at epoch {}'.format(epoch))


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    map_location=None):
    r"""
    Restore a checkpoint. Returns ``(start_epoch, steps)``.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('steps', 0)

    model.load_state_dict(checkpoint)
    print('Loaded a bare state_dict; optimizer/scheduler state and the epoch '
          'counter are unavailable, so training restarts from epoch 0.')
    return 0, 0


def read_best_state(path, map_location='cpu'):
    r"""
    Recover ``(min_val_loss, val_cls_ref, val_loc_ref)`` from an existing best
    checkpoint, or ``(inf, None, None)`` if there is nothing usable to read.

    The three travel together on purpose. The best-model criterion is a
    *normalised* sum (see :func:`train_rt_detr`), so a stored ``min_val_loss``
    only means anything alongside the two reference scales it was divided by.
    A checkpoint written before normalisation carries a raw sum on a completely
    different scale; comparing against it would either freeze the best
    checkpoint forever or overwrite it immediately, so it is discarded and the
    run re-baselines instead.
    """
    nothing = (float('inf'), None, None)
    if not os.path.exists(path):
        return nothing
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        return nothing

    cls_ref = checkpoint.get('val_cls_ref')
    loc_ref = checkpoint.get('val_loc_ref')
    if cls_ref is None or loc_ref is None:
        print('Best checkpoint at {} predates val-loss normalisation; its '
              'min_val_loss is a raw sum on a different scale and cannot be '
              'compared. Re-baselining from this epoch.'.format(path))
        return nothing

    return checkpoint.get('min_val_loss', float('inf')), cls_ref, loc_ref



# ---------------------------------------------------------------------------
# Training on VOC, driven entirely by rt_detr_config.yaml.
#
# Requires the shared-helpers cell above (set_seed, resolve_num_workers,
# collate_function, save_checkpoint, load_checkpoint, read_best_state).
#
# Checkpoint format matches detr_mor/engine/checkpoint.py, so the files written
# here stay loadable by scripts/evaluate.py and by the rest of the repo.
#
# Heads-up on memory: the config asks for a large batch_size at 640x640. The
# CCFM carries a (B, 256, 80, 80) activation and AIFI a (B, 8, 400, 400)
# attention matrix, so a big batch will not fit on a single consumer GPU. If you
# OOM, drop batch_size and raise acc_steps by the same factor - the effective
# batch, and therefore the optimizer trajectory, is batch_size * acc_steps.
# ---------------------------------------------------------------------------
import copy
import time

from torch.utils.data import Subset


def build_train_val_loaders(dataset_config, train_config, generator=None):
    r"""
    Build the train/val DataLoaders by randomly splitting the train imageset.

    :param dataset_config: config['dataset_params']
    :param train_config: config['train_params']
    :param generator: optional torch.Generator, for a reproducible split
    :return: (train_loader, val_loader, dataset)
    """
    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])

    # Validation must NOT be augmented. random_split hands back Subset views of
    # one dataset object, so a plain split would leave the val slice running the
    # train pipeline (random flip / zoom-out / IoU crop / photometric distort).
    # That makes the validation loss a moving target - it would drift with the
    # augmentation draw rather than the model - and scores the model on inputs
    # that never occur at inference.
    #
    # Shallow copy: shares the parsed annotations (so the ~16k XML files are not
    # re-read) but carries its own transform_split, leaving only the
    # deterministic resize + normalise.
    voc_eval = copy.copy(voc)
    voc_eval.transform_split = 'test'

    val_fraction = train_config.get('val_split', 0.2)
    val_size = int(val_fraction * len(voc))
    train_size = len(voc) - val_size

    # Permute once and slice by hand, rather than calling random_split twice:
    # both subsets must index the SAME permutation, or train and val overlap.
    if generator is None:
        permutation = torch.randperm(len(voc)).tolist()
    else:
        permutation = torch.randperm(len(voc), generator=generator).tolist()
    train_subset = Subset(voc, permutation[:train_size])
    val_subset = Subset(voc_eval, permutation[train_size:])

    assert not (set(train_subset.indices) & set(val_subset.indices)), \
        'train/val split overlaps'

    num_workers = resolve_num_workers(train_config)
    loader_kwargs = {'num_workers': num_workers,
                     'pin_memory': True} #torch.cuda.is_available()}
    if num_workers > 0:
        # Respawning workers every epoch costs seconds per epoch at this size.
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_subset,
                              batch_size=train_config['batch_size'],
                              shuffle=True,
                              collate_fn=collate_function,
                              **loader_kwargs)
    val_loader = DataLoader(val_subset,
                            batch_size=train_config['batch_size'],
                            shuffle=False,
                            collate_fn=collate_function,
                            **loader_kwargs)

    print('Dataset: {} images -> {} train / {} val | batch_size {} | '
          'num_workers {}'.format(len(voc), train_size, val_size,
                                  train_config['batch_size'], num_workers))
    print('  train transforms: {} (augmented)'.format(
        voc.transform_split))
    print('  val   transforms: {} (resize + normalise only, no augmentation)'
          .format(voc_eval.transform_split))
    return train_loader, val_loader, voc


def targets_to_device(targets, device):
    r"""Move 'boxes' and 'labels' onto ``device``, in the dtypes the loss wants."""
    for target in targets:
        target['boxes'] = target['boxes'].float().to(device)
        target['labels'] = target['labels'].long().to(device)
    return targets


def batch_images_to_device(images, device):
    r"""Stack the per-image tensors from ``collate_function`` into (B, C, H, W)."""
    return torch.stack([im.float().to(device) for im in images], dim=0)


def train_one_epoch(model, loader, optimizer, scheduler, device, train_config,
                    steps=0):
    r"""
    Run one training epoch.

    Gradients accumulate over ``acc_steps`` batches before each optimizer step;
    the last batch of the epoch always steps, so a partial accumulation window
    is never dropped.

    :param steps: global step counter carried across epochs
    :return: (mean classification loss, mean localization loss, steps, last loss)
    """
    model.train()
    acc_steps = train_config['acc_steps']
    grad_clip = train_config.get('grad_clip', 0.1)
    num_batches = len(loader)

    classification_losses = []
    localization_losses = []
    last_loss = 0.0

    for idx, (ims, targets, _) in enumerate(tqdm(loader, desc='Training')):
        targets = targets_to_device(targets, device)
        images = batch_images_to_device(ims, device)

        batch_losses = model(images, targets)['loss']

        loss = (sum(batch_losses['classification']) +
                sum(batch_losses['bbox_regression']))

        classification_losses.append(sum(batch_losses['classification']).item())
        localization_losses.append(sum(batch_losses['bbox_regression']).item())

        loss = loss / acc_steps
        last_loss = loss.detach()

        if not torch.isfinite(loss):
            raise RuntimeError(
                'Loss became non-finite at step {}. Exiting.'.format(steps))

        loss.backward()

        if ((idx + 1) % acc_steps == 0) or (idx + 1 == num_batches):
            if grad_clip:
                # Deformable sampling offsets are the usual source of a spike;
                # DETR-family training is normally clipped at 0.1.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if steps % train_config['log_steps'] == 0:
            print('Classification Loss : {:.4f} | Localization Loss : '
                  '{:.4f}'.format(np.mean(classification_losses),
                                  np.mean(localization_losses)),
                  scheduler.get_last_lr())

        steps += 1

    return (float(np.mean(classification_losses)),
            float(np.mean(localization_losses)),
            steps,
            last_loss)


@torch.no_grad()
def validate(model, loader, device):
    r"""
    Run the validation pass.

    ``model.eval()`` disables dropout and, as a side effect of how DETR.forward
    is written, also makes it emit detections alongside the losses. Only the
    losses are used here.

    :return: (mean classification loss, mean localization loss)
    """
    model.eval()
    classification_losses = []
    localization_losses = []

    for ims, targets, _ in tqdm(loader, desc='Validating'):
        targets = targets_to_device(targets, device)
        images = batch_images_to_device(ims, device)

        batch_losses = model(images, targets)['loss']

        classification_losses.append(sum(batch_losses['classification']).item())
        localization_losses.append(sum(batch_losses['bbox_regression']).item())

    return (float(np.mean(classification_losses)),
            float(np.mean(localization_losses)))


def train_rt_detr(config_path='rt_detr_config.yaml', resume=True):
    r"""
    Full RT-DETR training run on VOC, driven by the YAML config.

    Resumes from ``{task_name}/{ckpt_name}`` if it exists, then loops epochs,
    validating and checkpointing after each. Results are appended to
    ``{task_name}/train_results.txt`` so a killed-and-restarted run keeps its
    history. The best-validation epoch is kept separately as
    ``{task_name}/best_{ckpt_name}``.

    :param config_path: path to the YAML config
    :param resume: load the rolling checkpoint if one exists
    :return: the trained model
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    print(config)

    set_seed(train_config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Seeded generator so the train/val split is identical across restarts -
    # otherwise a resumed run validates on images it has already trained on.
    generator = torch.Generator().manual_seed(train_config['seed'])
    train_loader, val_loader, _ = build_train_val_loaders(
        dataset_config, train_config, generator=generator)

    # VFL: no bg_class_idx - num_classes is foreground only
    model = DETR(model_config, num_classes=dataset_config['num_classes'])
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print('Trainable parameters: {:,} across {} tensors'.format(
        sum(p.numel() for p in trainable), len(trainable)))

    # ---- Two learning rates: backbone and everything else -----------------
    # The backbone arrives pretrained on ImageNet; the detector is random. One
    # LR for both either moves the detector too slowly or destroys the backbone
    # features in the first few hundred steps. RT-DETR runs the backbone at a
    # tenth of the base rate (1e-5 against 1e-4), and so do DETR and
    # Deformable-DETR. This only does anything once freeze_backbone is False -
    # while it is True the backbone group is empty and is not added at all.
    #
    # Note the '.' in the prefix: it selects self.backbone.* without also
    # catching self.backbone_proj.*, which is detector-side and belongs on the
    # base rate.
    backbone_params, head_params = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = train_config.get('backbone_lr', train_config['lr'] * 0.1)
    param_groups = [{'params': head_params, 'lr': train_config['lr']}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        print('Backbone trainable: {:,} params at lr {:g} '
              '(detector {:,} at lr {:g})'.format(
                  sum(p.numel() for p in backbone_params), backbone_lr,
                  sum(p.numel() for p in head_params), train_config['lr']))
    else:
        print('Backbone frozen: all {:,} trainable params at lr {:g}'.format(
            sum(p.numel() for p in head_params), train_config['lr']))

    if model_config.get('num_denoising', 0) > 0:
        # Worth stating up front: CDN is training-only, so the train loss sums
        # 2*L+1 terms while the val loss sums only L+1. Train will therefore
        # read HIGHER than val for the whole run - that is not overfitting in
        # reverse, the two just are not on the same scale. Val is internally
        # consistent across epochs, so best-checkpoint selection is unaffected.
        print('CDN active (num_denoising={}): train loss includes the denoising '
              'terms, val loss does not, so the two are not directly '
              'comparable.'.format(model_config['num_denoising']))

    # Built before any resume, so restored state lands in the objects the loop
    # actually uses.
    # MultiStepLR scales every group by gamma, so the two rates keep their
    # ratio through the decay rather than collapsing onto one.
    optimizer = torch.optim.AdamW(params=param_groups,
                                  lr=train_config['lr'],
                                  weight_decay=1E-4)
    scheduler = MultiStepLR(optimizer,
                            milestones=train_config['lr_steps'],
                            gamma=0.1)

    task_dir = train_config['task_name']
    os.makedirs(task_dir, exist_ok=True)
    result_path = os.path.join(task_dir, 'train_results.txt')
    ckpt_path = os.path.join(task_dir, train_config['ckpt_name'])
    best_ckpt_path = os.path.join(task_dir, 'best_' + train_config['ckpt_name'])

    # Read back so a restarted run cannot overwrite a better epoch with a worse
    # one. val_cls_ref / val_loc_ref are the scales min_val_loss was normalised
    # against and must be restored with it - re-deriving them from a resumed
    # epoch would silently change what "best" means mid-run.
    if resume:
        min_val_loss, val_cls_ref, val_loc_ref = read_best_state(best_ckpt_path)
    else:
        min_val_loss, val_cls_ref, val_loc_ref = float('inf'), None, None

    start_epoch, steps = 0, 0
    if resume and os.path.exists(ckpt_path):
        print('Loading checkpoint as one exists at {}'.format(ckpt_path))
        start_epoch, steps = load_checkpoint(ckpt_path, model, optimizer,
                                             scheduler, map_location=device)
        print('Resuming at epoch {} with lr {:g}'.format(
            start_epoch + 1, scheduler.get_last_lr()[0]))

    num_epochs = train_config['num_epochs']
    last_loss = 0.0

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        print('\n=== Epoch {}/{} ==='.format(epoch + 1, num_epochs))

        try:
            train_cls_loss, train_loc_loss, steps, last_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                train_config, steps=steps)
        except RuntimeError as exc:
            with open(result_path, 'a') as log:
                log.write('{}\n'.format(exc))
            raise

        print('Running Validation for Epoch {}...'.format(epoch + 1))
        val_cls_loss, val_loc_loss = validate(model, val_loader, device)

        scheduler.step()

        print('Finished epoch {} in {:.1f}s'.format(epoch + 1,
                                                    time.time() - epoch_start))
        print('Classification Loss : {:.4f} | Localization Loss : {:.4f}'.format(
            train_cls_loss, train_loc_loss))
        # ---- Best-model criterion: normalised sum ----
        # A raw val_cls + val_loc is not a balanced criterion. Under VFL the
        # classification term sits around 0.5 while localization sits around 37,
        # so the plain sum is ~99% localization and classification effectively
        # does not participate in choosing the best epoch.
        #
        # Fix: divide each term by its value at the first validated epoch, so
        # both start at exactly 1.0 and thereafter contribute in proportion to
        # how far each has improved *relative to where it started*. That is
        # scale-free, so it keeps working if the loss weights or the
        # classification loss are changed again later.
        if val_cls_ref is None:
            # First validated epoch of the run establishes the scales. Floored
            # so a degenerate 0.0 cannot produce a division by zero.
            val_cls_ref = max(val_cls_loss, 1e-8)
            val_loc_ref = max(val_loc_loss, 1e-8)
            print('Best-model criterion baselined on epoch {}: '
                  'cls {:.4f}, loc {:.4f} (each counts as 1.0)'.format(
                      epoch + 1, val_cls_ref, val_loc_ref))

        val_cls_norm = val_cls_loss / val_cls_ref
        val_loc_norm = val_loc_loss / val_loc_ref
        val_loss = val_cls_norm + val_loc_norm

        print('Val Class Loss: {:.4f} | Val Loc Loss: {:.4f}'.format(
            val_cls_loss, val_loc_loss))
        print('Val criterion (normalised): {:.4f}  ='
              ' cls {:.4f} + loc {:.4f}'.format(
                  val_loss, val_cls_norm, val_loc_norm))

        with open(result_path, 'a') as log:
            log.write(
                'Epoch {}: Train_Class_Loss={:.6f}, Train_Loc_Loss={:.6f}, '
                'Val_Class_Loss={:.6f}, Val_Loc_Loss={:.6f}, '
                'Val_Norm_Criterion={:.6f}\n'.format(
                    epoch + 1, train_cls_loss, train_loc_loss,
                    val_cls_loss, val_loc_loss, val_loss))

        # epoch + 1 so a resumed run starts on the next epoch, not this one again
        save_checkpoint(model, optimizer, scheduler, epoch + 1, last_loss,
                        steps, ckpt_path)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # The reference scales are stored alongside min_val_loss: without
            # them a resumed run could not tell what this number was measured
            # against. See read_best_state.
            save_checkpoint(model, optimizer, scheduler, epoch + 1, last_loss,
                            steps, best_ckpt_path,
                            extra={'min_val_loss': min_val_loss,
                                   'val_cls_ref': val_cls_ref,
                                   'val_loc_ref': val_loc_ref,
                                   'train_cls_loss': train_cls_loss,
                                   'train_loc_loss': train_loc_loss,
                                   'val_cls_loss': val_cls_loss,
                                   'val_loc_loss': val_loc_loss})
            print('New best val criterion {:.4f} (cls {:.4f} + loc {:.4f}), '
                  'saved to {}'.format(min_val_loss, val_cls_norm, val_loc_norm,
                                       best_ckpt_path))

    with open(result_path, 'a') as log:
        log.write('Done Training...\n')
    print('Done Training...')
    return model


if __name__ == '__main__':
    trained_model = train_rt_detr('rt_detr_config.yaml')