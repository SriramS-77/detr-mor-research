"""Fixed sinusoidal spatial position embeddings for backbone feature maps."""

import torch


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
