"""Small helpers shared by the training / evaluation entry points."""

import random

import numpy as np
import torch

from detr_mor.utils.distributed import get_local_rank


def set_seed(seed):
    r"""Seed torch, numpy and python's rng (as the notebook did)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device=None):
    r"""
    Turn a device string into a torch.device.

    An index-less 'cuda' resolves to ``cuda:{LOCAL_RANK}``, which is a no-op
    (``cuda:0``) for an ordinary run and is what gives each torchrun process its
    own GPU - without it every rank would pile onto cuda:0.

    :param device: 'cuda', 'cpu', 'cuda:1', or None to auto-select.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA requested but torch.cuda.is_available() is False')
        if device.index is None:
            device = torch.device('cuda', get_local_rank())
    return device


def targets_to_device(targets, device):
    r"""
    Move the tensors the model reads ('boxes', 'labels') onto ``device``, casting
    to the dtypes the loss expects. Mutates and returns the same target dicts,
    matching the notebook's training loop.

    :param targets: tuple/list of per-image target dicts
    :param device: torch device
    """
    for target in targets:
        target['boxes'] = target['boxes'].float().to(device)
        target['labels'] = target['labels'].long().to(device)
    return targets


def batch_images_to_device(images, device):
    r"""
    Stack a tuple of per-image tensors (as produced by ``collate_function``) into
    a single float batch on ``device``.

    :param images: sequence of (C, H, W) tensors
    :return: (B, C, H, W) float tensor
    """
    return torch.stack([im.float().to(device) for im in images], dim=0)
