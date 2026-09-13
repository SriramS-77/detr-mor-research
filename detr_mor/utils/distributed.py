"""Optional multi-GPU support via torch.distributed (DDP).

Every helper here degrades to a no-op when the process was not launched under
``torchrun``, so ``python scripts/train.py`` on one GPU or on CPU goes through
exactly the same code path it always did.

Launch two GPUs with::

    torchrun --standalone --nproc_per_node=2 scripts/train.py \
        --config configs/mor_cyclic.yaml --model mor --device cuda

Note that ``batch_size`` in the config is then *per process*: the effective
batch is ``batch_size * world_size``.
"""

import inspect
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed_launch():
    r"""True if torchrun set the environment variables we need."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def get_local_rank():
    r"""Index of this process's GPU on this node (0 when not launched by torchrun)."""
    return int(os.environ.get('LOCAL_RANK', 0))


def get_rank():
    r"""Global rank of this process, 0 when running single-process."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    r"""Number of processes in the group, 1 when running single-process."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    r"""True on rank 0, and always true single-process. Guards prints and writes."""
    return get_rank() == 0


def init_distributed():
    r"""
    Join the process group if torchrun launched us, otherwise do nothing.

    NCCL is used when CUDA is present (the real multi-GPU case) and gloo
    otherwise - the gloo path is what makes a two-process CPU test possible on a
    laptop with no GPU at all.

    :return: True if this process is one of several, False if single-process
    """
    if not is_distributed_launch():
        return False
    if torch.cuda.is_available():
        # Must precede init_process_group so NCCL binds to the right device.
        torch.cuda.set_device(get_local_rank())
        backend = 'nccl'
    else:
        backend = 'gloo'
    dist.init_process_group(backend=backend)
    return dist.get_world_size() > 1


def cleanup_distributed():
    r"""Tear the group down. Safe to call when nothing was ever initialised."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def wrap_ddp(model, device):
    r"""
    Wrap ``model`` in DDP, or return it untouched when running single-process.

    Call this *after* any checkpoint restore, so state dicts stay
    ``module.``-free and the optimizer keeps pointing at the same Parameter
    objects.

    Buffer syncing is switched off: every buffer in this model is frozen
    (FrozenBatchNorm2d statistics, ``criterion.cls_weights``), so broadcasting
    them on each forward is pure overhead. The kwarg that does this was renamed
    in torch 2.14, hence the signature probe - older versions only understand
    ``broadcast_buffers``.

    :param device: this rank's device; its index selects the GPU
    :return: the model, wrapped if and only if world_size > 1
    """
    if get_world_size() <= 1:
        return model
    kwargs = {'device_ids': [get_local_rank()] if device.type == 'cuda' else None}
    params = inspect.signature(DistributedDataParallel.__init__).parameters
    if 'forward_sync_buffers' in params:
        kwargs['forward_sync_buffers'] = False
    else:
        kwargs['broadcast_buffers'] = False
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model):
    r"""
    The underlying module, so checkpoints never gain a ``module.`` key prefix
    and stay loadable by ``scripts/evaluate.py`` and by single-GPU runs.
    """
    return model.module if hasattr(model, 'module') else model


def reduce_mean(value, device):
    r"""
    Average a python float across all ranks.

    Each rank validates only its own shard, so without this the ranks would
    compute different validation losses and disagree about which epoch was the
    best - and only rank 0's opinion would reach the disk.

    :param value: float local to this rank
    :param device: device the collective runs on
    :return: the mean across ranks, or ``value`` unchanged single-process
    """
    world_size = get_world_size()
    if world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


def any_rank(flag, device):
    r"""
    True if ``flag`` holds on any rank.

    A NaN loss on one GPU has to abort all of them: raising on one rank alone
    leaves the others blocked forever on the next collective.

    :param flag: bool local to this rank
    :return: the OR across ranks, or ``flag`` unchanged single-process
    """
    if get_world_size() == 1:
        return flag
    tensor = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item() > 0
