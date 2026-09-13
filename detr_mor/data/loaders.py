"""DataLoader construction for the VOC dataset."""

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from detr_mor.data.voc import VOCDataset


def collate_function(data):
    r"""
    Detection targets are ragged (a different number of objects per image), so we
    cannot stack them. Transpose the batch into tuples instead:
    ``(images, targets, filenames)``.
    """
    return tuple(zip(*data))


def _loader_kwargs(train_config):
    r"""
    Worker settings shared by every loader.

    Augmentation (zoom-out, IoU crop, photometric distort, 640px resize) runs on
    the CPU per image, so with ``num_workers=0`` it blocks the training step and
    the GPU starves. Workers only help when something else can use the time,
    which on CPU-only runs is not the case - hence the CUDA-dependent default.
    ``num_workers`` in train_params overrides it.

    Under torchrun this count is *per process*, so two ranks at the default
    spawn eight loader processes between them. On a box with few vCPUs set
    ``num_workers`` explicitly rather than leaving the default.

    :param train_config: config['train_params']
    """
    default_workers = 4 if torch.cuda.is_available() else 0
    num_workers = train_config.get('num_workers', default_workers)
    kwargs = {'num_workers': num_workers,
              'pin_memory': torch.cuda.is_available()}
    if num_workers > 0:
        # Respawning workers every epoch costs seconds per epoch at this size.
        kwargs['persistent_workers'] = True
    return kwargs


def build_train_val_loaders(dataset_config, train_config, generator=None,
                            distributed=False):
    r"""
    Build the train/val DataLoaders by randomly splitting the train imageset.

    :param dataset_config: config['dataset_params']
    :param train_config: config['train_params']
    :param generator: optional torch.Generator, for a reproducible split
    :param distributed: shard each epoch across ranks with a DistributedSampler
        instead of shuffling locally. ``batch_size`` is then per process, so the
        effective batch is ``batch_size * world_size``.
    :return: (train_loader, val_loader, dataset)
    """
    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])

    val_fraction = train_config.get('val_split', 0.2)
    val_size = int(val_fraction * len(voc))
    train_size = len(voc) - val_size

    split_kwargs = {} if generator is None else {'generator': generator}
    train_subset, val_subset = random_split(voc, [train_size, val_size],
                                            **split_kwargs)

    loader_kwargs = _loader_kwargs(train_config)

    # The split above is already identical on every rank: scripts/train.py seeds
    # `generator` from train_config['seed']. The samplers only decide which
    # slice of that shared split each rank sees. They stay None single-process,
    # which leaves the loaders byte-identical to before.
    # The trainer reaches set_epoch through `train_loader.sampler`.
    train_sampler = (DistributedSampler(train_subset, shuffle=True, drop_last=True)
                     if distributed else None)
    val_sampler = (DistributedSampler(val_subset, shuffle=False)
                   if distributed else None)

    train_loader = DataLoader(train_subset,
                              batch_size=train_config['batch_size'],
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              drop_last=distributed,
                              collate_fn=collate_function,
                              **loader_kwargs)
    val_loader = DataLoader(val_subset,
                            batch_size=train_config['batch_size'],
                            shuffle=(val_sampler is None),
                            sampler=val_sampler,
                            collate_fn=collate_function,
                            **loader_kwargs)
    return train_loader, val_loader, voc


def build_test_loader(dataset_config, train_config=None):
    r"""
    Build the evaluation DataLoader over the test imageset.

    Uses batch_size=1 with the default collate, so targets come back with a
    leading batch dimension (``target['boxes'][0]``) - which is what
    :func:`detr_mor.evaluation.evaluator.evaluate_map` expects.

    :return: (test_loader, dataset)
    """
    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])
    test_loader = DataLoader(voc, batch_size=1, shuffle=False,
                             **_loader_kwargs(train_config or {}))
    return test_loader, voc
