"""Probe the largest training batch size that fits, using synthetic data.

Runs a real forward + backward + optimizer step at the config's own image size
and model dimensions, so the numbers reflect the model you will actually train.
No dataset needed.

Example::

    python scripts/find_batch_size.py --config configs/mor_voc.yaml --model mor
"""

import argparse

import _bootstrap  # noqa: F401  (sys.path setup)
import torch
from smoke_test import SyntheticDetectionDataset

from detr_mor.config import load_config, split_config
from detr_mor.data import collate_function
from detr_mor.models import build_model
from detr_mor.utils import (
    batch_images_to_device,
    resolve_device,
    targets_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find the largest batch size that fits in memory')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', choices=('detr', 'mor'), default='mor')
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-batch-size', type=int, default=64)
    parser.add_argument('--target-util', type=float, default=0.75,
                        help='report the largest batch below this fraction of '
                             'total VRAM as the recommendation (default: 0.75)')
    parser.add_argument('--no-pretrained', action='store_true')
    return parser.parse_args()


def try_batch(model, dataset, batch_size, device):
    r"""One full train step. Returns peak bytes, or None if it OOMed."""
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    samples = [dataset[i % len(dataset)] for i in range(batch_size)]
    ims, targets, _ = collate_function(samples)
    try:
        images = batch_images_to_device(ims, device)
        targets = targets_to_device(list(targets), device)
        losses = model(images, targets)['loss']
        total = (sum(losses['classification'])
                 + sum(losses['bbox_regression']))
        total.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
        peak = None
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    torch.cuda.empty_cache()
    return peak


def main():
    args = parse_args()
    config = load_config(args.config, model_type=args.model)
    dataset_config, model_config, _ = split_config(config)

    device = resolve_device(args.device)
    if device.type != 'cuda':
        raise SystemExit('This probe measures CUDA memory; got {}. On CPU just '
                         'use batch_size 1-2.'.format(device))

    total_mem = torch.cuda.get_device_properties(device).total_memory
    print('{} - {:.1f} GiB total'.format(
        torch.cuda.get_device_name(device), total_mem / 2 ** 30))

    model = build_model(args.model, model_config, dataset_config, device=device,
                        pretrained_backbone=not args.no_pretrained)
    model.train()
    dataset = SyntheticDetectionDataset(
        length=8, im_size=dataset_config['im_size'],
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx'],
        max_objects=min(8, model_config['num_queries']))

    def probe(batch_size):
        """Print one measurement; True if it fits under the target."""
        peak = try_batch(model, dataset, batch_size, device)
        if peak is None:
            print('  batch {:3d}: OOM'.format(batch_size))
            return False
        util = peak / total_mem
        print('  batch {:3d}: peak {:6.2f} GiB ({:4.1f}% of VRAM)'.format(
            batch_size, peak / 2 ** 30, 100 * util))
        return util <= args.target_util

    # Double until the target is exceeded, then walk up from the last good
    # size in small steps so the answer is not rounded down to a power of two.
    recommended, batch_size = None, 1
    while batch_size <= args.max_batch_size and probe(batch_size):
        recommended, batch_size = batch_size, batch_size * 2

    if recommended is not None:
        step = max(1, recommended // 4)
        candidate = recommended + step
        while candidate < batch_size and candidate <= args.max_batch_size:
            if not probe(candidate):
                break
            recommended = candidate
            candidate += step

    if recommended is None:
        print('\nEven batch 1 exceeds the target utilisation.')
    else:
        print('\nRecommended batch_size: {}'.format(recommended))
        print('Peak grows roughly linearly, so leave headroom: this probe uses '
              'a fixed object count, while real VOC images vary.')


if __name__ == '__main__':
    main()
