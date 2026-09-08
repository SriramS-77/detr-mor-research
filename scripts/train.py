"""Train DETR or DETR-MoR on Pascal VOC.

Examples::

    python scripts/train.py --config configs/mor_voc.yaml --model mor
    python scripts/train.py --config configs/voc.yaml --model detr --device cuda:0
"""

import argparse

import _bootstrap  # noqa: F401  (sys.path setup)
import torch

from detr_mor.config import load_config, split_config
from detr_mor.data import build_train_val_loaders
from detr_mor.engine import train
from detr_mor.models import build_model
from detr_mor.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train DETR / DETR-MoR on VOC')
    parser.add_argument('--config', required=True,
                        help='path to the yaml config')
    parser.add_argument('--model', choices=('detr', 'mor'), default='mor',
                        help='which architecture to train (default: mor)')
    parser.add_argument('--device', default=None,
                        help="torch device, e.g. cuda / cuda:1 / cpu "
                             "(default: cuda when available)")
    parser.add_argument('--no-resume', action='store_true',
                        help='ignore any existing checkpoint and start fresh')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='skip downloading ImageNet backbone weights')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config, model_type=args.model)
    print(config)
    dataset_config, model_config, train_config = split_config(config)

    set_seed(train_config['seed'])
    device = resolve_device(args.device)
    print('Using device: {}'.format(device))

    # Seed the train/val split separately so it is stable across resumes.
    split_generator = torch.Generator().manual_seed(train_config['seed'])
    train_loader, val_loader, _ = build_train_val_loaders(
        dataset_config, train_config, generator=split_generator)

    model = build_model(args.model, model_config, dataset_config, device=device,
                        pretrained_backbone=not args.no_pretrained)

    train(model, train_loader, val_loader, device, train_config,
          label=args.model.upper(), resume=not args.no_resume)


if __name__ == '__main__':
    main()
