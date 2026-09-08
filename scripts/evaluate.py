"""Evaluate a trained DETR / DETR-MoR checkpoint with VOC mAP.

Example::

    python scripts/evaluate.py --config configs/mor_voc.yaml --model mor
"""

import argparse
import os

import _bootstrap  # noqa: F401  (sys.path setup)

from detr_mor.config import load_config, split_config
from detr_mor.data import build_test_loader
from detr_mor.engine import checkpoint_path, load_checkpoint
from detr_mor.evaluation import evaluate_map
from detr_mor.models import build_model
from detr_mor.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DETR / DETR-MoR mAP on the VOC test set')
    parser.add_argument('--config', required=True,
                        help='path to the yaml config')
    parser.add_argument('--model', choices=('detr', 'mor'), default='mor',
                        help='which architecture to evaluate (default: mor)')
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda when available)')
    parser.add_argument('--ckpt', default=None,
                        help='checkpoint path (default: {task_name}/{ckpt_name})')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU above which a detection counts as a match')
    parser.add_argument('--method', choices=('area', 'interp'), default='area',
                        help="'area' for all-point AP, 'interp' for VOC2007 "
                             "11-point AP")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config, model_type=args.model)
    print(config)
    dataset_config, model_config, train_config = split_config(config)

    set_seed(train_config['seed'])
    device = resolve_device(args.device)
    print('Using device: {}'.format(device))

    test_loader, voc = build_test_loader(dataset_config)

    # The checkpoint supplies the weights, so there is no point downloading
    # ImageNet weights just to overwrite them.
    model = build_model(args.model, model_config, dataset_config, device=device,
                        pretrained_backbone=False)

    ckpt = args.ckpt or checkpoint_path(train_config)
    if not os.path.exists(ckpt):
        raise FileNotFoundError('No checkpoint exists at {}'.format(ckpt))
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    evaluate_map(model, voc, test_loader, device, train_config,
                 method=args.method, iou_threshold=args.iou_threshold)


if __name__ == '__main__':
    main()
