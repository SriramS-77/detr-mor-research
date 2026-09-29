"""Evaluate a trained DETR / DETR-MoR checkpoint with VOC mAP.

Example::

    python scripts/evaluate.py --config configs/mor_voc.yaml --model mor
"""

import argparse
import datetime
import json
import os

import _bootstrap  # noqa: F401  (sys.path setup)

from detr_mor.config import load_config, split_config
from detr_mor.data import build_test_loader
from detr_mor.engine import checkpoint_path, load_checkpoint
from detr_mor.evaluation import COCO_IOU_THRESHOLDS, evaluate_coco_map
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
    parser.add_argument('--iou-thresholds', type=float, nargs='+',
                        default=list(COCO_IOU_THRESHOLDS),
                        help='IoU thresholds to score at; AP is their average '
                             "(default: COCO's 0.50 0.55 ... 0.95). Pass a "
                             'single value for a plain VOC-style run.')
    parser.add_argument('--method', choices=('area', 'interp'), default='area',
                        help="'area' for all-point AP, 'interp' for VOC2007 "
                             "11-point AP")
    parser.add_argument('--out', default=None,
                        help='where to write the JSON results '
                             '(default: {task_name}/eval_results.json); '
                             "pass 'none' to skip writing")
    return parser.parse_args()


def _json_safe(value):
    r"""float(), but NaN becomes None so the result is strict-parseable JSON."""
    value = float(value)
    return None if value != value else value


def write_results(path, args, train_config, ckpt, results):
    r"""Dump the mAP numbers plus enough context to tell two runs apart."""
    payload = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': args.config,
        'model': args.model,
        'checkpoint': ckpt,
        'iou_thresholds': [float(t) for t in sorted(results['per_iou'])],
        'method': args.method,
        'score_threshold': train_config['eval_score_threshold'],
        'use_nms': train_config['use_nms_eval'],
        # NaN is not valid JSON - json.dump would emit a bare `NaN` token that
        # strict parsers reject. A class with no scorable ground truth is
        # reported as null instead, which round-trips cleanly.
        'ap': _json_safe(results['ap']),
        'ap50': _json_safe(results['ap50']),
        'ap75': _json_safe(results['ap75']),
        'class_ap': {name: _json_safe(ap)
                     for name, ap in results['class_ap'].items()},
        # Keyed by a formatted string, not the float itself: JSON object keys
        # are strings regardless, and mixing numeric and string keys in one
        # dict makes json.dump(sort_keys=True) raise on the comparison.
        'per_iou': {
            '{:.2f}'.format(threshold): {
                'mean_ap': _json_safe(result['mean_ap']),
                'class_ap': {name: _json_safe(ap)
                             for name, ap in result['class_ap'].items()},
            }
            for threshold, result in results['per_iou'].items()
        },
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        # allow_nan=False turns any NaN that slips past _json_safe into a
        # loud error rather than a silently invalid file.
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    print('Wrote results to {}'.format(path))


def main():
    args = parse_args()

    config = load_config(args.config, model_type=args.model)
    print(config)
    dataset_config, model_config, train_config = split_config(config)

    set_seed(train_config['seed'])
    device = resolve_device(args.device)
    print('Using device: {}'.format(device))

    test_loader, voc = build_test_loader(dataset_config, train_config)

    # The checkpoint supplies the weights, so there is no point downloading
    # ImageNet weights just to overwrite them.
    model = build_model(args.model, model_config, dataset_config, device=device,
                        pretrained_backbone=False)

    ckpt = args.ckpt or checkpoint_path(train_config)
    if not os.path.exists(ckpt):
        raise FileNotFoundError('No checkpoint exists at {}'.format(ckpt))
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    results = evaluate_coco_map(
        model, voc, test_loader, device, train_config,
        method=args.method, iou_thresholds=args.iou_thresholds)

    if args.out != 'none':
        out = args.out or os.path.join(train_config['task_name'],
                                       'eval_results.json')
        write_results(out, args, train_config, ckpt, results)


if __name__ == '__main__':
    main()
