"""Evaluation-only routing diagnostics for a trained DETR-MoR checkpoint.

Scores one checkpoint under several routing policies at identical capacity
(see detr_mor/evaluation/routing_diagnostics.py), and reports how recursion
depth relates to final IoU for matched queries and how much encoder capacity
the routers spend on object tokens. No weights are changed.

Example::

    python scripts/diagnose_routing.py --config configs/mor_cyclic.yaml \
        --ckpt mor_cyclic_voc_results/mor_cyclic_voc_results_300/best_mor_cyclic_detr.pth

Add ``--limit 50`` for a quick smoke run.
"""

import argparse
import datetime
import json
import os

import _bootstrap  # noqa: F401  (sys.path setup)

from detr_mor.config import load_config, split_config
from detr_mor.data import build_test_loader
from detr_mor.engine import checkpoint_path, load_checkpoint
from detr_mor.evaluation import COCO_IOU_THRESHOLDS
from detr_mor.evaluation.routing_diagnostics import run_routing_diagnostics
from detr_mor.models import build_model
from detr_mor.utils import resolve_device, set_seed

#: name -> (encoder policy, decoder policy)
PRESETS = {
    'default': ('default', 'default'),
    'full_both': ('full', 'full'),
    'full_encoder': ('full', 'default'),
    'full_decoder': ('default', 'full'),
    'random_both': ('random', 'random'),
    'random_decoder': ('default', 'random'),
    'oracle_decoder': ('default', 'oracle'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Routing diagnostics for a DETR-MoR checkpoint')
    parser.add_argument('--config', required=True,
                        help='path to the yaml config (architecture must match '
                             'the checkpoint)')
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda when available)')
    parser.add_argument('--ckpt', default=None,
                        help='checkpoint path (default: {task_name}/{ckpt_name})')
    parser.add_argument('--policies', nargs='+', default=list(PRESETS),
                        choices=list(PRESETS),
                        help='routing policies to score (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='only use the first N test images')
    parser.add_argument('--method', choices=('area', 'interp'), default='area')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the random-routing control')
    parser.add_argument('--out', default=None,
                        help='where to write the JSON results (default: '
                             '{task_name}/routing_diagnostics.json); '
                             "pass 'none' to skip writing")
    return parser.parse_args()


def _fmt(value, digits=3):
    return '   -  ' if value is None else '{:.{}f}'.format(value, digits)


def print_summary(results):
    print('\nAP under each routing policy (same checkpoint, same capacity '
          'except "full")')
    print('  {:16s} {:>7s} {:>7s} {:>7s} {:>7s}'.format(
        'policy', 'AP', 'AP50', 'AP75', 'dAP'))
    base = results['maps']['default']['ap']
    for name, result in results['maps'].items():
        print('  {:16s} {:7.4f} {:7.4f} {:7.4f} {:+7.4f}'.format(
            name, result['ap'], result['ap50'], result['ap75'],
            result['ap'] - base))
    print('  NOTE: "full" is out of distribution for an expert-choice-trained '
          'model; a rise indicts routing, a drop proves nothing.')

    print('\nDecoder: matched (non-difficult) objects by recursion depth '
          '(capacity per stage: {})'.format(results['decoder_capacity']))
    print('  {:>5s} {:>6s} {:>6s} {:>8s} {:>8s} {:>9s} {:>8s} {:>8s}'.format(
        'depth', 'count', 'share', 'meanIoU', 'IoU>=.5', 'IoU>=.75', 'p(gt)',
        'objs/img'))
    for row in results['decoder_depth']:
        print('  {:5d} {:6d} {:6.3f} {:>8s} {:>8s} {:>9s} {:>8s} {:>8s}'.format(
            row['depth'], row['count'], row['share'], _fmt(row['mean_iou']),
            _fmt(row['frac_iou_ge_0.5']), _fmt(row['frac_iou_ge_0.75']),
            _fmt(row['mean_gt_prob']), _fmt(row['mean_objects_in_image'], 2)))

    print('\nEncoder: share of foreground / background tokens kept per stage')
    print('  {:>5s} {:>9s} {:>8s} {:>8s} {:>8s}'.format(
        'stage', 'capacity', 'fg kept', 'bg kept', 'fg/bg'))
    for row in results['encoder_foreground']:
        print('  {:5d} {:9.3f} {:>8s} {:>8s} {:>8s}'.format(
            row['stage'], row['capacity_fraction'], _fmt(row['fg_kept_rate']),
            _fmt(row['bg_kept_rate']), _fmt(row['fg_over_bg'], 2)))


def _json_safe(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_results(path, args, ckpt, results):
    payload = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': args.config,
        'checkpoint': ckpt,
        'limit': args.limit,
        'method': args.method,
        'policies': {name: PRESETS[name] for name in results['maps']},
        'maps': {
            name: {
                'ap': result['ap'], 'ap50': result['ap50'],
                'ap75': result['ap75'],
                'per_iou': {'{:.2f}'.format(t): r['mean_ap']
                            for t, r in result['per_iou'].items()},
            }
            for name, result in results['maps'].items()
        },
        'decoder_capacity': results['decoder_capacity'],
        'decoder_depth': results['decoder_depth'],
        'encoder_foreground': results['encoder_foreground'],
        'num_matched_objects': results['num_matched_objects'],
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
    print('Wrote results to {}'.format(path))


def main():
    args = parse_args()

    config = load_config(args.config, model_type='mor')
    dataset_config, model_config, train_config = split_config(config)

    set_seed(train_config['seed'])
    device = resolve_device(args.device)
    print('Using device: {}'.format(device))

    test_loader, voc = build_test_loader(dataset_config, train_config)
    model = build_model('mor', model_config, dataset_config, device=device,
                        pretrained_backbone=False)

    ckpt = args.ckpt or checkpoint_path(train_config)
    if not os.path.exists(ckpt):
        raise FileNotFoundError('No checkpoint exists at {}'.format(ckpt))
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    policies = [(name,) + PRESETS[name] for name in args.policies]
    results = run_routing_diagnostics(
        model, voc, test_loader, device, train_config, policies,
        method=args.method, iou_thresholds=COCO_IOU_THRESHOLDS,
        limit=args.limit, seed=args.seed)
    print_summary(results)

    if args.out != 'none':
        out = args.out or os.path.join(train_config['task_name'],
                                       'routing_diagnostics.json')
        write_results(out, args, ckpt, results)


if __name__ == '__main__':
    main()
