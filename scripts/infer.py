"""Save qualitative detection samples from a trained checkpoint.

Requires opencv-python. Example::

    python scripts/infer.py --config configs/mor_voc.yaml --model mor --num-samples 5
"""

import argparse
import os

import _bootstrap  # noqa: F401  (sys.path setup)

from detr_mor.config import load_config, split_config
from detr_mor.data import VOCDataset
from detr_mor.engine import checkpoint_path, load_checkpoint
from detr_mor.evaluation import infer
from detr_mor.models import build_model
from detr_mor.utils import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Save sample detections from a DETR / DETR-MoR checkpoint')
    parser.add_argument('--config', required=True,
                        help='path to the yaml config')
    parser.add_argument('--model', choices=('detr', 'mor'), default='mor',
                        help='which architecture to run (default: mor)')
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda when available)')
    parser.add_argument('--ckpt', default=None,
                        help='checkpoint path (default: {task_name}/{ckpt_name})')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='how many test images to draw')
    parser.add_argument('--output-dir', default='samples',
                        help='where to write the images')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config, model_type=args.model)
    print(config)
    dataset_config, model_config, train_config = split_config(config)

    set_seed(train_config['seed'])
    device = resolve_device(args.device)
    print('Using device: {}'.format(device))

    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])

    model = build_model(args.model, model_config, dataset_config, device=device,
                        pretrained_backbone=False)

    ckpt = args.ckpt or checkpoint_path(train_config)
    if not os.path.exists(ckpt):
        raise FileNotFoundError('No checkpoint exists at {}'.format(ckpt))
    load_checkpoint(ckpt, model, map_location=device)

    infer(model, voc, device, train_config,
          num_samples=args.num_samples,
          output_dir=args.output_dir,
          model_label=args.model,
          seed=train_config['seed'])


if __name__ == '__main__':
    main()
