"""CPU smoke test: run one batch of synthetic data end to end.

Needs no VOC dataset, no GPU, no checkpoint and no network access. Run this
before shipping the code to a training machine::

    python scripts/smoke_test.py --no-pretrained

It exercises, for each model, the same code paths a real run uses:
forward + loss, backward, an optimizer step, eval-mode inference (with and
without NMS), the mAP computation, and a checkpoint save/load round trip.

Exits non-zero if any check fails, so it can be dropped into CI.
"""

import argparse
import copy
import os
import tempfile

import _bootstrap  # noqa: F401  (sys.path setup)
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from detr_mor.data import collate_function
from detr_mor.engine import (
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from detr_mor.evaluation import compute_map
from detr_mor.models import MoREncoder, build_model
from detr_mor.utils import batch_images_to_device, set_seed, targets_to_device

# Deliberately tiny so the whole thing runs in seconds on a laptop CPU.
# num_queries stays at the real value because the matcher and the dataset's
# max-objects filter are written around it.
SMOKE_DATASET_CONFIG = {
    'num_classes': 21,
    'bg_class_idx': 0,
    'im_size': 128,
}

SMOKE_MODEL_CONFIG = {
    'im_channels': 3,
    'backbone_channels': 512,
    'd_model': 64,
    'num_queries': 25,
    'freeze_backbone': True,
    'encoder_layers': 2,
    'encoder_attn_heads': 4,
    'decoder_layers': 2,
    'decoder_attn_heads': 4,
    # MoR: Middle-* over 4 blocks (2 unique + 2 shared) x 2 recursions ==
    # effective depth 2 + 2*2 = 6. Must be >= 3 blocks or the middle group is
    # empty; run_checks() overrides the schedules per variant.
    'encoder_num_blocks': 4,
    'encoder_num_recursions': 2,
    'encoder_recursion_type': 'cyclic',
    'decoder_num_blocks': 4,
    'decoder_num_recursions': 2,
    'decoder_recursion_type': 'cyclic',
    'dropout_prob': 0.1,
    'ff_inner_dim': 128,
    'cls_cost_weight': 1.,
    'l1_cost_weight': 5.,
    'giou_cost_weight': 2.,
    'bg_class_weight': 0.1,
    'nms_threshold': 0.5,
}

SMOKE_TRAIN_CONFIG = {
    'task_name': 'smoke_test_run',
    'ckpt_name': 'smoke.pth',
    'seed': 1111,
    'acc_steps': 1,
    'num_epochs': 1,
    'batch_size': 2,
    'lr_steps': [200],
    'lr': 1e-4,
    'log_steps': 1,
    'eval_score_threshold': 0.0,
    'infer_score_threshold': 0.5,
    'use_nms_eval': False,
    'use_nms_infer': True,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SyntheticDetectionDataset(Dataset):
    r"""
    Random data in exactly the tensor contract
    :meth:`detr_mor.data.voc.VOCDataset.__getitem__` produces:

      * ``im_tensor``: (3, im_size, im_size) float32, ImageNet-normalised
      * ``targets['boxes']``: (N, 4) float32, x1y1x2y2 in [0, 1], x1<x2 and y1<y2
      * ``targets['labels']``: (N,) int64 foreground labels
      * ``targets['difficult']``: (N,) int64
      * ``filename``: str

    N deliberately varies per sample so the ragged-target path (collate,
    per-image cost split, Hungarian assignment) is actually exercised.

    :param num_classes: including background at index ``bg_class_idx``
    :param max_objects: N is drawn from 1..max_objects
    """

    def __init__(self, length=4, im_size=128, num_classes=21, bg_class_idx=0,
                 max_objects=4, seed=0):
        self.length = length
        self.im_size = im_size
        self.num_classes = num_classes
        self.bg_class_idx = bg_class_idx
        self.max_objects = max_objects
        self.seed = seed

        # Same 21-name mapping shape the evaluation code expects.
        self.idx2label = {idx: 'class_{}'.format(idx)
                          for idx in range(num_classes)}
        self.idx2label[bg_class_idx] = 'background'
        self.label2idx = {name: idx for idx, name in self.idx2label.items()}

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._mean, self._std = mean, std

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)

        image = torch.rand(3, self.im_size, self.im_size, generator=generator)
        im_tensor = (image - self._mean) / self._std

        num_objects = int(torch.randint(1, self.max_objects + 1, (1,),
                                        generator=generator).item())

        # Build valid boxes: sample a top-left corner in [0, 0.8] and a positive
        # width/height, so x1 < x2 and y1 < y2 always hold.
        xy1 = torch.rand(num_objects, 2, generator=generator) * 0.8
        wh = 0.05 + torch.rand(num_objects, 2, generator=generator) * 0.15
        boxes = torch.cat([xy1, (xy1 + wh).clamp(max=1.0)], dim=1).float()

        # Foreground labels only; index 0 (or num_classes-1) is background.
        fg_labels = [idx for idx in range(self.num_classes)
                     if idx != self.bg_class_idx]
        label_positions = torch.randint(0, len(fg_labels), (num_objects,),
                                        generator=generator)
        labels = torch.tensor([fg_labels[i] for i in label_positions],
                              dtype=torch.int64)

        difficult = torch.zeros(num_objects, dtype=torch.int64)

        targets = {'boxes': boxes, 'labels': labels, 'difficult': difficult}
        return im_tensor, targets, 'synthetic_{}.jpg'.format(index)


class CheckReporter:
    """Prints PASS/FAIL per check and remembers whether anything failed."""

    def __init__(self):
        self.failures = []

    def check(self, name, fn):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - the point is to report anything
            self.failures.append(name)
            print('  FAIL  {}: {}: {}'.format(name, type(exc).__name__, exc))
            return False
        print('  PASS  {}{}'.format(name, '' if not detail else
                                    ' ({})'.format(detail)))
        return True


def _batch(dataset, batch_size, device):
    """Pull one collated batch off the dataset, moved to ``device``."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_function)
    ims, targets, _ = next(iter(loader))
    return (batch_images_to_device(ims, device),
            targets_to_device(list(targets), device))


def run_checks(model_type, device, pretrained, reporter, model_config=None,
               label=None):
    model_config = model_config or SMOKE_MODEL_CONFIG
    print('\n=== {} ==='.format(label or model_type.upper()))
    set_seed(SMOKE_TRAIN_CONFIG['seed'])

    dataset = SyntheticDetectionDataset(
        length=4,
        im_size=SMOKE_DATASET_CONFIG['im_size'],
        num_classes=SMOKE_DATASET_CONFIG['num_classes'],
        bg_class_idx=SMOKE_DATASET_CONFIG['bg_class_idx'])

    model = build_model(model_type, model_config, SMOKE_DATASET_CONFIG,
                        device=device, pretrained_backbone=pretrained)
    batch_size = SMOKE_TRAIN_CONFIG['batch_size']
    images, targets = _batch(dataset, batch_size, device)

    num_layers = model.num_decoder_layers
    state = {}

    def check_forward_loss():
        model.train()
        output = model(images, targets)
        losses = output['loss']
        assert set(losses) == {'classification', 'bbox_regression'}, \
            'unexpected loss keys: {}'.format(sorted(losses))
        for key, values in losses.items():
            assert len(values) == num_layers, \
                '{}: expected {} entries (one per decoder layer), got {}'.format(
                    key, num_layers, len(values))
            for layer_idx, value in enumerate(values):
                assert value.ndim == 0, \
                    '{}[{}] is not a scalar'.format(key, layer_idx)
                assert torch.isfinite(value), \
                    '{}[{}] is not finite: {}'.format(key, layer_idx, value)
                assert value.requires_grad, \
                    '{}[{}] is detached from the graph'.format(key, layer_idx)
        total = (sum(losses['classification']) + sum(losses['bbox_regression']))
        state['loss'] = total
        return 'depth {}, total loss {:.4f}'.format(num_layers, total.item())

    def check_backward():
        state['loss'].backward()
        trainable = [(name, param) for name, param in model.named_parameters()
                     if param.requires_grad]
        assert trainable, 'model has no trainable parameters'

        with_grad = [(name, param) for name, param in trainable
                     if param.grad is not None]
        assert with_grad, 'no parameter received a gradient'

        for name, param in with_grad:
            assert torch.isfinite(param.grad).all(), \
                'non-finite gradient in {}'.format(name)

        nonzero = [name for name, param in with_grad
                   if param.grad.abs().sum() > 0]
        assert nonzero, 'every gradient is exactly zero'
        return '{}/{} trainable tensors have non-zero grads'.format(
            len(nonzero), len(trainable))

    def check_optimizer_step():
        optimizer = build_optimizer(model, SMOKE_TRAIN_CONFIG)
        assert optimizer.param_groups[0]['params'], \
            'optimizer got an empty parameter list'
        watched = [param for param in model.parameters()
                   if param.requires_grad and param.grad is not None][:5]
        before = [param.detach().clone() for param in watched]
        optimizer.step()
        moved = sum(1 for prev, param in zip(before, watched)
                    if not torch.equal(prev, param))
        assert moved > 0, 'no parameter changed after an optimizer step'
        optimizer.zero_grad()
        return '{}/{} sampled params moved'.format(moved, len(watched))

    def check_train_one_epoch():
        # The real loop, over the whole tiny dataset: covers accumulation,
        # logging, the NaN guard and the end-of-epoch flush.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_function)
        optimizer = build_optimizer(model, SMOKE_TRAIN_CONFIG)
        scheduler = build_scheduler(optimizer, SMOKE_TRAIN_CONFIG)
        cls_loss, loc_loss, steps, _ = train_one_epoch(
            model, loader, optimizer, scheduler, device, SMOKE_TRAIN_CONFIG,
            steps=0, label=model_type.upper())
        assert steps == len(loader), \
            'expected {} steps, counted {}'.format(len(loader), steps)
        state['optimizer'] = optimizer
        state['scheduler'] = scheduler
        return 'cls {:.4f}, loc {:.4f} over {} steps'.format(
            cls_loss, loc_loss, steps)

    def _check_detections(detections, expect_batch):
        assert isinstance(detections, list), 'detections is not a list'
        assert len(detections) == expect_batch, \
            'expected {} detection dicts, got {}'.format(expect_batch,
                                                         len(detections))
        for image_idx, detection in enumerate(detections):
            boxes = detection['boxes']
            scores = detection['scores']
            labels = detection['labels']
            num_dets = boxes.shape[0]
            assert boxes.shape == (num_dets, 4), \
                'image {}: boxes shape {}'.format(image_idx, tuple(boxes.shape))
            assert scores.shape == (num_dets,), \
                'image {}: scores shape {} vs {} boxes'.format(
                    image_idx, tuple(scores.shape), num_dets)
            assert labels.shape == (num_dets,), \
                'image {}: labels shape {} vs {} boxes'.format(
                    image_idx, tuple(labels.shape), num_dets)
            assert labels.dtype == torch.int64, \
                'image {}: labels dtype {}'.format(image_idx, labels.dtype)
            assert torch.isfinite(boxes).all(), \
                'image {}: non-finite box coordinates'.format(image_idx)
            assert (labels != SMOKE_DATASET_CONFIG['bg_class_idx']).all(), \
                'image {}: a detection was labelled background'.format(image_idx)
        return detections

    def check_inference():
        model.eval()
        with torch.no_grad():
            output = model(images, targets=None, score_thresh=0.0,
                           use_nms=False)
        detections = _check_detections(output['detections'], batch_size)
        counts = [int(d['boxes'].shape[0]) for d in detections]
        assert all(count == model_config['num_queries']
                   for count in counts), \
            'with score_thresh=0 every query should survive, got {}'.format(counts)
        state['detections'] = detections
        return 'detections per image: {}'.format(counts)

    def check_inference_nms():
        model.eval()
        with torch.no_grad():
            output = model(images, targets=None, score_thresh=0.0, use_nms=True)
        detections = _check_detections(output['detections'], batch_size)
        counts = [int(d['boxes'].shape[0]) for d in detections]
        assert all(count <= model_config['num_queries']
                   for count in counts), \
            'NMS returned more boxes than queries: {}'.format(counts)
        return 'detections per image after NMS: {}'.format(counts)

    def check_eval_mode_returns_loss():
        # The validation pass relies on eval mode still producing losses.
        model.eval()
        with torch.no_grad():
            output = model(images, targets)
        assert 'loss' in output, 'eval-mode forward produced no loss'
        assert 'detections' in output, 'eval-mode forward produced no detections'
        return 'loss and detections both present'

    def check_map():
        # Feed the real detections plus the batch's ground truth through the
        # mAP code, in the bucketed-by-class-name form it expects.
        names = list(dataset.idx2label.values())
        preds, gts, difficults = [], [], []
        for image_idx, detection in enumerate(state['detections']):
            pred_boxes = {name: [] for name in names}
            gt_boxes = {name: [] for name in names}
            difficult_boxes = {name: [] for name in names}

            for box, label, score in zip(detection['boxes'],
                                         detection['labels'],
                                         detection['scores']):
                x1, y1, x2, y2 = box.tolist()
                pred_boxes[dataset.idx2label[int(label)]].append(
                    [x1, y1, x2, y2, float(score)])

            target = targets[image_idx]
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.tolist()
                name = dataset.idx2label[int(label)]
                gt_boxes[name].append([x1, y1, x2, y2])
                difficult_boxes[name].append(0)

            preds.append(pred_boxes)
            gts.append(gt_boxes)
            difficults.append(difficult_boxes)

        mean_ap, all_aps = compute_map(preds, gts, method='area',
                                       difficult=difficults)
        assert isinstance(all_aps, dict) and all_aps, 'no per-class APs returned'
        assert 0.0 <= mean_ap <= 1.0, 'mAP out of range: {}'.format(mean_ap)
        # Random weights, so a near-zero mAP here is expected and fine - the
        # point is that the code path runs.
        return 'mAP {:.4f} over {} classes'.format(mean_ap, len(all_aps))

    def check_checkpoint_roundtrip():
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'smoke.pth')
            save_checkpoint(model, state['optimizer'], state['scheduler'],
                            epoch=1, loss=0.5, steps=7, path=path)
            assert os.path.exists(path), 'checkpoint file was not written'

            original = copy.deepcopy(model.state_dict())

            reloaded = build_model(model_type, model_config,
                                   SMOKE_DATASET_CONFIG, device=device,
                                   pretrained_backbone=False)
            optimizer = build_optimizer(reloaded, SMOKE_TRAIN_CONFIG)
            scheduler = build_scheduler(optimizer, SMOKE_TRAIN_CONFIG)
            epoch, steps = load_checkpoint(path, reloaded, optimizer, scheduler,
                                           map_location=device)

            assert epoch == 1, 'epoch not restored: {}'.format(epoch)
            assert steps == 7, 'steps not restored: {}'.format(steps)

            restored = reloaded.state_dict()
            assert set(restored) == set(original), 'state_dict keys differ'
            for key in original:
                assert torch.equal(original[key], restored[key]), \
                    'tensor {} differs after reload'.format(key)
        return 'epoch/steps restored, {} tensors match'.format(len(original))

    reporter.check('forward + loss', check_forward_loss)
    reporter.check('backward', check_backward)
    reporter.check('optimizer step', check_optimizer_step)
    reporter.check('train_one_epoch', check_train_one_epoch)
    reporter.check('inference (no nms)', check_inference)
    reporter.check('inference (nms)', check_inference_nms)
    reporter.check('eval mode still returns loss', check_eval_mode_returns_loss)
    reporter.check('mAP computation', check_map)
    reporter.check('checkpoint round trip', check_checkpoint_roundtrip)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CPU smoke test for DETR / DETR-MoR on synthetic data')
    parser.add_argument('--model', choices=('detr', 'mor', 'both'),
                        default='both', help='which model(s) to check')
    parser.add_argument('--device', default='cpu',
                        help='torch device (default: cpu)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='skip the ImageNet backbone download (use this '
                             'offline; weights do not matter for a smoke test)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_types = (('detr', 'mor') if args.model == 'both' else (args.model,))

    print('Smoke test on device: {} | pretrained backbone: {}'.format(
        device, not args.no_pretrained))

    reporter = CheckReporter()
    for model_type in model_types:
        if model_type != 'mor':
            run_checks(model_type, device, not args.no_pretrained, reporter)
            continue
        # Each recursion schedule is a separate code path, so check both.
        for recursion_type in MoREncoder.RECURSION_TYPES:
            config = dict(SMOKE_MODEL_CONFIG,
                          encoder_recursion_type=recursion_type,
                          decoder_recursion_type=recursion_type)
            run_checks(model_type, device, not args.no_pretrained, reporter,
                       model_config=config,
                       label='MOR ({} recursion)'.format(recursion_type))

    print()
    if reporter.failures:
        print('FAILED {} check(s): {}'.format(len(reporter.failures),
                                              ', '.join(reporter.failures)))
        raise SystemExit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
