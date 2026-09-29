"""Evaluation-only routing diagnostics for an RT-DETR-MoR checkpoint.

The counterpart of scripts/diagnose_routing.py for the notebook model in
rt_detr_mor.ipynb, whose decoder starts from IoU-aware query selection instead
of zero-initialised queries. Comparing the two answers whether decoder routing
is only harmful when the early query states are poor.

Policies (decoder routers only; the model has no encoder routing):

``default``  the trained policy
``full``     every query recursed at every stage (out of distribution: a rise
             indicts routing, a drop proves nothing)
``random``   the trained capacity, random queries kept
``oracle``   the trained capacity, Hungarian-matched queries (from the
             ``default`` pass) kept first

Backbone, neck and query selection do not depend on the routing, so they run
once per image; the other policies re-run only the decoder on the captured
inputs and re-decode with the model's own top-k scheme. The re-decode is
checked against the model's detections on the first image.

Example::

    python scripts/diagnose_rt_detr_routing.py \
        --config rt_detr_mor_config.yaml \
        --ckpt "rt_detr_all_results/rt_detr_mor_voc_results (2 blocks, 3 recursions)/best_rt_detr_mor.pth"
"""

import argparse
import datetime
import json
import os

import _bootstrap  # noqa: F401  (sys.path setup)

import torch
import torchvision.ops
import yaml
from tqdm import tqdm

from detr_mor.evaluation.routing_diagnostics import summarise_depth
from detr_mor.utils.notebook import load_notebook_namespace

POLICIES = ('default', 'full', 'random', 'oracle')
_ORACLE_BONUS = 10.0
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


class RouterPatch:
    r"""
    Swaps every decoder router's selection rule and records, per routed stage,
    the absolute indices of the queries kept. Scores from the trained router
    are always used as the gates; only *which* queries are kept changes.
    """

    def __init__(self, decoder):
        self.routers = list(decoder.exp_routers)
        self.policy = 'default'
        self.oracle = None          # (B, n) absolute query indices, or None
        self.generator = torch.Generator().manual_seed(0)
        self.records = {}
        self._active = None
        self._stage = 0
        self._handle = decoder.register_forward_pre_hook(self._reset)
        for router in self.routers:
            router.forward = self._make_forward(router)

    def _reset(self, *_):
        self._active = None
        self._stage = 0
        self.records = {}

    def _make_forward(self, router):
        def forward(x, topk):
            scores = router.router_func(
                router.router_weights(router.norm(x))).squeeze(-1)
            if self._active is None:
                # At eval there are no denoising queries, so the routable set
                # is every query, in order.
                self._active = torch.arange(
                    x.size(1), device=x.device).unsqueeze(0).expand(
                        x.size(0), -1)
            if self.policy == 'full':
                topk = x.size(1)
            if self.policy == 'random':
                priority = torch.rand(scores.shape,
                                      generator=self.generator).to(scores)
            elif self.policy == 'oracle' and self.oracle is not None:
                is_oracle = torch.stack([
                    torch.isin(self._active[b], self.oracle[b].to(x.device))
                    for b in range(x.size(0))])
                priority = scores + _ORACLE_BONUS * is_oracle.to(scores)
            else:
                priority = scores
            _, rel_indices = torch.topk(priority, topk, dim=1)
            rel_indices, _ = torch.sort(rel_indices, dim=1)
            weights = torch.gather(scores, 1, rel_indices)

            self._active = torch.gather(self._active, 1, rel_indices)
            self._stage += 1
            self.records[self._stage] = self._active.detach().cpu()
            return weights, rel_indices
        return forward

    def remove(self):
        self._handle.remove()
        for router in self.routers:
            router.__dict__.pop('forward', None)


def decode(model, cls_last, box_last, score_thresh, use_nms):
    r"""
    The model's own inference decoding (see DETR.forward): top-k over the
    flattened (query x class) sigmoid scores, then optional class-wise NMS.
    """
    prob = cls_last.sigmoid()
    boxes = torchvision.ops.box_convert(box_last, 'cxcywh', 'xyxy')
    flat_scores = prob.flatten(1)
    topk = min(model.num_top_queries, flat_scores.shape[1])
    scores, flat_idx = torch.topk(flat_scores, topk, dim=1)
    query_idx = torch.div(flat_idx, model.num_classes, rounding_mode='floor')
    labels = flat_idx % model.num_classes
    boxes = boxes.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, 4))

    detections = []
    for b in range(boxes.shape[0]):
        keep = scores[b] >= score_thresh
        s, l, bx = scores[b][keep], labels[b][keep], boxes[b][keep]
        if use_nms:
            keep = torchvision.ops.batched_nms(bx, s, l,
                                               iou_threshold=model.nms_threshold)
            s, l, bx = s[keep], l[keep], bx[keep]
        detections.append({'boxes': bx, 'scores': s, 'labels': l})
    return detections


def bucket(store, dataset, detections, gt_boxes, gt_labels, difficult):
    preds, gts, diffs = store
    p = {name: [] for name in dataset.label2idx}
    g = {name: [] for name in dataset.label2idx}
    d = {name: [] for name in dataset.label2idx}
    for box, label, score in zip(detections['boxes'].tolist(),
                                 detections['labels'].tolist(),
                                 detections['scores'].tolist()):
        p[dataset.idx2label[label]].append(box + [score])
    for box, label, diff in zip(gt_boxes.tolist(), gt_labels.tolist(),
                                difficult.tolist()):
        g[dataset.idx2label[label]].append(box)
        d[dataset.idx2label[label]].append(diff)
    preds.append(p)
    gts.append(g)
    diffs.append(d)


@torch.no_grad()
def run(model, ns, dataset, loader, device, train_config, policies, limit):
    score_thresh = train_config['eval_score_threshold']
    use_nms = train_config['use_nms_eval']
    patch = RouterPatch(model.decoder)

    captured = {}

    def capture_inputs(module, args, kwargs):
        captured['args'], captured['kwargs'] = args, kwargs

    def capture_outputs(module, args, kwargs, output):
        captured['output'] = output

    handles = [
        model.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True),
        model.decoder.register_forward_hook(capture_outputs, with_kwargs=True),
    ]
    stores = {name: ([], [], []) for name in policies}
    depth_rows = []
    checked = False
    num_stages = len(patch.routers) + 1

    try:
        for image_idx, (im_tensor, target, _) in enumerate(
                tqdm(loader, desc='RT-DETR routing diagnostics')):
            if limit is not None and image_idx >= limit:
                break
            im_tensor = im_tensor.float().to(device)
            gt_boxes = target['boxes'].float()[0]
            gt_labels = target['labels'].long()[0]
            difficult = target['difficult'].long()[0]
            patch.generator.manual_seed(image_idx)
            patch.oracle = None

            for name in policies:
                patch.policy = name
                if name == 'default':
                    detr_output = model(im_tensor, score_thresh=score_thresh,
                                        use_nms=use_nms)
                    model_dets = detr_output['detections']
                    output, refs = captured['output']
                else:
                    output, refs = model.decoder(*captured['args'],
                                                 **captured['kwargs'])
                cls_last = model.class_heads[-1](output[-1])
                box_last = refs[-1]
                detections = decode(model, cls_last, box_last, score_thresh,
                                    use_nms)

                if name == 'default' and not checked:
                    ref = model_dets[0]
                    assert torch.allclose(ref['scores'], detections[0]['scores']) \
                        and torch.allclose(ref['boxes'], detections[0]['boxes']), \
                        're-decoding does not reproduce the model detections'
                    checked = True

                bucket(stores[name], dataset, detections[0], gt_boxes,
                       gt_labels, difficult)

                if name != 'default' or len(gt_labels) == 0:
                    continue
                records = dict(patch.records)
                targets = [{'boxes': gt_boxes.to(device),
                            'labels': gt_labels.to(device)}]
                (pred_idx, gt_idx), = model.match(cls_last, box_last, targets)
                pred_idx, gt_idx = pred_idx.cpu(), gt_idx.cpu()
                patch.oracle = [pred_idx]

                depth = torch.ones(model.num_queries, dtype=torch.int64)
                for kept in records.values():
                    depth[kept[0]] += 1
                ious = torchvision.ops.box_iou(
                    torchvision.ops.box_convert(box_last[0, pred_idx].cpu(),
                                                'cxcywh', 'xyxy'),
                    gt_boxes[gt_idx]).diagonal()
                gt_prob = cls_last[0, pred_idx].sigmoid().cpu().gather(
                    1, gt_labels[gt_idx].unsqueeze(1)).squeeze(1)
                for q, g, iou, prob in zip(pred_idx.tolist(), gt_idx.tolist(),
                                           ious.tolist(), gt_prob.tolist()):
                    depth_rows.append({'depth': int(depth[q]), 'iou': iou,
                                       'gt_prob': prob,
                                       'difficult': int(difficult[g]),
                                       'num_objects': len(gt_labels)})
    finally:
        for handle in handles:
            handle.remove()
        patch.remove()

    maps = {name: ns['compute_coco_map'](
        preds, gts, difficult=diffs, method='area',
        iou_thresholds=COCO_IOU_THRESHOLDS)
        for name, (preds, gts, diffs) in stores.items()}
    num_routable = model.num_queries
    capacity = [num_routable] + [
        max(1, int(((num_stages - s) / num_stages) * num_routable))
        for s in range(1, num_stages)]
    return {'maps': maps,
            'decoder_depth': summarise_depth(depth_rows, num_stages),
            'decoder_capacity': capacity,
            'num_matched_objects': len(depth_rows)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Routing diagnostics for an RT-DETR-MoR checkpoint')
    parser.add_argument('--notebook', default='rt_detr_mor.ipynb')
    parser.add_argument('--config', default='rt_detr_mor_config.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--policies', nargs='+', default=list(POLICIES),
                        choices=POLICIES)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out', default=None,
                        help="JSON output path; 'none' to skip (default: next "
                             'to the checkpoint)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    ns = load_notebook_namespace(
        args.notebook, stop_after=None,
        preset={'COCO_IOU_THRESHOLDS': COCO_IOU_THRESHOLDS},
        required=('DETR', 'VOCDataset', 'build_test_loader',
                  'compute_coco_map', 'load_checkpoint', 'set_seed'))

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    ns['set_seed'](train_config['seed'])

    model_config = dict(config['model_params'], pretrained_backbone=False)
    model = ns['DETR'](model_config, num_classes=dataset_config['num_classes'])
    ns['load_checkpoint'](args.ckpt, model, map_location=device)
    model.to(device).eval()
    model.convert_to_deploy()

    test_loader, voc = ns['build_test_loader'](dataset_config, train_config)
    policies = ['default'] + [p for p in args.policies if p != 'default']
    results = run(model, ns, voc, test_loader, device, train_config, policies,
                  args.limit)

    base = results['maps']['default']['ap']
    print('\nAP under each decoder routing policy')
    for name, r in results['maps'].items():
        print('  {:8s} AP {:.4f}  AP50 {:.4f}  AP75 {:.4f}  dAP {:+.4f}'.format(
            name, r['ap'], r['ap50'], r['ap75'], r['ap'] - base))
    print('\nMatched objects by recursion depth (capacity {})'.format(
        results['decoder_capacity']))
    for row in results['decoder_depth']:
        print('  depth {} n={:5d} share={:.3f} IoU={} IoU>=.75={} p(gt)={}'
              .format(row['depth'], row['count'], row['share'],
                      *['-' if row[k] is None else '{:.3f}'.format(row[k])
                        for k in ('mean_iou', 'frac_iou_ge_0.75',
                                  'mean_gt_prob')]))

    if args.out != 'none':
        out = args.out or os.path.join(os.path.dirname(args.ckpt),
                                       'routing_diagnostics.json')
        payload = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'config': args.config, 'checkpoint': args.ckpt,
            'limit': args.limit,
            'maps': {name: {'ap': r['ap'], 'ap50': r['ap50'], 'ap75': r['ap75'],
                            'per_iou': {'{:.2f}'.format(t): v['mean_ap']
                                        for t, v in r['per_iou'].items()}}
                     for name, r in results['maps'].items()},
            'decoder_capacity': results['decoder_capacity'],
            'decoder_depth': results['decoder_depth'],
            'num_matched_objects': results['num_matched_objects'],
        }
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        print('Wrote {}'.format(out))


if __name__ == '__main__':
    main()
