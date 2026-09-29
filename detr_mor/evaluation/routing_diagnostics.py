"""Evaluation-only diagnostics for the MoR routers of a trained MoRDETR.

Nothing here retrains or modifies weights. The routers' ``route`` method is
swapped per call, so one checkpoint can be scored under several routing
policies at identical capacity:

``default``
    The trained policy, unchanged.
``full``
    Every still-active token is kept at every stage (capacity = N), with the
    learned router scores still used as gates. This is OUT OF DISTRIBUTION for
    a model trained with expert-choice routing, so the test is asymmetric: an
    AP rise indicts routing, an AP drop proves nothing.
``random``
    Same capacity as ``default`` but the kept tokens are drawn at random
    (gates are still the learned scores). If this matches ``default`` the
    router is not selecting anything useful.
``oracle`` (decoder only)
    Same capacity as ``default``, but the queries Hungarian-matched to ground
    truth in the ``default`` pass are kept first. An upper bound on what a
    better decoder router could buy at the same compute.

Alongside AP, the ``default`` pass records, per matched query, how many middle
stages it took part in (its recursion depth) together with its final IoU and
ground-truth class probability, and, per encoder stage, how much of the kept
capacity falls on tokens inside ground-truth boxes.
"""

import contextlib
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.ops
from tqdm import tqdm

from detr_mor.evaluation.mean_ap import COCO_IOU_THRESHOLDS, compute_coco_map
from detr_mor.models.mor import _gather_tokens

#: Routing policies understood by :func:`routing_policy`.
POLICIES = ('default', 'full', 'random', 'oracle')

#: Added to the router score of oracle tokens so they always win the top-k.
_ORACLE_BONUS = 10.0


class _CachedBackbone(nn.Module):
    r"""
    Wraps the backbone so repeated forwards on the *same* image tensor reuse
    the first result. The backbone is frozen and dominates the cost, and every
    policy sees the same image, so this makes N policies cost about one
    backbone pass instead of N.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self._key = None
        self._out = None

    def forward(self, x):
        key = (x.data_ptr(), tuple(x.shape))
        if key != self._key:
            self._key = key
            self._out = self.backbone(x)
        return self._out


class _HeadRecorder:
    r"""Captures the raw class / box head outputs of the latest forward."""

    def __init__(self, model):
        self.cls = None
        self.box = None
        self._handles = [
            model.class_mlp.register_forward_hook(self._save_cls),
            model.bbox_mlp.register_forward_hook(self._save_box),
        ]

    def _save_cls(self, module, inputs, output):
        self.cls = output

    def _save_box(self, module, inputs, output):
        # bbox_mlp returns logits; the model applies the sigmoid afterwards.
        self.box = output.sigmoid()

    def remove(self):
        for handle in self._handles:
            handle.remove()


def _make_route(stack, policy, recorder, oracle_indices, generator):
    r"""
    Build a replacement for ``_MoRMiddleStack.route`` implementing ``policy``.

    Mirrors the original: capacity is a fraction of the full length, the active
    set can only shrink, and the kept indices come back in ascending order with
    their router scores as gates.
    """

    def route(stage, out, active_indices):
        num_tokens = out.size(1)
        topk = int(((stack.num_stages - stage) / stack.num_stages) * num_tokens)
        if policy == 'full':
            topk = active_indices.size(1)

        active_out = _gather_tokens(out, active_indices, stack.embed_dim)
        router = stack.exp_routers[stage - 1]
        scores = router.router_func(router.router_weights(active_out)).squeeze(-1)
        # scores -> (B, k_prev)

        if policy == 'random':
            priority = torch.rand(scores.shape, generator=generator).to(scores)
        elif policy == 'oracle' and oracle_indices is not None:
            is_oracle = torch.zeros_like(active_indices, dtype=torch.bool)
            for batch_idx, keep in enumerate(oracle_indices):
                if len(keep):
                    is_oracle[batch_idx] = torch.isin(
                        active_indices[batch_idx], keep.to(active_indices.device))
            priority = scores + _ORACLE_BONUS * is_oracle.to(scores)
        else:
            priority = scores

        _, rel_indices = torch.topk(priority, topk, dim=1)
        rel_indices, _ = torch.sort(rel_indices, dim=1)
        weights = torch.gather(scores, 1, rel_indices)

        active_indices = torch.gather(active_indices, 1, rel_indices)
        tokens = _gather_tokens(active_out, rel_indices, stack.embed_dim)
        if recorder is not None:
            recorder[stage] = active_indices.detach().cpu()
        return weights, active_indices, tokens

    return route


@contextlib.contextmanager
def routing_policy(model, encoder_policy='default', decoder_policy='default',
                   oracle_indices=None, seed=0):
    r"""
    Temporarily replace the encoder / decoder routers' selection rule.

    :param oracle_indices: list of B int64 tensors of query indices to keep
        first, used only when ``decoder_policy == 'oracle'``
    :return: yields ``{'encoder': {stage: (B, k)}, 'decoder': {...}}``, the
        absolute indices each stage kept during the forward(s) run inside
    """
    for policy in (encoder_policy, decoder_policy):
        if policy not in POLICIES:
            raise ValueError('policy must be one of {}, got {!r}'.format(
                POLICIES, policy))
    if encoder_policy == 'oracle':
        raise ValueError('the oracle policy is defined for the decoder only')

    generator = torch.Generator().manual_seed(seed)
    records = {'encoder': {}, 'decoder': {}}
    stacks = {
        'encoder': (model.encoder.middle_recursion_blocks, encoder_policy),
        'decoder': (model.decoder.middle_recursion_blocks, decoder_policy),
    }
    try:
        for name, (stack, policy) in stacks.items():
            # An instance attribute shadows the class method for this call only.
            stack.route = _make_route(stack, policy, records[name],
                                      oracle_indices, generator)
        yield records
    finally:
        for stack, _ in stacks.values():
            stack.__dict__.pop('route', None)


def _depth_per_token(records, num_stages, num_tokens):
    r"""
    Number of middle stages each token took part in, from the kept indices.
    Stage 0 is dense, so every token has depth >= 1.

    :return: (B, num_tokens) int64
    """
    batch_size = next(iter(records.values())).size(0) if records else 1
    depth = torch.ones(batch_size, num_tokens, dtype=torch.int64)
    for stage in range(1, num_stages):
        if stage in records:
            depth.scatter_add_(1, records[stage],
                               torch.ones_like(records[stage]))
    return depth


def _foreground_tokens(gt_boxes, feat_h, feat_w):
    r"""
    Mark encoder tokens whose cell centre falls inside any ground-truth box.

    :param gt_boxes: (N, 4) x1y1x2y2 in [0, 1]
    :return: (feat_h * feat_w,) bool
    """
    ys = (torch.arange(feat_h, dtype=torch.float32) + 0.5) / feat_h
    xs = (torch.arange(feat_w, dtype=torch.float32) + 0.5) / feat_w
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
    boxes = gt_boxes.cpu().float()
    inside = ((cx >= boxes[:, 0]) & (cx <= boxes[:, 2]) &
              (cy >= boxes[:, 1]) & (cy <= boxes[:, 3]))
    return inside.any(dim=1) if len(boxes) else torch.zeros(
        feat_h * feat_w, dtype=torch.bool)


def _bucket(preds, gts, difficults, dataset, detections, target_boxes,
            target_labels, difficult):
    r"""Append one image's detections / ground truth in compute_map's layout."""
    pred_boxes = {name: [] for name in dataset.label2idx}
    gt_boxes = {name: [] for name in dataset.label2idx}
    difficult_boxes = {name: [] for name in dataset.label2idx}
    for box, label, score in zip(detections['boxes'].tolist(),
                                 detections['labels'].tolist(),
                                 detections['scores'].tolist()):
        pred_boxes[dataset.idx2label[label]].append(box + [score])
    for box, label, diff in zip(target_boxes.tolist(), target_labels.tolist(),
                                difficult.tolist()):
        gt_boxes[dataset.idx2label[label]].append(box)
        difficult_boxes[dataset.idx2label[label]].append(diff)
    preds.append(pred_boxes)
    gts.append(gt_boxes)
    difficults.append(difficult_boxes)


@torch.no_grad()
def run_routing_diagnostics(model, dataset, loader, device, train_config,
                            policies, method='area',
                            iou_thresholds=COCO_IOU_THRESHOLDS, limit=None,
                            seed=0):
    r"""
    Score one MoRDETR checkpoint under several routing policies in one pass
    over the test set.

    :param policies: list of ``(name, encoder_policy, decoder_policy)``; the
        ``default`` / ``default`` entry is always run, since the oracle and the
        depth statistics are derived from it
    :param limit: stop after this many images (for a quick smoke run)
    :return: dict with 'maps' {name: compute_coco_map result},
        'decoder_depth' and 'encoder_foreground' statistics
    """
    model.eval()
    policies = [('default', 'default', 'default')] + [
        p for p in policies if p[0] != 'default']
    original_backbone = model.backbone
    model.backbone = _CachedBackbone(original_backbone)
    heads = _HeadRecorder(model)
    matcher = model.criterion.matcher
    score_thresh = train_config['eval_score_threshold']
    use_nms = train_config['use_nms_eval']

    buckets = {name: ([], [], []) for name, _, _ in policies}
    enc_stack = model.encoder.middle_recursion_blocks
    dec_stack = model.decoder.middle_recursion_blocks
    depth_rows = []  # one per matched ground-truth object
    enc_stats = defaultdict(lambda: {'fg_kept': 0, 'fg_total': 0,
                                     'bg_kept': 0, 'bg_total': 0})

    try:
        for image_idx, (im_tensor, target, _) in enumerate(
                tqdm(loader, desc='Routing diagnostics')):
            if limit is not None and image_idx >= limit:
                break
            im_tensor = im_tensor.float().to(device)
            target_boxes = target['boxes'].float()[0].to(device)
            target_labels = target['labels'].long()[0].to(device)
            difficult = target['difficult'].long()[0]
            oracle_indices = None

            for name, enc_policy, dec_policy in policies:
                with routing_policy(model, enc_policy, dec_policy,
                                    oracle_indices=oracle_indices,
                                    seed=seed + image_idx) as records:
                    detr_output = model(im_tensor, score_thresh=score_thresh,
                                        use_nms=use_nms)
                cls_last, box_last = heads.cls[-1], heads.box[-1]
                _bucket(*buckets[name], dataset, detr_output['detections'][0],
                        target_boxes, target_labels, difficult)

                if name != 'default' or len(target_labels) == 0:
                    continue

                # Everything below is measured on the trained policy only.
                (pred_idx, gt_idx), = matcher(
                    cls_last, box_last,
                    [{'boxes': target_boxes, 'labels': target_labels}])
                oracle_indices = [pred_idx]

                depth = _depth_per_token(records['decoder'],
                                         dec_stack.num_stages,
                                         model.num_queries)[0]
                matched_xyxy = torchvision.ops.box_convert(
                    box_last[0, pred_idx], 'cxcywh', 'xyxy')
                ious = torchvision.ops.box_iou(
                    matched_xyxy, target_boxes[gt_idx]).diagonal()
                probs = cls_last[0, pred_idx].softmax(-1)
                gt_prob = probs.gather(
                    1, target_labels[gt_idx].unsqueeze(1)).squeeze(1)
                for q, g, iou, prob in zip(pred_idx.tolist(), gt_idx.tolist(),
                                           ious.tolist(), gt_prob.tolist()):
                    depth_rows.append({
                        'depth': int(depth[q]),
                        'iou': iou,
                        'gt_prob': prob,
                        'difficult': int(difficult[g]),
                        'num_objects': len(target_labels),
                    })

                # How much of each encoder stage's kept capacity lands on
                # tokens inside ground-truth boxes.
                feat_h, feat_w = model.backbone._out.shape[-2:]
                is_fg = _foreground_tokens(target_boxes, feat_h, feat_w)
                for stage, kept in records['encoder'].items():
                    kept_mask = torch.zeros_like(is_fg)
                    kept_mask[kept[0]] = True
                    stats = enc_stats[stage]
                    stats['fg_kept'] += int((kept_mask & is_fg).sum())
                    stats['fg_total'] += int(is_fg.sum())
                    stats['bg_kept'] += int((kept_mask & ~is_fg).sum())
                    stats['bg_total'] += int((~is_fg).sum())
    finally:
        heads.remove()
        model.backbone = original_backbone

    maps = {}
    for name, (preds, gts, difficults) in buckets.items():
        maps[name] = compute_coco_map(preds, gts, difficult=difficults,
                                      method=method,
                                      iou_thresholds=iou_thresholds)

    return {
        'maps': maps,
        'decoder_depth': summarise_depth(depth_rows, dec_stack.num_stages),
        'decoder_capacity': [
            int(((dec_stack.num_stages - s) / dec_stack.num_stages)
                * model.num_queries)
            for s in range(dec_stack.num_stages)],
        'encoder_foreground': summarise_encoder(enc_stats, enc_stack.num_stages),
        'num_matched_objects': len(depth_rows),
    }


def summarise_depth(rows, num_stages):
    r"""
    Aggregate matched-query statistics by recursion depth, over non-difficult
    objects.

    :return: list of dicts, one per depth 1..num_stages
    """
    rows = [row for row in rows if not row['difficult']]
    total = max(len(rows), 1)
    table = []
    for depth in range(1, num_stages + 1):
        subset = [row for row in rows if row['depth'] == depth]
        count = len(subset)

        def mean(key, subset=subset):
            return sum(row[key] for row in subset) / count if count else None

        table.append({
            'depth': depth,
            'count': count,
            'share': count / total,
            'mean_iou': mean('iou'),
            'frac_iou_ge_0.5': (sum(row['iou'] >= 0.5 for row in subset) / count
                                if count else None),
            'frac_iou_ge_0.75': (sum(row['iou'] >= 0.75 for row in subset) / count
                                 if count else None),
            'mean_gt_prob': mean('gt_prob'),
            'mean_objects_in_image': mean('num_objects'),
        })
    return table


def summarise_encoder(stats, num_stages):
    r"""
    Per encoder stage: the kept fraction of foreground and background tokens,
    and how concentrated the kept set is on foreground relative to chance.
    """
    table = []
    for stage in range(1, num_stages):
        if stage not in stats:
            continue
        s = stats[stage]
        fg_rate = s['fg_kept'] / s['fg_total'] if s['fg_total'] else None
        bg_rate = s['bg_kept'] / s['bg_total'] if s['bg_total'] else None
        table.append({
            'stage': stage,
            'capacity_fraction': (num_stages - stage) / num_stages,
            'fg_kept_rate': fg_rate,
            'bg_kept_rate': bg_rate,
            # > 1 means the router prefers object tokens over background ones.
            'fg_over_bg': (fg_rate / bg_rate) if fg_rate and bg_rate else None,
        })
    return table
