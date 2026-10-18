"""Paired bootstrap confidence intervals for AP differences between detectors.

Input: per-image detection dumps written by dump_rtdetr_dets.py (one .npz per
model, all over the same test images in the same order).

Matching replicates compute_map in the notebooks / detr_mor/evaluation/mean_ap.py
exactly: per class, detections are ranked by score (stable sort, so ties keep
dump order), each is matched to the ground-truth box of highest IoU (first
maximum), a detection on a 'difficult' box is ignored, a second hit on an
already matched box is a false positive, and AP is the all-point ('area')
envelope. Because matching only interacts within an image, TP / FP / ignore
flags are computed once per image; a bootstrap resample then just re-weights
images by how often they were drawn, which is exactly equivalent to scoring
the resampled image multiset.

Sanity check: with every weight 1 the script reproduces the AP of the dump's
summary JSON (computed by the notebook's own compute_coco_map) and stops if it
does not.

Example::

    python latex_project/analysis/bootstrap.py --dets-dir <dir with 2L.npz 6L.npz 2x3.npz 3x2.npz> \
        --pairs 2x3:6L 3x2:6L 2x3:2L 6L:2L --B 1000
"""

import argparse
import json
import os

import numpy as np

THRESHOLDS = np.array([round(0.5 + 0.05 * k, 2) for k in range(10)])
EPS = np.finfo(np.float32).eps
# keys used by make_tables.py for the manuscript
KEY = {'2L': 'rt2', '6L': 'rt6', '2x3': 'rt23', '3x2': 'rt32'}


def iou_matrix(det, gt):
    r"""Pairwise IoU, same arithmetic (and order of operations) as get_iou."""
    x_left = np.maximum(det[:, None, 0], gt[None, :, 0])
    y_top = np.maximum(det[:, None, 1], gt[None, :, 1])
    x_right = np.minimum(det[:, None, 2], gt[None, :, 2])
    y_bottom = np.minimum(det[:, None, 3], gt[None, :, 3])
    valid = ~((x_right < x_left) | (y_bottom < y_top))
    inter = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = det_area[:, None] + gt_area[None, :] - inter + 1e-6
    return np.where(valid, inter / union, 0.0)


class Scored:
    r"""Per-detection TP(1) / FP(0) / ignored(-1) flags at every threshold."""

    def __init__(self, path):
        z = np.load(path)
        self.num_images = int(z['num_images'])
        boxes = z['det_boxes'].astype(np.float64)
        scores = z['det_scores'].astype(np.float64)
        labels = z['det_labels'].astype(np.int64)
        image = z['det_image'].astype(np.int64)
        g_boxes = z['gt_boxes'].astype(np.float64)
        g_labels = z['gt_labels'].astype(np.int64)
        g_diff = z['gt_difficult'].astype(np.int64)
        g_image = z['gt_image'].astype(np.int64)
        self.num_classes = len(z['class_names'])

        flags = np.zeros((len(scores), len(THRESHOLDS)), dtype=np.int8)
        det_start = np.searchsorted(image, np.arange(self.num_images + 1))
        gt_start = np.searchsorted(g_image, np.arange(self.num_images + 1))
        self.npos = np.zeros((self.num_images, self.num_classes))
        for i in range(self.num_images):
            ds, de = det_start[i], det_start[i + 1]
            gs, ge = gt_start[i], gt_start[i + 1]
            for c in range(self.num_classes):
                gmask = g_labels[gs:ge] == c
                self.npos[i, c] = np.sum(gmask & (g_diff[gs:ge] == 0))
                didx = ds + np.where(labels[ds:de] == c)[0]
                if len(didx) == 0:
                    continue
                if not gmask.any():
                    continue  # flags already 0 (false positive)
                # within-image ranking = global ranking restricted to it
                didx = didx[np.argsort(-scores[didx], kind='stable')]
                gb = g_boxes[gs:ge][gmask]
                gd = g_diff[gs:ge][gmask]
                ious = iou_matrix(boxes[didx], gb)
                best = np.argmax(ious, axis=1)        # first maximum
                best_iou = ious[np.arange(len(didx)), best]
                for t_idx, t in enumerate(THRESHOLDS):
                    matched = np.zeros(len(gb), dtype=bool)
                    for k, d in enumerate(didx):
                        j = best[k]
                        if best_iou[k] < t:
                            flags[d, t_idx] = 0
                        elif gd[j]:
                            flags[d, t_idx] = -1
                        elif not matched[j]:
                            matched[j] = True
                            flags[d, t_idx] = 1
                        else:
                            flags[d, t_idx] = 0
        # global per-class ranking (stable: ties keep dump order, as sorted())
        self.classes = []
        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            idx = idx[np.argsort(-scores[idx], kind='stable')]
            self.classes.append((image[idx], flags[idx]))

    def ap(self, weights):
        r"""(AP, AP50, AP75) for image weights (counts in the resample)."""
        per_t = np.zeros(len(THRESHOLDS))
        num_scored = 0
        for c, (img, fl) in enumerate(self.classes):
            npos = float(np.dot(weights, self.npos[:, c]))
            if npos <= 0:
                continue
            num_scored += 1
            if len(img) == 0:
                continue
            w = weights[img][:, None]
            tp = np.cumsum((fl == 1) * w, axis=0)
            fp = np.cumsum((fl == 0) * w, axis=0)
            rec = tp / max(npos, EPS)
            prec = tp / np.maximum(tp + fp, EPS)
            zeros = np.zeros((1, len(THRESHOLDS)))
            rec = np.vstack([zeros, rec, zeros + 1.0])
            prec = np.vstack([zeros, prec, zeros])
            prec = np.maximum.accumulate(prec[::-1], axis=0)[::-1]
            per_t += np.sum((rec[1:] - rec[:-1]) * prec[1:], axis=0)
        per_t /= num_scored
        return per_t.mean(), per_t[0], per_t[5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dets-dir', required=True)
    parser.add_argument('--pairs', nargs='+', default=['2x3:6L', '3x2:6L', '2x3:2L', '6L:2L'])
    parser.add_argument('--B', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'out', 'bootstrap.json'))
    args = parser.parse_args()

    names = sorted({n for p in args.pairs for n in p.split(':')})
    models = {}
    for name in names:
        models[name] = Scored(os.path.join(args.dets_dir, name + '.npz'))
        full = models[name].ap(np.ones(models[name].num_images))
        with open(os.path.join(args.dets_dir, name + '.json'), encoding='utf-8') as handle:
            ref = json.load(handle)
        print('{}: AP {:.6f} (notebook {:.6f})'.format(name, full[0], ref['ap']))
        assert abs(full[0] - ref['ap']) < 1e-6 and abs(full[1] - ref['ap50']) < 1e-6 \
            and abs(full[2] - ref['ap75']) < 1e-6, 'bootstrap scorer does not reproduce the notebook AP'
    num_images = {m.num_images for m in models.values()}
    assert len(num_images) == 1, 'dumps cover different image sets'
    n = num_images.pop()

    rng = np.random.default_rng(args.seed)
    samples = {name: np.zeros((args.B, 3)) for name in names}
    for b in range(args.B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for name, model in models.items():
            samples[name][b] = model.ap(w)

    result = {'B': args.B, 'seed': args.seed, 'num_images': n, 'pairs': {}}
    for pair in args.pairs:
        a, b = pair.split(':')
        point = np.array(models[a].ap(np.ones(n))) - np.array(models[b].ap(np.ones(n)))
        diff = 100 * (samples[a] - samples[b])
        entry = {}
        for k, metric in enumerate(('ap', 'ap50', 'ap75')):
            lo, hi = np.percentile(diff[:, k], [2.5, 97.5])
            entry[metric] = {'delta': 100 * point[k], 'lo': lo, 'hi': hi,
                             'p_le_0': float(np.mean(diff[:, k] <= 0))}
        result['pairs']['{}-{}'.format(KEY.get(a, a), KEY.get(b, b))] = entry
        print('{}: dAP {:+.2f} [{:+.2f}, {:+.2f}]  P(d<=0)={:.3f}'.format(
            pair, entry['ap']['delta'], entry['ap']['lo'], entry['ap']['hi'],
            entry['ap']['p_le_0']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
