"""Dump per-image test detections of an RT-DETR(-MoR) checkpoint.

Feeds the paired bootstrap in bootstrap.py. The model, test loader and metric
all come from the notebook the checkpoint was trained with (same loading path
as scripts/diagnose_rt_detr_routing.py), and the dump is scored with the
notebook's compute_coco_map so it can be checked against the stored
eval_results.json before any interval is computed from it.

Example::

    python paper/analysis/dump_rtdetr_dets.py --name 6L \
        --notebook rt_detr.ipynb \
        --config "rt_detr_all_results/rt_detr_new_results (6 blocks)/rt_detr_config.yaml" \
        --ckpt "rt_detr_all_results/rt_detr_new_results (6 blocks)/best_rt_detr.pth" \
        --out-dir <scratch>/dets
"""

import argparse
import datetime
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)  # configs hold repo-relative dataset paths

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from detr_mor.utils.notebook import load_notebook_namespace  # noqa: E402

COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


def parse_args():
    parser = argparse.ArgumentParser(description='Dump RT-DETR test detections')
    parser.add_argument('--name', required=True)
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--eval-json', default=None,
                        help='stored eval_results.json to check against')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda when available)')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    ns = load_notebook_namespace(
        args.notebook, stop_after=None,
        preset={'COCO_IOU_THRESHOLDS': COCO_IOU_THRESHOLDS},
        required=('DETR', 'build_test_loader', 'compute_coco_map',
                  'load_checkpoint', 'set_seed'))
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    ns['set_seed'](train_config['seed'])

    model_config = dict(config['model_params'], pretrained_backbone=False)
    model = ns['DETR'](model_config, num_classes=dataset_config['num_classes'])
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    ns['load_checkpoint'](args.ckpt, model, map_location=device)
    model.to(device).eval()
    model.convert_to_deploy()

    loader, voc = ns['build_test_loader'](dataset_config, train_config)
    score_thresh = train_config['eval_score_threshold']
    use_nms = train_config['use_nms_eval']

    det_boxes, det_scores, det_labels, det_image = [], [], [], []
    gt_boxes_all, gt_labels_all, gt_diff_all, gt_image = [], [], [], []
    preds, gts, diffs = [], [], []
    start = time.time()
    for image_idx, (im_tensor, target, _) in enumerate(loader):
        if args.limit is not None and image_idx >= args.limit:
            break
        out = model(im_tensor.float().to(device), score_thresh=score_thresh,
                    use_nms=use_nms)['detections'][0]
        boxes = out['boxes'].float().cpu().numpy()
        scores = out['scores'].float().cpu().numpy()
        labels = out['labels'].long().cpu().numpy()
        g_boxes = target['boxes'].float()[0].numpy()
        g_labels = target['labels'].long()[0].numpy()
        g_diff = target['difficult'].long()[0].numpy()

        det_boxes.append(boxes)
        det_scores.append(scores)
        det_labels.append(labels)
        det_image.append(np.full(len(scores), image_idx, dtype=np.int32))
        gt_boxes_all.append(g_boxes)
        gt_labels_all.append(g_labels)
        gt_diff_all.append(g_diff)
        gt_image.append(np.full(len(g_labels), image_idx, dtype=np.int32))

        # Same bucketing as the notebook's evaluate_on_test
        p = {name: [] for name in voc.label2idx}
        g = {name: [] for name in voc.label2idx}
        d = {name: [] for name in voc.label2idx}
        for box, label, score in zip(boxes.tolist(), labels.tolist(),
                                     scores.tolist()):
            p[voc.idx2label[label]].append(box + [score])
        for box, label, diff in zip(g_boxes.tolist(), g_labels.tolist(),
                                    g_diff.tolist()):
            g[voc.idx2label[label]].append(box)
            d[voc.idx2label[label]].append(diff)
        preds.append(p)
        gts.append(g)
        diffs.append(d)

        if image_idx % 250 == 0:
            rate = (time.time() - start) / (image_idx + 1)
            print('[{}] {} images, {:.2f} s/img'.format(args.name, image_idx + 1,
                                                       rate), flush=True)

    num_images = len(preds)
    npz_path = os.path.join(args.out_dir, '{}.npz'.format(args.name))
    np.savez_compressed(
        npz_path,
        det_boxes=np.concatenate(det_boxes), det_scores=np.concatenate(det_scores),
        det_labels=np.concatenate(det_labels), det_image=np.concatenate(det_image),
        gt_boxes=np.concatenate(gt_boxes_all), gt_labels=np.concatenate(gt_labels_all),
        gt_difficult=np.concatenate(gt_diff_all), gt_image=np.concatenate(gt_image),
        num_images=num_images,
        class_names=np.array([voc.idx2label[i] for i in range(len(voc.idx2label))]))

    result = ns['compute_coco_map'](preds, gts, difficult=diffs, method='area',
                                    iou_thresholds=COCO_IOU_THRESHOLDS)
    summary = {
        'name': args.name, 'checkpoint': args.ckpt, 'config': args.config,
        'notebook': args.notebook, 'num_images': num_images,
        'ap': result['ap'], 'ap50': result['ap50'], 'ap75': result['ap75'],
        'seconds': round(time.time() - start, 1),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    if args.eval_json and args.limit is None:
        with open(args.eval_json, encoding='utf-8') as handle:
            stored = json.load(handle)
        summary['stored_ap'] = stored['ap']
        summary['ap_abs_diff'] = abs(stored['ap'] - result['ap'])
        summary['reproduces_stored'] = summary['ap_abs_diff'] < 1e-4
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out_dir, '{}.json'.format(args.name)), 'w',
              encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
