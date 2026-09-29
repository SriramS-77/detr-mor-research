"""mAP evaluation over the VOC test set."""

import torch
from tqdm import tqdm

from detr_mor.evaluation.mean_ap import (COCO_IOU_THRESHOLDS,
                                         compute_coco_map,
                                         compute_map)


@torch.no_grad()
def collect_predictions(model, dataset, loader, device, score_thresh=0.0,
                        use_nms=False):
    r"""
    Run the model over the test loader and bucket predictions and ground truth
    by class name, in the shape :func:`compute_map` expects.

    The loader must yield one image at a time with the default collate, so each
    target tensor carries a leading batch dimension.

    :param dataset: the VOCDataset behind the loader, for its idx2label mapping
    :return: (preds, gts, difficults)
    """
    gts = []
    preds = []
    difficults = []

    for im_tensor, target, _ in tqdm(loader, desc='Evaluating'):
        im_tensor = im_tensor.float().to(device)
        target_bboxes = target['boxes'].float()[0].to(device)
        target_labels = target['labels'].long()[0].to(device)
        difficult = target['difficult'].long()[0].to(device)

        detr_output = model(
            im_tensor,
            score_thresh=score_thresh,
            use_nms=use_nms
        )
        detections = detr_output['detections']

        boxes = detections[0]['boxes']
        labels = detections[0]['labels']
        scores = detections[0]['scores']

        pred_boxes = {}
        gt_boxes = {}
        difficult_boxes = {}

        for label_name in dataset.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
            difficult_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = dataset.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_bboxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = dataset.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
            difficult_boxes[label_name].append(
                difficult[idx].detach().cpu().item())

        gts.append(gt_boxes)
        preds.append(pred_boxes)
        difficults.append(difficult_boxes)

    return preds, gts, difficults


def evaluate_map(model, dataset, loader, device, train_config, method='area',
                 iou_threshold=0.5, verbose=True):
    r"""
    Compute and print class-wise APs and the mean AP.

    :param train_config: config['train_params']; reads 'eval_score_threshold'
        and 'use_nms_eval'
    :return: (mean_ap, {class_name: ap})
    """
    model.eval()
    preds, gts, difficults = collect_predictions(
        model, dataset, loader, device,
        score_thresh=train_config['eval_score_threshold'],
        use_nms=train_config['use_nms_eval'])

    mean_ap, all_aps = compute_map(preds, gts, iou_threshold=iou_threshold,
                                   method=method, difficult=difficults)

    if verbose:
        print('Class Wise Average Precisions')
        scored = 0
        for idx in range(len(dataset.idx2label)):
            label_name = dataset.idx2label[idx]
            if label_name not in all_aps:
                continue
            ap = all_aps[label_name]
            # NaN means the class had no scorable ground truth in this split -
            # either absent entirely, or present only as 'difficult' boxes.
            # Printing it as a number would read as a genuine AP of 0.
            if ap != ap:
                print('AP for class {:12s} = n/a (no scorable ground '
                      'truth)'.format(label_name))
            else:
                print('AP for class {:12s} = {:.4f}'.format(label_name, ap))
                scored += 1
        print('Mean Average Precision : {:.4f}  (over {} of {} classes)'.format(
            mean_ap, scored, len(dataset.idx2label)))

    return mean_ap, all_aps


def print_class_aps(all_aps, dataset, mean_ap, header):
    r"""
    Print the class-wise AP table for one IoU threshold.

    :param header: what to call this metric, e.g. 'AP50'
    """
    print('\n{} - class wise average precisions'.format(header))
    scored = 0
    for idx in range(len(dataset.idx2label)):
        label_name = dataset.idx2label[idx]
        if label_name not in all_aps:
            continue
        ap = all_aps[label_name]
        # NaN means the class had no scorable ground truth in this split -
        # either absent entirely, or present only as 'difficult' boxes.
        # Printing it as a number would read as a genuine AP of 0.
        if ap != ap:
            print('  {:12s} = n/a (no scorable ground truth)'.format(label_name))
        else:
            print('  {:12s} = {:.4f}'.format(label_name, ap))
            scored += 1
    print('  {:12s} = {:.4f}  (over {} of {} classes)'.format(
        'mean', mean_ap, scored, len(dataset.idx2label)))


def evaluate_coco_map(model, dataset, loader, device, train_config,
                      method='area', iou_thresholds=COCO_IOU_THRESHOLDS,
                      verbose=True):
    r"""
    Run the model over the test set once, then score it at every IoU threshold.

    Reports COCO-style AP (the average over the thresholds) together with the
    AP50 and AP75 that make it up, so a run is comparable both to VOC-style
    numbers and to COCO-style ones.

    :param train_config: config['train_params']; reads 'eval_score_threshold'
        and 'use_nms_eval'
    :return: the dict :func:`compute_coco_map` returns
    """
    model.eval()
    preds, gts, difficults = collect_predictions(
        model, dataset, loader, device,
        score_thresh=train_config['eval_score_threshold'],
        use_nms=train_config['use_nms_eval'])

    results = compute_coco_map(preds, gts, difficult=difficults, method=method,
                               iou_thresholds=iou_thresholds)

    if verbose:
        # The full per-class table only for the two thresholds anyone quotes.
        # Printing all ten would be 200 lines and bury the summary.
        for threshold, header in ((0.5, 'AP50'), (0.75, 'AP75')):
            if threshold in results['per_iou']:
                print_class_aps(results['per_iou'][threshold]['class_ap'],
                                dataset, results['per_iou'][threshold]['mean_ap'],
                                header)

        print('\nmAP by IoU threshold')
        for threshold in sorted(results['per_iou']):
            print('  IoU {:.2f} = {:.4f}'.format(
                threshold, results['per_iou'][threshold]['mean_ap']))

        print('\nCOCO-style AP - per class, averaged over IoU {:.2f}:{:.2f}'
              .format(min(results['per_iou']), max(results['per_iou'])))
        for idx in range(len(dataset.idx2label)):
            label_name = dataset.idx2label[idx]
            if label_name not in results['class_ap']:
                continue
            ap = results['class_ap'][label_name]
            print('  {:12s} = {}'.format(
                label_name,
                'n/a' if ap != ap else '{:.4f}'.format(ap)))

        print('\n{:>6s} = {:.4f}   (IoU {:.2f}:{:.2f}, {} thresholds)'.format(
            'AP', results['ap'], min(results['per_iou']),
            max(results['per_iou']), len(results['per_iou'])))
        print('{:>6s} = {:.4f}'.format('AP50', results['ap50']))
        print('{:>6s} = {:.4f}'.format('AP75', results['ap75']))

    return results
