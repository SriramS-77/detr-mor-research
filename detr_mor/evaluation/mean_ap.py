"""Pascal VOC style mean average precision."""

import numpy as np


def get_iou(det, gt):
    r"""
    IoU between two x1y1x2y2 boxes.

    :param det: (x1, y1, x2, y2) detection
    :param gt: (x1, y1, x2, y2) ground truth
    :return: float in [0, 1]
    """
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area',
                difficult=None):
    r"""
    Mean average precision over all classes present in the ground truth.

    Detections of a class are ranked by score; each is a true positive if it
    overlaps an as-yet-unmatched ground-truth box of that class by at least
    ``iou_threshold``, and a false positive otherwise.

    Expected shapes::

        det_boxes = [
          {
              'person' : [[x1, y1, x2, y2, score], ...],
              'car' : [[x1, y1, x2, y2, score], ...]
          },
          {det_boxes_img_2},
          ...
          {det_boxes_img_N},
        ]

        gt_boxes = [
          {
              'person' : [[x1, y1, x2, y2], ...],
              'car' : [[x1, y1, x2, y2], ...]
          },
          {gt_boxes_img_2},
          ...
          {gt_boxes_img_N},
        ]

    :param iou_threshold: IoU above which a detection counts as a match
    :param method: 'area' for the all-point interpolated AP, 'interp' for the
        11-point VOC2007 metric
    :param difficult: same structure as gt_boxes, holding the VOC 'difficult'
        flag per box. Difficult boxes are excluded from the metric on BOTH
        sides: they do not count toward the recall denominator, and a detection
        that lands on one is discarded rather than scored either way. Pass None
        to score every box.
    :return: (mean_ap, {class_name: ap}); a class with no scorable ground truth
        gets NaN and is left out of the mean.
    """
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])

        # VOC 'difficult' flags per image, aligned with gt_boxes[im][label]
        if difficult is not None:
            is_difficult = [difficults_label[label]
                            for difficults_label in difficult]
        else:
            is_difficult = [[0] * len(im_gts[label]) for im_gts in gt_boxes]
        num_difficults = sum([sum(flags) for flags in is_difficult])
        # Only the non-difficult boxes are on the hook for recall
        num_scored_gts = num_gts - num_difficults

        # Appended to rather than preallocated: a detection that lands on a
        # difficult box is dropped from the ranked list entirely, so these end
        # up shorter than cls_dets.
        tp = []
        fp = []

        # For each prediction
        for im_idx, det_pred in cls_dets:
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]

            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box. Difficult boxes stay in this argmax: a
            # detection sitting on a difficult object has to be recognised as
            # such before it can be excused.
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx

            if max_iou_found < iou_threshold:
                # Matches nothing -> false positive
                tp.append(0)
                fp.append(1)
            elif is_difficult[im_idx][max_iou_gt_idx]:
                # Landed on a box VOC marks 'difficult'. The protocol ignores
                # these: not credited as a hit, not penalised as a mistake, and
                # the gt is left unmatched so it can absorb further detections.
                #
                # Dropping it here is the half that used to be missing:
                # difficult boxes were removed from the recall DENOMINATOR
                # while still counting as true positives in the NUMERATOR, so
                # recall could reach num_gts / (num_gts - num_difficults) > 1
                # and the reported AP came out above 1.0.
                continue
            elif not gt_matched[im_idx][max_iou_gt_idx]:
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                # Already found by a higher-scoring detection -> duplicate
                tp.append(0)
                fp.append(1)

        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_scored_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # Recall is a fraction of the scorable ground truth; it cannot exceed 1.
        # Assert rather than clip, so any future mismatch between what counts as
        # a hit and what counts in the denominator fails loudly instead of
        # quietly reporting an AP above 1.
        assert recalls.size == 0 or recalls[-1] <= 1.0 + 1e-6, (
            'recall {:.4f} > 1 for class {}: {} true positives against {} '
            'scorable ground-truth boxes'.format(
                float(recalls[-1]), label, int(tp[-1]), num_scored_gts))

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r. This computes the precision envelope.
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = (prec_interp_pt.max()
                                  if prec_interp_pt.size > 0.0 else 0.0)
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        assert ap <= 1.0 + 1e-6, 'AP {:.4f} > 1 for class {}'.format(ap, label)

        # A class whose every ground-truth box is marked difficult has nothing
        # scorable, so it gets NaN rather than a meaningless 0.
        if num_scored_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold, over the scorable classes only
    mean_ap = sum(aps) / len(aps) if aps else np.nan
    return mean_ap, all_aps


# The ten IoU thresholds COCO averages over: 0.50, 0.55, ..., 0.95. Written out
# by construction rather than via np.arange(0.5, 1.0, 0.05), because the latter
# depends on floating-point rounding for whether it yields ten values or eleven.
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


def _nanmean(values):
    r"""Mean over the non-NaN entries; NaN when there are none."""
    kept = [value for value in values if value == value]
    return sum(kept) / len(kept) if kept else float('nan')


def compute_coco_map(det_boxes, gt_boxes, difficult=None, method='area',
                     iou_thresholds=COCO_IOU_THRESHOLDS):
    r"""
    COCO-style AP: :func:`compute_map` evaluated at several IoU thresholds and
    averaged, alongside the individual AP50 and AP75 numbers.

    Detections are collected once by the caller and re-scored at each threshold,
    so the sweep costs ten cheap matching passes, not ten forward passes.

    Note this is COCO-style only in the IoU sweep. The precision-recall curve is
    still integrated the VOC way ('area' = all-point, or 'interp' = 11-point);
    COCO itself uses a 101-point grid, and also applies a 100-detection cap and
    reports small/medium/large breakdowns, none of which are done here.

    :param det_boxes: list over images of ``{class_name: [[x1,y1,x2,y2,score]]}``
    :param gt_boxes: list over images of ``{class_name: [[x1,y1,x2,y2]]}``
    :param difficult: VOC 'difficult' flags, as :func:`compute_map` takes them
    :param method: 'area' or 'interp', passed through to :func:`compute_map`
    :param iou_thresholds: thresholds to average over; defaults to COCO's ten
    :return: dict with 'ap' (the average over thresholds), 'ap50', 'ap75',
        'class_ap' (per class, averaged the same way) and 'per_iou' keyed by
        threshold. Every value is a float, NaN where nothing was scorable.
    """
    per_iou = {}
    for iou_threshold in iou_thresholds:
        mean_ap, all_aps = compute_map(det_boxes, gt_boxes,
                                       iou_threshold=iou_threshold,
                                       method=method, difficult=difficult)
        per_iou[iou_threshold] = {'mean_ap': float(mean_ap),
                                  'class_ap': {name: float(ap)
                                               for name, ap in all_aps.items()}}

    # Whether a class is scorable depends only on its ground truth, not on the
    # threshold, so the label set is the same at every threshold. Taking the
    # union anyway keeps this correct if that ever stops being true.
    labels = sorted({name for result in per_iou.values()
                     for name in result['class_ap']})
    class_ap = {
        name: _nanmean([result['class_ap'].get(name, float('nan'))
                        for result in per_iou.values()])
        for name in labels
    }

    def _at(threshold):
        result = per_iou.get(threshold)
        return result['mean_ap'] if result is not None else float('nan')

    return {
        'ap': _nanmean([result['mean_ap'] for result in per_iou.values()]),
        'ap50': _at(0.5),
        'ap75': _at(0.75),
        'class_ap': class_ap,
        'per_iou': per_iou,
    }
