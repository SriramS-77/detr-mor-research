"""Turn raw decoder outputs into per-image detections."""

import torch
import torchvision.ops


def postprocess_detections(cls_output, bbox_output, bg_class_idx,
                           score_thresh=0.0, use_nms=False, nms_threshold=0.5):
    r"""
    Convert the final decoder layer's outputs into detection dicts.

    DETR is trained with one-to-one matching, so NMS is not required; it is
    exposed because the configs enable it for qualitative inference.

    :param cls_output: (B, num_queries, num_classes) logits, last decoder layer
    :param bbox_output: (B, num_queries, 4) cxcywh in [0, 1], last decoder layer
    :param bg_class_idx: index of the background class (0 or num_classes-1);
        the background column is excluded when picking each query's label
    :param score_thresh: drop detections scoring below this
    :param use_nms: run class-wise NMS over the survivors
    :param nms_threshold: IoU threshold for that NMS
    :return: list of B dicts with 'boxes' (K, 4) xyxy in [0, 1], 'scores' (K,)
        and 'labels' (K,)
    """
    prob = torch.nn.functional.softmax(cls_output, -1)

    # Get all query boxes and their best fg class as label
    if bg_class_idx == 0:
        scores, labels = prob[..., 1:].max(-1)
        labels = labels + 1
    else:
        scores, labels = prob[..., :-1].max(-1)

    # convert to x1y1x2y2 format
    boxes = torchvision.ops.box_convert(bbox_output,
                                        'cxcywh',
                                        'xyxy')

    detections = []
    for batch_idx in range(boxes.shape[0]):
        scores_idx = scores[batch_idx]
        labels_idx = labels[batch_idx]
        boxes_idx = boxes[batch_idx]

        # Low score filtering
        keep_idxs = scores_idx >= score_thresh
        scores_idx = scores_idx[keep_idxs]
        boxes_idx = boxes_idx[keep_idxs]
        labels_idx = labels_idx[keep_idxs]

        # NMS filtering
        if use_nms:
            keep_idxs = torchvision.ops.batched_nms(
                boxes_idx,
                scores_idx,
                labels_idx,
                iou_threshold=nms_threshold)
            scores_idx = scores_idx[keep_idxs]
            boxes_idx = boxes_idx[keep_idxs]
            labels_idx = labels_idx[keep_idxs]

        detections.append(
            {
                "boxes": boxes_idx,
                "scores": scores_idx,
                "labels": labels_idx,
            }
        )
    return detections
