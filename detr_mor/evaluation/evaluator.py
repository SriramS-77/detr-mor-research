"""mAP evaluation over the VOC test set."""

import torch
from tqdm import tqdm

from detr_mor.evaluation.mean_ap import compute_map


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
        for idx in range(len(dataset.idx2label)):
            label_name = dataset.idx2label[idx]
            if label_name in all_aps:
                print('AP for class {} = {:.4f}'.format(label_name,
                                                        all_aps[label_name]))
        print('Mean Average Precision : {:.4f}'.format(mean_ap))

    return mean_ap, all_aps
