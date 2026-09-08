"""Qualitative inference: draw ground-truth and predicted boxes onto samples."""

import os
import random

import torch
from tqdm import tqdm


def _draw_boxes(cv2, image, boxes, texts, color):
    r"""
    Draw labelled boxes onto ``image`` in place, returning the blended result.

    Boxes are drawn twice - once on the image and once on a copy where the label
    sits on a filled white plate - and the two are blended, so labels stay
    readable without hiding the image.

    :param boxes: iterable of (x1, y1, x2, y2) in absolute pixels
    :param texts: one label string per box
    """
    overlay = image.copy()
    for (x1, y1, x2, y2), text in zip(boxes, texts):
        cv2.rectangle(image, (x1, y1), (x2, y2), thickness=2, color=color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), thickness=2, color=color)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(overlay, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h),
                      [255, 255, 255], -1)
        for target_im in (image, overlay):
            cv2.putText(target_im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image


@torch.no_grad()
def infer(model, dataset, device, train_config, num_samples=5,
          output_dir='samples', model_label='detr', seed=None):
    r"""
    Save side-by-side ground-truth and prediction images for random samples.

    Writes ``{output_dir}/output_{model_label}_gt_{i}.png`` and
    ``{output_dir}/output_{model_label}_{i}.jpg``.

    Requires ``opencv-python``, which is an optional dependency - it is imported
    lazily so the rest of the package works without it.

    :param dataset: VOCDataset in 'test' split
    :param train_config: config['train_params']; reads 'infer_score_threshold'
        and 'use_nms_infer'
    :param seed: seed for the sample choice, for reproducible figures
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            'infer() needs opencv-python: pip install opencv-python') from exc

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    rng = random.Random(seed)

    for i in tqdm(range(num_samples), desc='Detecting'):
        dataset_idx = rng.randrange(len(dataset))
        im_tensor, target, fname = dataset[dataset_idx]

        detr_output = model(
            im_tensor.unsqueeze(0).to(device),
            score_thresh=train_config['infer_score_threshold'],
            use_nms=train_config['use_nms_infer']
        )
        detections = detr_output['detections']

        gt_im = cv2.imread(fname)
        if gt_im is None:
            raise FileNotFoundError('cv2 could not read {}'.format(fname))
        h, w = gt_im.shape[:2]

        # Saving images with ground truth boxes. Boxes are normalised to [0, 1],
        # so scale them back up to the original image size.
        gt_pixel_boxes = []
        gt_texts = []
        for idx, box in enumerate(target['boxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            gt_pixel_boxes.append(
                (int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
            gt_texts.append(
                dataset.idx2label[target['labels'][idx].detach().cpu().item()])
        gt_im = _draw_boxes(cv2, gt_im, gt_pixel_boxes, gt_texts, [0, 255, 0])
        cv2.imwrite(os.path.join(
            output_dir, 'output_{}_gt_{}.png'.format(model_label, i)), gt_im)

        # Saving images with predicted boxes
        boxes = detections[0]['boxes']
        labels = detections[0]['labels']
        scores = detections[0]['scores']

        im = cv2.imread(fname)
        pred_pixel_boxes = []
        pred_texts = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            pred_pixel_boxes.append(
                (int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
            pred_texts.append('{} : {:.2f}'.format(
                dataset.idx2label[labels[idx].detach().cpu().item()],
                scores[idx].detach().cpu().item()))
        im = _draw_boxes(cv2, im, pred_pixel_boxes, pred_texts, [0, 0, 255])
        cv2.imwrite(os.path.join(
            output_dir, 'output_{}_{}.jpg'.format(model_label, i)), im)

    print('Done Detecting...')
