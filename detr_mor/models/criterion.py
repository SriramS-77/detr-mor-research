"""Hungarian matching and the DETR set-prediction loss.

Shared by :class:`~detr_mor.models.detr.DETR` and
:class:`~detr_mor.models.mor_detr.MoRDETR`, which in the original notebook each
carried their own byte-identical copy of this code.

Convention note: the model's bbox head predicts cxcywh in [0, 1] (via sigmoid),
while targets are stored as x1y1x2y2 in [0, 1]. Predictions are therefore
converted to xyxy before every box cost / loss, and targets never are.
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.ops
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    r"""
    One-to-one assignment between predicted queries and ground-truth objects.

    The cost of matching a prediction to a target is a weighted sum of a
    classification term (negative probability of the target class), an L1 box
    term and a negative GIoU term. ``scipy.optimize.linear_sum_assignment``
    then finds the minimum-cost perfect matching, per image.

    :param cls_cost_weight: weight on the classification cost
    :param l1_cost_weight: weight on the L1 box cost
    :param giou_cost_weight: weight on the GIoU cost
    """

    def __init__(self, cls_cost_weight, l1_cost_weight, giou_cost_weight):
        super().__init__()
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight

    @torch.no_grad()
    def forward(self, cls_output, bbox_output, targets):
        r"""
        :param cls_output: (B, num_queries, num_classes) logits
        :param bbox_output: (B, num_queries, 4) cxcywh in [0, 1]
        :param targets: list of B dicts with 'boxes' (N_i, 4) xyxy and
            'labels' (N_i,)
        :return: list of B ``(pred_idx, target_idx)`` int64 tensor pairs
        """
        batch_size, num_queries, num_classes = cls_output.shape

        # Concat all prediction boxes and class prob together
        class_prob = cls_output.reshape((-1, num_classes))
        class_prob = class_prob.softmax(dim=-1)
        # class_prob -> (B*num_queries, num_classes)

        pred_boxes = bbox_output.reshape((-1, 4))
        # pred_boxes -> (B*num_queries, 4)

        # Concat all target boxes and labels also together
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])
        # len(target_labels) -> num_targets_for_entire_batch
        # target_boxes -> (num_targets_for_entire_batch, 4)

        # Classification Cost
        cost_classification = -class_prob[:, target_labels]
        # cost_cls -> (B*num_queries, num_targets_for_entire_batch)

        # DETR predicts cx,cy,w,h , we need to convert to x1y1x2y2 for giou.
        # Don't need to convert targets as they are already in x1y1x2y2
        pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
            pred_boxes,
            'cxcywh',
            'xyxy')

        cost_localization_l1 = torch.cdist(
            pred_boxes_x1y1x2y2,
            target_boxes,
            p=1
        )
        # cost_l1 -> (B*num_queries, num_targets_for_entire_batch)

        cost_localization_giou = -torchvision.ops.generalized_box_iou(
            pred_boxes_x1y1x2y2,
            target_boxes
        )
        # cost_giou -> (B*num_queries, num_targets_for_entire_batch)

        total_cost = (self.l1_cost_weight * cost_localization_l1
                      + self.cls_cost_weight * cost_classification
                      + self.giou_cost_weight * cost_localization_giou)

        # linear_sum_assignment is a scipy call, so the cost has to come back to cpu
        total_cost = total_cost.reshape(batch_size, num_queries, -1).cpu()
        # total_cost -> (B, num_queries, num_targets_for_entire_batch)

        num_targets_per_image = [len(target["labels"]) for target in targets]
        total_cost_per_batch_image = total_cost.split(
            num_targets_per_image,
            dim=-1
        )
        # total_cost_per_batch_image[i] = (B, num_queries, num_targets_ith_image)

        match_indices = []
        for batch_idx in range(batch_size):
            batch_idx_assignments = linear_sum_assignment(
                total_cost_per_batch_image[batch_idx][batch_idx]
            )
            batch_idx_pred, batch_idx_target = batch_idx_assignments
            # len(batch_idx_pred) = num_targets_ith_image
            match_indices.append((torch.as_tensor(batch_idx_pred,
                                                  dtype=torch.int64),
                                  torch.as_tensor(batch_idx_target,
                                                  dtype=torch.int64)))
            # match_indices -> [
            #   ([pred_box_a1, ...], [target_box_i1, ...]),
            #   ([pred_box_a2, ...], [target_box_i2, ...]),
            #   ... assignment pairs for ith batch image
            #   ]
        return match_indices


class SetCriterion(nn.Module):
    r"""
    DETR set-prediction loss, computed independently for every decoder layer
    (deep supervision).

    Every query that is not matched to a target is trained towards the
    background class, down-weighted by ``bg_class_weight`` so the loss is not
    dominated by it.

    :param num_classes: including background
    :param bg_class_idx: index of the background class (0 or num_classes-1)
    :param bg_class_weight: cross-entropy weight for the background class
    :param cls_cost_weight: weight applied to the classification loss
    :param l1_cost_weight: weight applied to the L1 box loss
    :param giou_cost_weight: weight applied to the GIoU loss
    """

    def __init__(self, num_classes, bg_class_idx, bg_class_weight,
                 cls_cost_weight, l1_cost_weight, giou_cost_weight):
        super().__init__()
        self.num_classes = num_classes
        self.bg_class_idx = bg_class_idx
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight

        self.matcher = HungarianMatcher(cls_cost_weight=cls_cost_weight,
                                        l1_cost_weight=l1_cost_weight,
                                        giou_cost_weight=giou_cost_weight)

        # To ensure background class is not disproportionately attended by model.
        # Registered as a buffer so it follows the model onto the right device.
        cls_weights = torch.ones(num_classes)
        cls_weights[bg_class_idx] = bg_class_weight
        self.register_buffer('cls_weights', cls_weights)

    def forward(self, cls_output, bbox_output, targets, num_decoder_layers):
        r"""
        :param cls_output: (num_decoder_layers, B, num_queries, num_classes)
        :param bbox_output: (num_decoder_layers, B, num_queries, 4) cxcywh
        :param targets: list of B dicts with 'boxes' (N_i, 4) xyxy and 'labels'
        :param num_decoder_layers: how many leading entries to supervise
        :return: dict of lists, keyed 'classification' and 'bbox_regression',
            with one scalar tensor per decoder layer. The caller sums them.
        """
        losses = defaultdict(list)

        # Perform matching for each decoder layer
        for decoder_idx in range(num_decoder_layers):
            cls_idx_output = cls_output[decoder_idx]
            bbox_idx_output = bbox_output[decoder_idx]

            match_indices = self.matcher(cls_idx_output, bbox_idx_output, targets)

            # pred_batch_idxs are batch indexes for each assignment pair
            pred_batch_idxs = torch.cat([
                torch.ones_like(pred_idx) * i
                for i, (pred_idx, _) in enumerate(match_indices)
            ])
            # pred_batch_idxs -> (num_targets_for_entire_batch, )

            # pred_query_idx are prediction box indexes (out of num_queries)
            # for each assignment pair
            pred_query_idx = torch.cat([pred_idx
                                        for (pred_idx, _) in match_indices])
            # pred_query_idx -> (num_targets_for_entire_batch, )

            # For all assigned prediction boxes, get the target label
            valid_obj_target_cls = torch.cat([
                target["labels"][target_obj_idx]
                for target, (_, target_obj_idx) in zip(targets, match_indices)
            ])
            # valid_obj_target_cls -> (num_targets_for_entire_batch, )

            # Initialize target class for all predicted boxes to be background class
            target_classes = torch.full(
                cls_idx_output.shape[:2],
                fill_value=self.bg_class_idx,
                dtype=torch.int64,
                device=cls_idx_output.device
            )
            # target_classes -> (B, num_queries)

            # For predicted boxes that were assigned to some target,
            # update their target label accordingly
            target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls

            # Compute classification loss
            loss_cls = torch.nn.functional.cross_entropy(
                cls_idx_output.reshape(-1, self.num_classes),
                target_classes.reshape(-1),
                self.cls_weights.to(cls_idx_output.device))

            # Get pred box coordinates for all matched pred boxes
            matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
            # matched_pred_boxes -> (num_targets_for_entire_batch, 4)

            # Get target box coordinates for all matched target boxes
            target_boxes = torch.cat([
                target['boxes'][target_obj_idx]
                for target, (_, target_obj_idx) in zip(targets, match_indices)],
                dim=0
            )
            # target_boxes -> (num_targets_for_entire_batch, 4)

            # Convert matched pred boxes to x1y1x2y2 format
            matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                matched_pred_boxes,
                'cxcywh',
                'xyxy'
            )

            # Don't need to convert target boxes as they are in x1y1x2y2 format.
            # Compute L1 Localization loss
            loss_bbox = torch.nn.functional.l1_loss(
                matched_pred_boxes_x1y1x2y2,
                target_boxes,
                reduction='none')
            loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

            # Compute GIoU loss
            loss_giou = torchvision.ops.generalized_box_iou_loss(
                matched_pred_boxes_x1y1x2y2,
                target_boxes
            )
            loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

            losses['classification'].append(loss_cls * self.cls_cost_weight)
            losses['bbox_regression'].append(
                loss_bbox * self.l1_cost_weight
                + loss_giou * self.giou_cost_weight
            )
        return losses
