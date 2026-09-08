"""DETR with Mixture-of-Recursions encoder and decoder stacks."""

import torch
import torch.nn as nn

from detr_mor.models.backbone import build_backbone
from detr_mor.models.criterion import SetCriterion
from detr_mor.models.mor import MoRDecoder, MoREncoder
from detr_mor.models.position_encoding import get_spatial_position_embedding
from detr_mor.models.postprocess import postprocess_detections


class MoRDETR(nn.Module):
    r"""
    DETR with the plain transformer stacks swapped for MoR recursion blocks.

    A forward pass goes through the following layers:
        1. Backbone Call (currently frozen resnet 34)
        2. Backbone Featuremap Projection to d_model of transformer
        3. MoR Encoder
        4. MoR Decoder
        5. Class and BBox MLP

    Effective decoder depth (and therefore the number of deeply-supervised
    outputs) is ``decoder_num_blocks * decoder_num_recursions``.

    :param config: config['model_params']
    :param num_classes: including background
    :param bg_class_idx: 0 or num_classes-1
    :param device: kept for compatibility with the notebook's signature. When
        None (the default) the active-index tensors are built on the input's own
        device, which is what you want in almost all cases.
    :param training: initial value of ``self.training``; ``model.train()`` /
        ``model.eval()`` are the real switch.
    :param pretrained_backbone: load ImageNet weights into the ResNet trunk
    """

    def __init__(self, config, num_classes, bg_class_idx, device=None,
                 training=True, pretrained_backbone=True):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_recursions = config['decoder_num_recursions']  # per block
        self.num_blocks = config['decoder_num_blocks']
        self.num_decoder_layers = self.num_recursions * self.num_blocks
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']
        self.nms_threshold = config['nms_threshold']
        self.bg_class_idx = bg_class_idx
        self.training = training
        valid_bg_idx = (self.bg_class_idx == 0 or
                        self.bg_class_idx == (self.num_classes - 1))
        assert valid_bg_idx, "Background can only be 0 or num_classes-1"

        self.backbone = build_backbone(config, pretrained=pretrained_backbone)

        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model,
                                       kernel_size=1)
        self.encoder = MoREncoder(
            num_recursions=config['encoder_num_recursions'],
            num_blocks=config['encoder_num_blocks'],
            num_heads=config['encoder_attn_heads'],
            d_model=config['d_model'],
            ff_inner_dim=config['ff_inner_dim'],
            dropout_prob=config['dropout_prob'])
        self.query_embed = nn.Parameter(
            torch.randn(self.num_queries, self.d_model))
        self.decoder = MoRDecoder(
            num_recursions=config['decoder_num_recursions'],
            num_blocks=config['decoder_num_blocks'],
            num_heads=config['decoder_attn_heads'],
            d_model=config['d_model'],
            ff_inner_dim=config['ff_inner_dim'],
            dropout_prob=config['dropout_prob'])
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4),
        )

        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            bg_class_idx=self.bg_class_idx,
            bg_class_weight=self.bg_cls_weight,
            cls_cost_weight=self.cls_cost_weight,
            l1_cost_weight=self.l1_cost_weight,
            giou_cost_weight=self.giou_cost_weight)

        self.device = device

    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        r"""
        :param x: (B, C, H, W) normalised images
        :param targets: list of B target dicts, or None
        :param score_thresh: inference score threshold
        :param use_nms: run NMS at inference
        :return: dict with 'loss' (when targets given) and, when not training,
            'detections'. Unlike :class:`~detr_mor.models.detr.DETR`, no
            'enc_attn'/'dec_attn' are returned - the MoR stacks do not surface
            attention weights.
        """
        # Fall back to the input's device so the model works on CPU/GPU without
        # having to be told which at construction time.
        index_device = x.device if self.device is None else self.device

        conv_out = self.backbone(x)  # (B, C_back, feat_h, feat_w)
        # default C_back - 512

        conv_out = self.backbone_proj(conv_out)  # (B, d_model, feat_h, feat_w)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        conv_out = (conv_out.reshape(batch_size, d_model, feat_h * feat_w).
                    transpose(1, 2))
        # conv_out -> (B, feat_h*feat_w, d_model)

        # Encoder Call. Every token starts out active; the routers narrow this
        # set block by block.
        encoder_active_indices = torch.arange(
            feat_h * feat_w,
            dtype=torch.int64,
            device=index_device).unsqueeze(0).expand(conv_out.size(0), -1)
        enc_output = self.encoder(conv_out, spatial_pos_embed,
                                  encoder_active_indices)
        # enc_output -> (B, feat_h*feat_w, d_model)

        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).
                                         repeat((batch_size, 1, 1)))
        # query_objects -> (B, num_queries, d_model)

        decoder_active_indices = torch.arange(
            self.num_queries,
            dtype=torch.int64,
            device=index_device).unsqueeze(0).expand(enc_output.size(0), -1)
        query_objects = self.decoder(
            query_objects,
            enc_output,
            self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
            spatial_pos_embed,
            decoder_active_indices)
        # query_objects -> (num_decoder_layers, B, num_queries, d_model)

        cls_output = self.class_mlp(query_objects)
        # cls_output -> (num_decoder_layers, B, num_queries, num_classes)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        # bbox_output -> (num_decoder_layers, B, num_queries, 4)

        detr_output = {}

        if targets is not None:
            detr_output['loss'] = self.criterion(cls_output, bbox_output, targets,
                                                 self.num_decoder_layers)

        if not self.training:
            # For inference we are only interested in last layer outputs
            detr_output['detections'] = postprocess_detections(
                cls_output[-1],
                bbox_output[-1],
                bg_class_idx=self.bg_class_idx,
                score_thresh=score_thresh,
                use_nms=use_nms,
                nms_threshold=self.nms_threshold)
        return detr_output
