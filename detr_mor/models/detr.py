"""DETR: DEtection TRansformer."""

import torch
import torch.nn as nn

from detr_mor.models.backbone import build_backbone
from detr_mor.models.criterion import SetCriterion
from detr_mor.models.position_encoding import get_spatial_position_embedding
from detr_mor.models.postprocess import postprocess_detections
from detr_mor.models.transformer import TransformerDecoder, TransformerEncoder


class DETR(nn.Module):
    r"""
    DETR model class which instantiates all layers of DETR.
    A forward pass goes through the following layers:
        1. Backbone Call (currently frozen resnet 34)
        2. Backbone Featuremap Projection to d_model of transformer
        3. Encoder of Transformer
        4. Decoder of Transformer
        5. Class and BBox MLP

    :param config: config['model_params']
    :param num_classes: including background
    :param bg_class_idx: 0 or num_classes-1
    :param training: initial value of ``self.training``. Note that
        ``model.train()`` / ``model.eval()`` are the real switch - this only sets
        the starting state, and is kept for signature compatibility with the
        original notebook.
    :param pretrained_backbone: load ImageNet weights into the ResNet trunk
    """

    def __init__(self, config, num_classes, bg_class_idx, training=True,
                 pretrained_backbone=True):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_decoder_layers = config['decoder_layers']
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
        self.encoder = TransformerEncoder(
            num_layers=config['encoder_layers'],
            num_heads=config['encoder_attn_heads'],
            d_model=config['d_model'],
            ff_inner_dim=config['ff_inner_dim'],
            dropout_prob=config['dropout_prob'])
        self.query_embed = nn.Parameter(
            torch.randn(self.num_queries, self.d_model))
        self.decoder = TransformerDecoder(
            num_layers=config['decoder_layers'],
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

    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        r"""
        :param x: (B, C, H, W) normalised images. Defaults: C=3, H=W=640,
            d_model=256, feat_h=feat_w=20.
        :param targets: list of B target dicts, or None. When given, losses are
            computed for every decoder layer.
        :param score_thresh: inference score threshold
        :param use_nms: run NMS at inference
        :return: dict with 'loss' (when targets given) and, when not training,
            'detections', 'enc_attn' and 'dec_attn'
        """
        conv_out = self.backbone(x)  # (B, C_back, feat_h, feat_w)
        # default C_back - 512

        conv_out = self.backbone_proj(conv_out)  # (B, d_model, feat_h, feat_w)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        conv_out = (conv_out.reshape(batch_size, d_model, feat_h * feat_w).
                    transpose(1, 2))
        # conv_out -> (B, feat_h*feat_w, d_model)

        # Encoder Call
        enc_output, enc_attn_weights = self.encoder(conv_out, spatial_pos_embed)
        # enc_output -> (B, feat_h*feat_w, d_model)
        # enc_attn_weights -> (num_encoder_layers, B, feat_h*feat_w, feat_h*feat_w)

        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).
                                         repeat((batch_size, 1, 1)))
        # query_objects -> (B, num_queries, d_model)

        decoder_outputs = self.decoder(
            query_objects,
            enc_output,
            self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
            spatial_pos_embed)
        query_objects, decoder_attn_weights = decoder_outputs
        # query_objects -> (num_decoder_layers, B, num_queries, d_model)
        # decoder_attn_weights -> (num_decoder_layers, B, num_queries, feat_h*feat_w)

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
            detr_output['enc_attn'] = enc_attn_weights
            detr_output['dec_attn'] = decoder_attn_weights
        return detr_output
