"""CNN backbone shared by DETR and MoR-DETR."""

import torch.nn as nn
import torchvision.models
import torchvision.ops
from torchvision.models import resnet34


def build_backbone(config, pretrained=True):
    r"""
    ResNet-34 trunk with the avgpool and fc heads removed, so it returns a
    (B, 512, H/32, W/32) feature map.

    BatchNorm is replaced by FrozenBatchNorm2d: detection batches are small, so
    running BN statistics would be noisy.

    :param config: config['model_params']; reads 'freeze_backbone'
    :param pretrained: load ImageNet weights. Pass False for offline machines or
        fast smoke tests - training from scratch here is not recommended.
    :return: nn.Sequential backbone
    """
    weights = (torchvision.models.ResNet34_Weights.IMAGENET1K_V1
               if pretrained else None)
    backbone = nn.Sequential(*list(resnet34(
        weights=weights,
        norm_layer=torchvision.ops.FrozenBatchNorm2d
    ).children())[:-2])

    if config['freeze_backbone']:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone
