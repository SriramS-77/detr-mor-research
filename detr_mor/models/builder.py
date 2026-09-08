"""Model factory."""

from detr_mor.models.detr import DETR
from detr_mor.models.mor_detr import MoRDETR

MODEL_TYPES = ('detr', 'mor')


def build_model(model_type, model_config, dataset_config, device=None,
                pretrained_backbone=True):
    r"""
    Instantiate a model and move it to ``device``.

    :param model_type: 'detr' or 'mor'
    :param model_config: config['model_params']
    :param dataset_config: config['dataset_params']; reads 'num_classes' and
        'bg_class_idx'
    :param device: torch device to move the model to
    :param pretrained_backbone: load ImageNet weights into the ResNet trunk
    :return: nn.Module
    """
    if model_type not in MODEL_TYPES:
        raise ValueError('model_type must be one of {}, got {!r}'.format(
            MODEL_TYPES, model_type))

    kwargs = dict(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx'],
        pretrained_backbone=pretrained_backbone,
    )

    if model_type == 'detr':
        model = DETR(**kwargs)
    else:
        # device=None lets MoRDETR read the device off its input tensors.
        model = MoRDETR(device=None, **kwargs)

    if device is not None:
        model.to(device)
    return model
