"""YAML config loading and validation."""

import os

import yaml

#: Keys every config must define, per section.
REQUIRED_KEYS = {
    'dataset_params': ('train_im_sets', 'test_im_sets', 'num_classes',
                       'bg_class_idx', 'im_size'),
    'model_params': ('backbone_channels', 'd_model', 'num_queries',
                     'freeze_backbone', 'ff_inner_dim', 'dropout_prob',
                     'cls_cost_weight', 'l1_cost_weight', 'giou_cost_weight',
                     'bg_class_weight', 'nms_threshold'),
    'train_params': ('task_name', 'seed', 'acc_steps', 'num_epochs',
                     'batch_size', 'lr_steps', 'lr', 'log_steps', 'ckpt_name'),
}

#: Extra model_params keys required per model type.
MODEL_SPECIFIC_KEYS = {
    'detr': ('encoder_layers', 'encoder_attn_heads',
             'decoder_layers', 'decoder_attn_heads'),
    'mor': ('encoder_num_blocks', 'encoder_num_recursions',
            'encoder_recursion_type', 'encoder_attn_heads',
            'decoder_num_blocks', 'decoder_num_recursions',
            'decoder_recursion_type', 'decoder_attn_heads'),
}


def load_config(config_path, model_type=None):
    r"""
    Read a YAML config and check that the keys the code depends on are present.

    :param config_path: path to the yaml file
    :param model_type: 'detr' or 'mor'. When given, the model-specific keys are
        validated too, which turns a mid-training KeyError into an upfront error.
    :return: config dict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError('No config found at {}'.format(config_path))

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise KeyError('Config {} is missing section "{}"'.format(
                config_path, section))
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise KeyError('Config {} section "{}" is missing keys: {}'.format(
                config_path, section, ', '.join(missing)))

    if model_type is not None:
        if model_type not in MODEL_SPECIFIC_KEYS:
            raise ValueError('model_type must be one of {}, got {}'.format(
                sorted(MODEL_SPECIFIC_KEYS), model_type))
        missing = [key for key in MODEL_SPECIFIC_KEYS[model_type]
                   if key not in config['model_params']]
        if missing:
            raise KeyError(
                'Config {} does not look like a "{}" config, model_params is '
                'missing: {}'.format(config_path, model_type, ', '.join(missing)))

    return config


def split_config(config):
    r"""
    Convenience unpacking of the three config sections.

    :return: (dataset_params, model_params, train_params)
    """
    return (config['dataset_params'],
            config['model_params'],
            config['train_params'])
