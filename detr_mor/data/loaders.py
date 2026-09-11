"""DataLoader construction for the VOC dataset."""

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from detr_mor.data.voc import VOCDataset


def collate_function(data):
    r"""
    Detection targets are ragged (a different number of objects per image), so we
    cannot stack them. Transpose the batch into tuples instead:
    ``(images, targets, filenames)``.
    """
    return tuple(zip(*data))


def build_train_val_loaders(dataset_config, train_config, generator=None):
    r"""
    Build the train/val DataLoaders by randomly splitting the train imageset.

    :param dataset_config: config['dataset_params']
    :param train_config: config['train_params']
    :param generator: optional torch.Generator, for a reproducible split
    :return: (train_loader, val_loader, dataset)
    """
    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])

    val_fraction = train_config.get('val_split', 0.2)
    val_size = int(val_fraction * len(voc))
    train_size = len(voc) - val_size

    split_kwargs = {} if generator is None else {'generator': generator}
    train_subset, val_subset = random_split(voc, [train_size, val_size],
                                            **split_kwargs)

    train_loader = DataLoader(train_subset,
                              batch_size=train_config['batch_size'],
                              shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(val_subset,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=collate_function)
    return train_loader, val_loader, voc


def build_test_loader(dataset_config):
    r"""
    Build the evaluation DataLoader over the test imageset.

    Uses batch_size=1 with the default collate, so targets come back with a
    leading batch dimension (``target['boxes'][0]``) - which is what
    :func:`detr_mor.evaluation.evaluator.evaluate_map` expects.

    :return: (test_loader, dataset)
    """
    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])
    test_loader = DataLoader(voc, batch_size=1, shuffle=False)
    return test_loader, voc
