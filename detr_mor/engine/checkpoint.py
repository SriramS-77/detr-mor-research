"""Checkpoint save / load."""

import os

import torch


def checkpoint_path(train_config):
    r"""Path a run's checkpoint lives at: ``{task_name}/{ckpt_name}``."""
    return os.path.join(train_config['task_name'], train_config['ckpt_name'])


def save_checkpoint(model, optimizer, scheduler, epoch, loss, steps, path):
    r"""
    Write a resumable checkpoint.

    :param epoch: index of the epoch that just finished
    :param loss: last observed loss (tensor or float), stored for reference
    :param steps: global step counter
    :param path: destination file
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item() if torch.is_tensor(loss) else loss,
        'steps': steps,
    }
    torch.save(checkpoint, path)
    print('Checkpoint saved at epoch {}'.format(epoch))


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    map_location=None):
    r"""
    Restore a checkpoint, accepting both formats this project has produced:
    the full training dict written by :func:`save_checkpoint`, and a bare
    ``state_dict`` (what the notebook's DETR path saved).

    Optimizer / scheduler state is only restored if both the object is passed
    and the checkpoint carries it.

    :return: (start_epoch, steps) - both 0 for a bare state_dict
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('steps', 0)

    # Bare state_dict from an older run.
    model.load_state_dict(checkpoint)
    print('Loaded a bare state_dict checkpoint; optimizer/scheduler state and '
          'epoch counter are not available, training would restart from epoch 0.')
    return 0, 0
