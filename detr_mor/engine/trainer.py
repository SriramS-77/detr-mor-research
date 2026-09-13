"""Training and validation loops."""

import os
from collections import Counter

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from detr_mor.engine.checkpoint import (
    best_checkpoint_path,
    checkpoint_path,
    load_checkpoint,
    read_min_val_loss,
    save_checkpoint,
)
from detr_mor.utils.misc import batch_images_to_device, targets_to_device


class NaNLossError(RuntimeError):
    """Raised when the loss goes non-finite, so the caller can bail out."""


def build_optimizer(model, train_config):
    r"""AdamW over the trainable parameters (the backbone is usually frozen)."""
    return torch.optim.AdamW(lr=train_config['lr'],
                             params=filter(lambda p: p.requires_grad,
                                           model.parameters()),
                             weight_decay=1E-4)


def build_scheduler(optimizer, train_config):
    r"""Step-decay the LR by 10x at each milestone epoch."""
    return MultiStepLR(optimizer,
                       milestones=train_config['lr_steps'],
                       gamma=0.1)


def reconcile_schedule(optimizer, scheduler, train_config):
    r"""
    Re-apply the config's LR policy after a checkpoint restore.

    ``load_state_dict`` restores ``lr``, ``initial_lr``, ``milestones``,
    ``gamma`` and ``base_lrs`` alongside the state that genuinely has to survive
    a resume (Adam's moment estimates, ``last_epoch``, ``_step_count``). That
    makes the config silently inert: editing ``lr`` or ``lr_steps`` has no
    effect on a resumed run. Here the config owns *policy* (where the run is
    going) and the checkpoint keeps *position* (how far it has got).

    The current LR is recomputed in closed form, ``lr * gamma ** passed``, so a
    milestone already behind ``last_epoch`` is applied rather than skipped -
    MultiStepLR itself only ever fires on the exact epoch it lands on.

    :param train_config: config['train_params']; reads 'lr' and 'lr_steps'
    :return: list of descriptions of what the config changed, empty if nothing
    """
    lr = train_config['lr']
    milestones = sorted(train_config['lr_steps'])
    changes = []

    restored = sorted(scheduler.milestones.elements())
    if restored != milestones:
        changes.append('lr_steps {} -> {}'.format(restored, milestones))
    scheduler.milestones = Counter(milestones)

    if scheduler.base_lrs and scheduler.base_lrs[0] != lr:
        changes.append('lr {:g} -> {:g}'.format(scheduler.base_lrs[0], lr))
    scheduler.base_lrs = [lr] * len(optimizer.param_groups)

    passed = sum(1 for m in milestones if m <= scheduler.last_epoch)
    current = lr * scheduler.gamma ** passed
    for group in optimizer.param_groups:
        group['initial_lr'] = lr
        group['lr'] = current
    scheduler._last_lr = [current] * len(optimizer.param_groups)
    return changes


def train_one_epoch(model, loader, optimizer, scheduler, device, train_config,
                    steps=0, label='DETR'):
    r"""
    Run one training epoch.

    Gradients are accumulated over ``acc_steps`` batches before each optimizer
    step, and the leftover gradients are flushed with a final step at the end of
    the epoch.

    :param steps: global step counter carried across epochs
    :param label: prefix used in the log lines ('DETR' / 'MOR')
    :return: (mean classification loss, mean localization loss, steps, last loss)
    :raises NaNLossError: if the loss goes non-finite
    """
    model.train()
    acc_steps = train_config['acc_steps']

    classification_losses = []
    localization_losses = []
    last_loss = 0.0

    for idx, (ims, targets, _) in enumerate(tqdm(loader, desc='Training')):
        targets = targets_to_device(targets, device)
        images = batch_images_to_device(ims, device)

        batch_losses = model(images, targets)['loss']

        loss = (sum(batch_losses['classification']) +
                sum(batch_losses['bbox_regression']))

        classification_losses.append(
            sum(batch_losses['classification']).item())
        localization_losses.append(
            sum(batch_losses['bbox_regression']).item())

        loss = loss / acc_steps
        last_loss = loss.detach()

        if not torch.isfinite(loss):
            raise NaNLossError(
                'Loss is becoming nan at step {}. Exiting'.format(steps))

        loss.backward()

        if (idx + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if steps % train_config['log_steps'] == 0:
            loss_output = ''
            loss_output += '{} Classification Loss : {:.4f}'.format(
                label, np.mean(classification_losses))
            loss_output += ' | {} Localization Loss : {:.4f}'.format(
                label, np.mean(localization_losses))
            print(loss_output, scheduler.get_last_lr())

        steps += 1

    # Flush any gradients left over from a partial accumulation window.
    optimizer.step()
    optimizer.zero_grad()

    return (float(np.mean(classification_losses)),
            float(np.mean(localization_losses)),
            steps,
            last_loss)


@torch.no_grad()
def validate(model, loader, device):
    r"""
    Run the validation pass.

    ``model.eval()`` disables dropout and, as a side effect of how the model
    classes are written, also makes the forward pass emit detections alongside
    the losses. Only the losses are used here.

    :return: (mean classification loss, mean localization loss)
    """
    model.eval()
    classification_losses = []
    localization_losses = []

    for ims, targets, _ in tqdm(loader, desc='Validating'):
        targets = targets_to_device(targets, device)
        images = batch_images_to_device(ims, device)

        batch_losses = model(images, targets)['loss']

        classification_losses.append(sum(batch_losses['classification']).item())
        localization_losses.append(sum(batch_losses['bbox_regression']).item())

    return (float(np.mean(classification_losses)),
            float(np.mean(localization_losses)))


def train(model, train_loader, val_loader, device, train_config,
          label='DETR', resume=True):
    r"""
    Full training run: resume if a checkpoint exists, then loop epochs,
    validating and checkpointing after each one.

    Results are appended to ``{task_name}/train_results.txt`` so a run that is
    killed and restarted keeps its history.

    :param label: prefix used in log lines and the results file
    :param resume: load ``{task_name}/{ckpt_name}`` if it exists
    """
    task_dir = train_config['task_name']
    os.makedirs(task_dir, exist_ok=True)
    result_path = os.path.join(task_dir, 'train_results.txt')
    ckpt_path = checkpoint_path(train_config)

    # Built once, before any resume, so that the restored optimizer/scheduler
    # state actually stays in the objects the loop uses.
    optimizer = build_optimizer(model, train_config)
    lr_scheduler = build_scheduler(optimizer, train_config)

    # Lowest val loss seen so far; read back on resume so a restarted run
    # cannot overwrite a better epoch's checkpoint with a worse one.
    best_ckpt_path = best_checkpoint_path(train_config)
    min_val_loss = read_min_val_loss(best_ckpt_path) if resume else float('inf')

    start_epoch = 0
    steps = 0
    if resume and os.path.exists(ckpt_path):
        print('Loading checkpoint as one exists at {}'.format(ckpt_path))
        start_epoch, steps = load_checkpoint(ckpt_path, model, optimizer,
                                             lr_scheduler, map_location=device)
        # The checkpoint carries the old schedule; the config is authoritative.
        changes = reconcile_schedule(optimizer, lr_scheduler, train_config)
        if changes:
            print('Config overrides the restored schedule: {}'.format(
                '; '.join(changes)))
        print('Resuming at epoch {} with lr {:g}'.format(
            start_epoch + 1, lr_scheduler.get_last_lr()[0]))

    num_epochs = train_config['num_epochs']
    last_loss = 0.0

    for epoch in range(start_epoch, num_epochs):
        try:
            train_cls_loss, train_loc_loss, steps, last_loss = train_one_epoch(
                model, train_loader, optimizer, lr_scheduler, device,
                train_config, steps=steps, label=label)
        except NaNLossError as exc:
            with open(result_path, 'a') as log:
                log.write('{}\n'.format(exc))
            raise

        print('Running Validation for Epoch {}...'.format(epoch + 1))
        val_cls_loss, val_loc_loss = validate(model, val_loader, device)

        lr_scheduler.step()

        print('Finished epoch {}'.format(epoch + 1))
        print('{} Classification Loss : {:.4f} | {} Localization Loss : '
              '{:.4f}'.format(label, train_cls_loss, label, train_loc_loss))
        print('Val Class Loss: {:.4f} | Val Loc Loss: {:.4f}'.format(
            val_cls_loss, val_loc_loss))

        with open(result_path, 'a') as log:
            log.write(
                'Epoch {}: Train_Class_Loss={:.6f}, Train_Loc_Loss={:.6f}, '
                'Val_Class_Loss={:.6f}, Val_Loc_Loss={:.6f}\n'.format(
                    epoch + 1, train_cls_loss, train_loc_loss,
                    val_cls_loss, val_loc_loss))

        # epoch + 1 so a resumed run starts on the next epoch, not this one again.
        save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, last_loss,
                        steps, ckpt_path)

        # Same format, written to a separate file, only when this epoch is the
        # best so far. The rolling checkpoint above still drives resume.
        val_loss = val_cls_loss + val_loc_loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1,
                            last_loss, steps, best_ckpt_path,
                            extra={'min_val_loss': min_val_loss,
                                   'train_cls_loss': train_cls_loss,
                                   'train_loc_loss': train_loc_loss,
                                   'val_cls_loss': val_cls_loss,
                                   'val_loc_loss': val_loc_loss})
            print('New best val loss {:.4f}, saved to {}'.format(
                min_val_loss, best_ckpt_path))

    with open(result_path, 'a') as log:
        log.write('Done Training...\n')
    print('Done Training...')
