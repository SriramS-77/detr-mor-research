"""Batch-1 latency and parameter breakdown of the notebook RT-DETR, per stage.

Answers Phase-0 kill criterion 1: does the encoder + neck share of latency grow
materially with input resolution? If it does not, token-adaptive compute in the
encoder has nothing to save.

The model code is loaded straight from the notebook (rt_detr.ipynb or
rt_detr_mor.ipynb), so the profile is of exactly the code that was trained.
Only import / def / class / assignment statements of the definition cells are
executed; nothing that trains, evaluates or reads data.

Stages are timed with forward hooks on the model's own submodules:

    backbone      conv1 .. layer4
    input_proj    the three 1x1 conv + BN projections
    aifi          positional embedding + the S5 transformer encoder
    ccfm          top-down / bottom-up fusion (CSPRepLayers)
    query_select  flatten, anchors, encoder heads, top-k gather
    decoder       the deformable (MoR) decoder
    heads_post    per-layer class heads + top-k decoding

Weights are random: latency does not depend on them. RepVGG blocks are fused
(``convert_to_deploy``) unless ``--no-deploy`` is passed, as they would be at
deployment. This is eager PyTorch, not TensorRT, so absolute numbers are
pessimistic; the *shares* are what the kill criterion reads.

Example (on a Kaggle T4)::

    python scripts/profile_latency.py --config rt_detr_config.yaml \
        --sizes 640 960 1280 --num-queries 25 300 --fp16
"""

import argparse
import copy
import json
import os
import statistics
import time

import _bootstrap  # noqa: F401  (sys.path setup)

import torch
import yaml

from detr_mor.utils.notebook import load_notebook_namespace

STAGES = ('backbone', 'input_proj', 'aifi', 'ccfm', 'query_select', 'decoder',
          'heads_post')


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def param_breakdown(model):
    r"""Parameters per stage, in millions."""
    groups = {
        'backbone': [model.backbone],
        'input_proj': [model.backbone_proj],
        'aifi': [model.s5_encoder],
        'ccfm': [model.up_sampling, model.down_sampling, model.rep_blocks_up,
                 model.rep_blocks_down],
        'query_select': [model.enc_output, model.enc_class_head,
                         model.enc_bbox_head],
        'decoder': [model.decoder, model.ref_to_query_embed],
        'heads_post': [model.class_heads],
    }
    out = {name: sum(count_params(m) for m in mods) / 1e6
           for name, mods in groups.items()}
    out['total'] = count_params(model) / 1e6
    return out


class StageTimer:
    r"""
    Records a timestamp at each stage boundary via forward hooks.

    On CUDA these are CUDA events (read after a synchronize); on CPU plain
    ``perf_counter`` values, since CPU kernels run synchronously.
    """

    def __init__(self, model, device):
        self.cuda = device.type == 'cuda'
        self.marks = {}
        boundaries = [
            (model, 'pre', 'start'),
            (model.backbone.layer4, 'post', 'backbone'),
            (model.backbone_proj[-1], 'post', 'input_proj'),
            (model.s5_encoder, 'post', 'aifi'),
            (model.rep_blocks_down[-1], 'post', 'ccfm'),
            (model.decoder, 'pre', 'query_select'),
            (model.decoder, 'post', 'decoder'),
            (model, 'post', 'heads_post'),
        ]
        self.order = [name for _, _, name in boundaries]
        self._handles = []
        for module, when, name in boundaries:
            hook = self._make_hook(name)
            if when == 'pre':
                self._handles.append(module.register_forward_pre_hook(hook))
            else:
                self._handles.append(module.register_forward_hook(hook))

    def _make_hook(self, name):
        def hook(*_):
            if self.cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.marks[name] = event
            else:
                self.marks[name] = time.perf_counter()
        return hook

    def read(self):
        r""":return: {stage: ms} for the forward just run, plus 'total'."""
        if self.cuda:
            torch.cuda.synchronize()

            def elapsed(a, b):
                return self.marks[a].elapsed_time(self.marks[b])
        else:
            def elapsed(a, b):
                return (self.marks[b] - self.marks[a]) * 1e3
        times = {self.order[i]: elapsed(self.order[i - 1], self.order[i])
                 for i in range(1, len(self.order))}
        times['total'] = elapsed(self.order[0], self.order[-1])
        return times

    def remove(self):
        for handle in self._handles:
            handle.remove()


@torch.no_grad()
def profile(model, device, size, iters, warmup, fp16):
    r""":return: {stage: median ms} over ``iters`` batch-1 forwards."""
    x = torch.randn(1, 3, size, size, device=device)
    timer = StageTimer(model, device)
    samples = {name: [] for name in STAGES + ('total',)}
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=fp16 and device.type == 'cuda')
    try:
        for step in range(warmup + iters):
            with autocast:
                model(x)
            times = timer.read()
            if step >= warmup:
                for name, value in times.items():
                    samples[name].append(value)
    finally:
        timer.remove()
    return {name: statistics.median(values) for name, values in samples.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-stage batch-1 latency of the notebook RT-DETR')
    parser.add_argument('--notebook', default='rt_detr.ipynb',
                        help='rt_detr.ipynb or rt_detr_mor.ipynb')
    parser.add_argument('--config', default='rt_detr_config.yaml',
                        help='yaml whose model_params build the model')
    parser.add_argument('--sizes', type=int, nargs='+', default=[640, 960, 1280],
                        help='square input sizes, multiples of 32')
    parser.add_argument('--num-queries', type=int, nargs='+', default=None,
                        help='override num_queries (default: the config value)')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--device', default=None,
                        help='default: cuda when available')
    parser.add_argument('--fp16', action='store_true',
                        help='autocast to fp16 on CUDA')
    parser.add_argument('--no-deploy', action='store_true',
                        help='skip RepVGG / BN fusion')
    parser.add_argument('--out', default=None,
                        help='write the results as JSON here')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    namespace = load_notebook_namespace(args.notebook)
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    base_model_config = dict(config['model_params'], pretrained_backbone=False)
    num_classes = config['dataset_params']['num_classes']

    print('Device: {}{}  |  notebook: {}  |  config: {}  |  fp16: {}'.format(
        device, ' ({})'.format(torch.cuda.get_device_name(device))
        if device.type == 'cuda' else '', args.notebook, args.config,
        args.fp16 and device.type == 'cuda'))

    results = []
    for num_queries in args.num_queries or [base_model_config['num_queries']]:
        model_config = copy.deepcopy(base_model_config)
        model_config['num_queries'] = num_queries
        model = namespace['DETR'](model_config, num_classes=num_classes)
        model.eval()
        if not args.no_deploy:
            model.convert_to_deploy()
        model.to(device)
        params = param_breakdown(model)

        for size in args.sizes:
            times = profile(model, device, size, args.iters, args.warmup,
                            args.fp16)
            tokens = {'S3': (size // 8) ** 2, 'S4': (size // 16) ** 2,
                      'S5': (size // 32) ** 2}
            results.append({'num_queries': num_queries, 'size': size,
                            'tokens': tokens, 'ms': times, 'params_m': params})

            total = times['total']
            enc_neck = times['aifi'] + times['ccfm']
            print('\nq={} size={}  tokens S3/S4/S5={}/{}/{}  total {:.2f} ms'
                  .format(num_queries, size, tokens['S3'], tokens['S4'],
                          tokens['S5'], total))
            for name in STAGES:
                print('  {:13s} {:8.2f} ms  {:5.1f}%   params {:6.2f}M'.format(
                    name, times[name], 100 * times[name] / total,
                    params[name]))
            print('  {:13s} {:8.2f} ms  {:5.1f}%'.format(
                'aifi+ccfm', enc_neck, 100 * enc_neck / total))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\nKill criterion 1 - encoder+neck (aifi+ccfm) share of latency:')
    for num_queries in sorted({r['num_queries'] for r in results}):
        shares = ['{}px {:.1f}%'.format(
            r['size'], 100 * (r['ms']['aifi'] + r['ms']['ccfm']) / r['ms']['total'])
            for r in results if r['num_queries'] == num_queries]
        print('  q={:<4d} {}'.format(num_queries, '  ->  '.join(shares)))

    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'device': str(device), 'notebook': args.notebook,
                       'config': args.config, 'fp16': args.fp16,
                       'deploy': not args.no_deploy, 'iters': args.iters,
                       'results': results}, handle, indent=2)
        print('Wrote {}'.format(args.out))


if __name__ == '__main__':
    main()
