"""Parameters and batch-1 GFLOPs of DETR / DETR-MoR (detr_mor package).

The RT-DETR family is profiled with scripts/profile_latency.py --flops; this is
the same measurement for the package models, with the same conventions:
2 FLOPs per multiply-add, FlopCounterMode for convs / linears, and
nn.MultiheadAttention counted analytically (its eval fast path is not seen by
the counter): projections 2*E*E*(2L + 2S) plus scores / weighted sum 4*L*S*E
per call, on the shapes it is actually called with - so MoR's routed subsets
are counted as executed.

Example::

    python paper/analysis/complexity_detr.py --out paper/analysis/out/complexity_detr.json
"""

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from detr_mor.config import load_config, split_config  # noqa: E402
from detr_mor.models import build_model  # noqa: E402

RUNS = (
    ('DETR', 'detr', 'configs/detr.yaml'),
    ('MoR-cyclic', 'mor', 'configs/mor_cyclic.yaml'),
    ('MoR-sequential', 'mor', 'configs/mor_sequential.yaml'),
)
STAGES = ('backbone', 'backbone_proj', 'encoder', 'decoder', 'heads')


def stage_of(fqn):
    parts = fqn.split('.')
    if len(parts) < 2:
        return None
    top = parts[1]
    if top in ('class_mlp', 'bbox_mlp'):
        return 'heads'
    return top if top in STAGES else None


@torch.no_grad()
def measure(model, size):
    attn = {}
    mha_names = set()
    handles = []
    root = type(model).__name__
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            mha_names.add('{}.{}'.format(root, name))

            def hook(mod, args, kwargs, name=name):
                query = kwargs.get('query', args[0] if args else None)
                key = kwargs.get('key', args[1] if len(args) > 1 else None)
                if mod.batch_first:
                    batch, length, embed = query.shape
                    source = key.shape[1]
                else:
                    length, batch, embed = query.shape
                    source = key.shape[0]
                proj = 2 * batch * embed * embed * (2 * length + 2 * source)
                core = 4 * batch * length * source * embed
                attn[name] = attn.get(name, 0) + proj + core
            handles.append(module.register_forward_pre_hook(hook,
                                                            with_kwargs=True))
    x = torch.randn(1, 3, size, size)
    try:
        with FlopCounterMode(display=False) as counter:
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    counts = {fqn: sum(ops.values())
              for fqn, ops in counter.get_flop_counts().items()}
    out = {stage: 0.0 for stage in STAGES}
    for fqn, flops in counts.items():
        stage = stage_of(fqn)
        if stage is None:
            continue
        parts = fqn.split('.')
        ancestors = ['.'.join(parts[:i]) for i in range(2, len(parts))]
        if any(a in counts for a in ancestors):
            continue  # a counted ancestor already includes it
        out[stage] += flops
    # Remove whatever the counter did see inside MHA (top-most MHA entries
    # only), then add the analytic count for every call.
    for fqn in mha_names:
        parts = fqn.split('.')
        ancestors = ['.'.join(parts[:i]) for i in range(2, len(parts))]
        if fqn in counts and not any(a in mha_names for a in ancestors):
            stage = stage_of(fqn)
            out[stage] -= counts[fqn]
    for name, flops in attn.items():
        out[stage_of('{}.{}'.format(root, name))] += flops
    result = {stage: flops / 1e9 for stage, flops in out.items()}
    result['total'] = sum(out.values()) / 1e9
    return result


def params(model):
    def count(module, trainable_only=False):
        return sum(p.numel() for p in module.parameters()
                   if p.requires_grad or not trainable_only)
    routers = sum(p.numel() for n, p in model.named_parameters()
                  if 'router' in n.lower())
    return {
        'total': count(model),
        'trainable': count(model, True),
        'backbone': count(model.backbone),
        'encoder': count(model.encoder),
        'decoder': count(model.decoder),
        'transformer': count(model.encoder) + count(model.decoder),
        'routers': routers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--out', default='paper/analysis/out/complexity_detr.json')
    args = parser.parse_args()
    torch.manual_seed(0)
    results = {}
    for name, model_type, config_path in RUNS:
        config = load_config(config_path, model_type)
        dataset_config, model_config, _ = split_config(config)
        model = build_model(model_type, model_config, dataset_config,
                            pretrained_backbone=False).eval()
        results[name] = {'params': params(model),
                         'gflops': measure(model, args.size)}
        print(name, json.dumps(results[name], indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
