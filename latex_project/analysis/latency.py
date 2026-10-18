"""End-to-end latency and FPS of all seven detectors in the paper.

Times ``model(x)`` (backbone to post-processed detections) for one 640x640
input, plus the time spent inside the encoder / decoder transformer stacks,
measured with forward hooks:

* RT-DETR family (models from the notebooks): ``decoder``.
* DETR family (detr_mor package): ``encoder`` and ``decoder``.

Weights are random. Latency does not depend on them: every MoR router keeps a
fixed number of tokens per recursion (expert-choice capacity), so the executed
shapes are those of the trained model. RepVGG blocks are fused as at
deployment. This is eager PyTorch (no TensorRT / torch.compile), so absolute
numbers are pessimistic; the comparison between models is the point.

Examples::

    # CPU
    python latex_project/analysis/latency.py --device cpu --threads 8 --iters 30 \
        --out latex_project/analysis/out/latency_cpu.json
    # GPU, fp32 and fp16, batch 1 and batch 8 throughput
    python latex_project/analysis/latency.py --device cuda --precisions fp32 fp16 \
        --batch-sizes 1 8 --out latency_gpu.json
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402
import yaml  # noqa: E402

from detr_mor.config import load_config, split_config  # noqa: E402
from detr_mor.models import build_model  # noqa: E402
from detr_mor.utils.notebook import load_notebook_namespace  # noqa: E402

R = 'rt_detr_all_results'
MODELS = {
    # key: (family, notebook or model type, config)
    'rt2': ('rt', 'rt_detr.ipynb', R + '/rt_detr_voc_results (2 blocks)/rt_detr_config.yaml'),
    'rt6': ('rt', 'rt_detr.ipynb', R + '/rt_detr_new_results (6 blocks)/rt_detr_config.yaml'),
    'rt23': ('rt', 'rt_detr_mor.ipynb', 'rt_detr_mor_config.yaml'),
    'rt32': ('rt', 'rt_detr_mor.ipynb',
             R + '/rt_detr_mor_voc_results (3 blocks, 2 recursions)/rt_detr_mor_config.yaml'),
    'detr': ('detr', 'detr', 'configs/detr.yaml'),
    'cyc': ('detr', 'mor', 'configs/mor_cyclic.yaml'),
    'seq': ('detr', 'mor', 'configs/mor_sequential.yaml'),
}
NAMES = {'rt2': 'RT-DETR (2L)', 'rt6': 'RT-DETR (6L)', 'rt23': 'RT-DETR-MoR (2x3)',
         'rt32': 'RT-DETR-MoR (3x2)', 'detr': 'DETR', 'cyc': 'DETR-MoR (cyclic)',
         'seq': 'DETR-MoR (sequential)'}


def build(key, device):
    family, source, config_path = MODELS[key]
    if family == 'rt':
        ns = load_notebook_namespace(source)
        with open(config_path, encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        model = ns['DETR'](dict(config['model_params'], pretrained_backbone=False),
                           num_classes=config['dataset_params']['num_classes'])
        model.eval()
        model.convert_to_deploy()
        stages = ('decoder',)
    else:
        config = load_config(config_path, source)
        dataset_config, model_config, _ = split_config(config)
        model = build_model(source, model_config, dataset_config,
                            pretrained_backbone=False).eval()
        stages = ('encoder', 'decoder')
    return model.to(device), stages


class StageClock:
    r"""Wall time spent inside named submodules, via pre/post forward hooks."""

    def __init__(self, model, stages, cuda):
        self.cuda, self.marks, self.handles = cuda, {}, []
        for name in stages:
            module = getattr(model, name)
            self.handles.append(module.register_forward_pre_hook(self._mark(name, 0)))
            self.handles.append(module.register_forward_hook(self._mark(name, 1)))
        self.stages = stages

    def _mark(self, name, idx):
        def hook(*_):
            if self.cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            else:
                event = time.perf_counter()
            self.marks[(name, idx)] = event
        return hook

    def read(self):
        out = {}
        for name in self.stages:
            a, b = self.marks[(name, 0)], self.marks[(name, 1)]
            out[name] = a.elapsed_time(b) if self.cuda else (b - a) * 1e3
        return out

    def remove(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def bench(model, stages, device, batch, size, precision, iters, warmup):
    cuda = device.type == 'cuda'
    x = torch.randn(batch, 3, size, size, device=device)
    clock = StageClock(model, stages, cuda)
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=precision == 'fp16' and cuda)
    totals, parts = [], {name: [] for name in stages}
    try:
        for step in range(warmup + iters):
            if cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                t0 = time.perf_counter()
            with autocast:
                model(x)
            if cuda:
                end.record()
                torch.cuda.synchronize()
                total = start.elapsed_time(end)
            else:
                total = (time.perf_counter() - t0) * 1e3
            if step >= warmup:
                totals.append(total)
                for name, value in clock.read().items():
                    parts[name].append(value)
    finally:
        clock.remove()
    totals.sort()
    median = statistics.median(totals)
    return {
        'batch': batch, 'precision': precision,
        'ms_median': median, 'ms_mean': statistics.mean(totals),
        'ms_p90': totals[min(len(totals) - 1, int(0.9 * len(totals)))],
        'ms_std': statistics.pstdev(totals),
        'fps': 1000.0 * batch / median,
        'stage_ms_median': {name: statistics.median(v) for name, v in parts.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=None, help='CPU threads')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1])
    parser.add_argument('--precisions', nargs='+', default=['fp32'], choices=['fp32', 'fp16'])
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.threads:
        torch.set_num_threads(args.threads)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        hw = torch.cuda.get_device_name(device)
    else:
        hw = platform.processor() or platform.machine()
    print('device {} ({}) | torch {} | threads {}'.format(
        device, hw, torch.__version__, torch.get_num_threads()))

    rows = []
    for key in args.models:
        torch.manual_seed(0)
        model, stages = build(key, device)
        for precision in args.precisions:
            if precision == 'fp16' and device.type != 'cuda':
                continue
            for batch in args.batch_sizes:
                r = bench(model, stages, device, batch, args.size, precision,
                          args.iters, args.warmup)
                r.update(model=key, name=NAMES[key])
                rows.append(r)
                print('{:24s} {} b={:<2d} median {:8.2f} ms  p90 {:8.2f}  '
                      'FPS {:7.2f}  {}'.format(
                          NAMES[key], precision, batch, r['ms_median'], r['ms_p90'],
                          r['fps'], '  '.join('{} {:.2f} ms'.format(k, v)
                                              for k, v in r['stage_ms_median'].items())),
                      flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'device': str(device), 'hardware': hw,
                       'torch': torch.__version__, 'threads': torch.get_num_threads(),
                       'size': args.size, 'iters': args.iters, 'warmup': args.warmup,
                       'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                       'results': rows}, handle, indent=2)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
