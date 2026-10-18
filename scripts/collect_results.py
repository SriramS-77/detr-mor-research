"""Collect every run's text artifacts into results/ (tracked) plus a summary.

The run folders (``*_results/``) are git-ignored because they hold multi-GB
checkpoints. This copies the small, irreplaceable files out of them - eval
results, routing diagnostics, training logs, configs - into ``results/runs``
and writes ``results/summary.json`` with the headline metrics of each variant.

Example::

    python scripts/collect_results.py
"""

import datetime
import glob
import json
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(REPO, 'results')
KEEP = ('.json', '.txt', '.yaml', '.yml', '.py', '.ipynb')

# run id -> (source folder, description)
RUNS = {
    'detr_8x8_150ep': ('detr_voc_results', 'DETR-R34, 8 enc / 8 dec layers, 150 epochs (early run; LR schedule differs)'),
    'detr_8x8_300ep': ('detr_voc_results/detr_voc_results_300', 'DETR-R34, 8 enc / 8 dec layers, 300 epochs'),
    'detr_mor_cyclic_150ep': ('mor_cyclic_voc_results', 'DETR-MoR cyclic, 150 epochs (early run; LR schedule differs)'),
    'detr_mor_cyclic_300ep': ('mor_cyclic_voc_results/mor_cyclic_voc_results_300', 'DETR-MoR cyclic (1 + 2x3 + 1), 300 epochs'),
    'detr_mor_sequential_300ep': ('mor_sequential_voc_results_300', 'DETR-MoR sequential (1 + 2x3 + 1), 300 epochs'),
    'rtdetr_2L': ('rt_detr_all_results/rt_detr_voc_results (2 blocks)', 'RT-DETR-R34, 2 decoder layers, 72 epochs'),
    'rtdetr_6L': ('rt_detr_all_results/rt_detr_new_results (6 blocks)', 'RT-DETR-R34, 6 decoder layers, 72 epochs'),
    'rtdetr_mor_2x3': ('rt_detr_all_results/rt_detr_mor_voc_results (2 blocks, 3 recursions)', 'RT-DETR-MoR, 2 shared blocks x 3 recursions, 72 epochs'),
    'rtdetr_mor_3x2': ('rt_detr_all_results/rt_detr_mor_voc_results (3 blocks, 2 recursions)', 'RT-DETR-MoR, 3 shared blocks x 2 recursions, 72 epochs'),
    'rtdetr_early_6L_frozen': ('rt_detr_all_results/rt_detr_old_results (Shashaank)', 'Early RT-DETR run: frozen backbone, 6 decoder blocks, FFN 2048 (mAP@0.5, score thr 0.25; different protocol)'),
}


def main():
    runs_dir = os.path.join(OUT, 'runs')
    summary = {'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'dataset': 'PASCAL VOC07+12 trainval (80/20 train/val), VOC07 test (4,945 images)',
               'metric': 'all-point VOC AP averaged over IoU 0.50:0.05:0.95 unless noted (values are fractions)',
               'checkpoints': 'not stored in git (2.7 GB); see results/README.md',
               'runs': {}}
    for run_id, (src, desc) in RUNS.items():
        src_abs = os.path.join(REPO, src)
        dst = os.path.join(runs_dir, run_id)
        os.makedirs(dst, exist_ok=True)
        copied = []
        for path in sorted(glob.glob(os.path.join(glob.escape(src_abs), '*'))):
            if os.path.isfile(path) and path.lower().endswith(KEEP):
                shutil.copy2(path, dst)
                copied.append(os.path.basename(path))
        entry = {'description': desc, 'source_folder': src, 'files': copied}
        eval_path = os.path.join(src_abs, 'eval_results.json')
        if os.path.exists(eval_path):
            ev = json.load(open(eval_path, encoding='utf-8'))
            entry['test'] = {k: ev[k] for k in ('ap', 'ap50', 'ap75', 'mean_ap', 'iou_threshold',
                                                'score_threshold', 'checkpoint', 'timestamp') if k in ev}
            if 'per_iou' in ev:
                entry['test']['per_iou'] = {t: v['mean_ap'] for t, v in ev['per_iou'].items()}
            if 'class_ap' in ev:
                entry['test']['class_ap'] = {c: v for c, v in ev['class_ap'].items() if v is not None}
        diag_path = os.path.join(src_abs, 'routing_diagnostics_1000.json')
        if os.path.exists(diag_path):
            dg = json.load(open(diag_path, encoding='utf-8'))
            entry['routing_probes_first1000'] = {k: v['ap'] for k, v in dg['maps'].items()}
            entry['decoder_capacity'] = dg.get('decoder_capacity')
        summary['runs'][run_id] = entry
        print('{:28s} {:2d} files  AP {}'.format(run_id, len(copied), entry.get('test', {}).get('ap')))

    # complexity / latency measurements from the manuscript analysis, if present
    extra = {}
    for path in sorted(glob.glob(os.path.join(REPO, '*', 'analysis', 'out', '*.json'))):
        extra[os.path.relpath(path, REPO).replace(os.sep, '/')] = json.load(open(path, encoding='utf-8'))
    if extra:
        os.makedirs(os.path.join(OUT, 'measurements'), exist_ok=True)
        for rel, data in extra.items():
            with open(os.path.join(OUT, 'measurements', os.path.basename(rel)), 'w', encoding='utf-8') as h:
                json.dump(data, h, indent=2)
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print('wrote', os.path.join(OUT, 'summary.json'))


if __name__ == '__main__':
    main()
