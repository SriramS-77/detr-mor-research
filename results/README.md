# Experiment results (all variants)

The run folders (`*_results/`) are git-ignored because they hold checkpoints. This folder keeps their small, irreplaceable outputs under version control.

- `runs/<variant>/`: copied verbatim from each run folder: `eval_results.json`, `routing_diagnostics_1000.json`, training logs and configs.
- `summary.json`: one entry per variant. It holds:
  - test AP, AP50 and AP75;
  - AP at each IoU threshold;
  - per-class AP;
  - routing-probe AP on the first 1,000 test images;
  - the source folder.
- `measurements/`: parameter counts and GFLOPs (`complexity_*.json`, at 640 px with 25 queries) and CPU latency (`latency_cpu.json`).

To refresh after new runs, run this and commit `results/`:

```bash
python scripts/collect_results.py
```

## Headline test results (VOC07 test, AP@[.50:.95], all-point)

| Variant | AP | AP50 | AP75 |
|---|---|---|---|
| DETR 8/8, 300 ep | 47.18 | 74.27 | 49.20 |
| DETR-MoR cyclic, 300 ep | 45.69 | 73.03 | 47.80 |
| DETR-MoR sequential, 300 ep | 43.12 | 69.98 | 44.36 |
| DETR 8/8, 150 ep* | 39.87 | 67.13 | 40.62 |
| DETR-MoR cyclic, 150 ep* | 36.70 | 63.59 | 36.63 |
| RT-DETR 2L | 64.74 | 85.76 | 70.74 |
| RT-DETR 6L | 65.31 | 86.58 | 70.58 |
| RT-DETR-MoR 2×3 | 66.03 | 86.82 | 71.84 |
| RT-DETR-MoR 3×2 | 66.10 | 86.35 | 72.06 |
| Early RT-DETR (frozen backbone, 6 blocks)† | — | 83.79 | — |

\* Early runs with a different LR schedule; not used in the paper.
† Different protocol: mAP@0.5 at score threshold 0.25.

## Checkpoints

The `.pth` files total 2.7 GB, with single files up to 455 MB, which is above GitHub's 100 MB file limit. They are not in git. Keep them backed up separately, for example as GitHub Release assets (up to 2 GB per file) or in a Kaggle Dataset.
