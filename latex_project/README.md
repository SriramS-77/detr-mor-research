# Manuscript: Mixture-of-Recursions in detection transformers

The target journal is *Neural Computing and Applications* (Springer, double-blind review). The manuscript uses the Springer Nature `sn-jnl` template (v3.1). Options are `iicol` (two-column) and `sn-basic,Numbered`, which gives numbered [n] citations as NCA requires.

| File | What it is |
|---|---|
| `main.tex` / `main.pdf` | Anonymised manuscript, 12 pages |
| `title_page.tex` / `.pdf` | Separate title page: authors, funding, CRediT, AI statement. **Fill in the red placeholders.** |
| `submission/` | Upload set. `manuscript.tex` is a single file with every value inlined. |
| `references.bib` | Generated from `references_verified.bib` by `analysis/compact_bib.py` |
| `analysis/bib_verification.md` | Where each reference was verified |
| `generated/numbers.tex` | Every number and plot table. Generated; do not edit. |

## Where the numbers come from

`main.tex` contains no typed-in results. Every value is a `\val{key}` lookup that `analysis/make_tables.py` generates from these sources:

- **Test AP, per IoU and per class:** the stored `eval_results.json` of each run.
- **Routing probes, exit depth and encoder foreground ratios:** the stored `routing_diagnostics_1000.json` files.
- **Parameters and FLOPs** (`analysis/out/`):
  - RT-DETR: `scripts/profile_latency.py --flops --num-queries 25 --sizes 640`;
  - DETR: `analysis/complexity_detr.py`.

An undefined key renders as red `??key`.

## Build

You need [Tectonic](https://tectonic-typesetting.github.io). It fetches TeX packages on first use.

```bash
TECTONIC=/path/to/tectonic bash build.sh main
```

```bash
TECTONIC=/path/to/tectonic bash build.sh title_page
```

```bash
python analysis/flatten.py
```

`flatten.py` refreshes `submission/`. Copy `main.pdf` and `title_page.pdf` into it afterwards.

For the referee single-column layout, remove `iicol` from `\documentclass`. The paper will then run longer than 12 pages.

## Before submitting

- Fill in the title-page placeholders: authors, affiliations, ORCID, funding, CRediT, acknowledgements.
- Confirm the "Competing interests" statement and the generative-AI statement. NCA requires LLM use to be documented in Methods.
- The manuscript says the code will be released on acceptance. Have a repository ready.

## Latency and FPS

`analysis/latency.py` times all seven models end to end, from input to post-processed detections, at 640×640. It also reports encoder and decoder stage times and FPS. It uses random weights: latency doesn't depend on them, and the MoR routers keep a fixed number of tokens per recursion.

- **CPU:**

  ```bash
  python latex_project/analysis/latency.py --device cpu --iters 20 --warmup 3 --out latex_project/analysis/out/latency_cpu.json
  ```

- **GPU:** run `analysis/latency_gpu.ipynb` on Kaggle or locally. It needs only the code and the configs, no checkpoints or data. It writes `latency_gpu_<gpu>.json` and `profile_rtdetr_<gpu>_*.json`; copy them to `analysis/out/`.

## Optional: bootstrap confidence intervals (needs a GPU, e.g. Kaggle)

On CPU this takes more than 10 h per checkpoint; on a T4 it takes a few minutes. From the repo root, with the VOC data in `data/`, run one dump per checkpoint.

2-layer baseline:

```bash
python latex_project/analysis/dump_rtdetr_dets.py --name 2L --notebook rt_detr.ipynb --config "rt_detr_all_results/rt_detr_voc_results (2 blocks)/rt_detr_config.yaml" --ckpt "rt_detr_all_results/rt_detr_voc_results (2 blocks)/best_rt_detr.pth" --eval-json "rt_detr_all_results/rt_detr_voc_results (2 blocks)/eval_results.json" --out-dir dets
```

6-layer baseline:

```bash
python latex_project/analysis/dump_rtdetr_dets.py --name 6L --notebook rt_detr.ipynb --config "rt_detr_all_results/rt_detr_new_results (6 blocks)/rt_detr_config.yaml" --ckpt "rt_detr_all_results/rt_detr_new_results (6 blocks)/best_rt_detr.pth" --eval-json "rt_detr_all_results/rt_detr_new_results (6 blocks)/eval_results.json" --out-dir dets
```

MoR 2×3:

```bash
python latex_project/analysis/dump_rtdetr_dets.py --name 2x3 --notebook rt_detr_mor.ipynb --config rt_detr_mor_config.yaml --ckpt "rt_detr_all_results/rt_detr_mor_voc_results (2 blocks, 3 recursions)/best_rt_detr_mor.pth" --eval-json "rt_detr_all_results/rt_detr_mor_voc_results (2 blocks, 3 recursions)/eval_results.json" --out-dir dets
```

MoR 3×2:

```bash
python latex_project/analysis/dump_rtdetr_dets.py --name 3x2 --notebook rt_detr_mor.ipynb --config "rt_detr_all_results/rt_detr_mor_voc_results (3 blocks, 2 recursions)/rt_detr_mor_config.yaml" --ckpt "rt_detr_all_results/rt_detr_mor_voc_results (3 blocks, 2 recursions)/best_rt_detr_mor.pth" --eval-json "rt_detr_all_results/rt_detr_mor_voc_results (3 blocks, 2 recursions)/eval_results.json" --out-dir dets
```

Then run the bootstrap. It asserts that it reproduces each stored AP before resampling.

```bash
python latex_project/analysis/bootstrap.py --dets-dir dets --B 1000
```

Each `dets/<name>.json` must show `"reproduces_stored": true`. If it does not, GPU kernels have shifted the scores; do not use that dump.

`bootstrap.py` writes `analysis/out/bootstrap.json`. After that, rebuild: the CI sentence in §5.1 appears on its own and the paper stays at 12 pages.

**Wording rule.** If the 95% interval for MoR − 6L excludes zero, the current wording ("outperforms") stands. If it includes zero, change it to "matches or exceeds … with 65% fewer decoder parameters". This applies to the abstract, the contributions, §5.1 and the conclusion.
