# detr-mor

DETR and DETR with **Mixture-of-Recursions (MoR)** blocks, for object detection on
Pascal VOC. This is the `train_mor.ipynb` research notebook restructured into a library
plus entry-point scripts, with the model maths left untouched.

## Layout

```
configs/            voc.yaml (plain DETR) and mor_voc.yaml (MoR)
detr_mor/
  config.py         YAML loading + key validation
  data/             VOCDataset, ragged-target collate, DataLoader builders
  models/           backbone, position encoding, transformer, MoR blocks,
                    matcher + loss, postprocess, DETR, MoRDETR, build_model
  engine/           train/validate loops, checkpoint save/load
  evaluation/       VOC mAP, evaluator, sample visualisation
  utils/            seeding, device resolution, target/batch movement
scripts/
  train.py          training entry point
  evaluate.py       mAP on the test imageset
  infer.py          draw sample detections (needs opencv-python)
  smoke_test.py     CPU 1-batch check on synthetic data - no dataset needed
```

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux use `.venv/bin/python`. Pick the torch index URL that matches the machine's
CUDA version (`cpu` works for the smoke test). No `pip install -e .` is needed — the
scripts put the repo root on `sys.path` themselves.

## Check it works before touching a GPU

```bash
python scripts/smoke_test.py --no-pretrained
```

Runs one batch of **synthetic** data through both models on CPU: no VOC dataset, no
checkpoint, no network access. For each of `detr` and `mor` it checks forward + loss
(one entry per decoder layer, all finite and differentiable), backward, an optimizer
step, a full `train_one_epoch`, eval-mode inference with and without NMS, the mAP code
path, and a checkpoint save/load round trip. It prints `PASS`/`FAIL` per check and exits
non-zero on any failure. Expect `ALL CHECKS PASSED` in a few seconds.

Drop `--no-pretrained` if the machine can reach the torchvision weight CDN and you want
the real ImageNet backbone in the loop.

## Data

Each entry in `train_im_sets` / `test_im_sets` must be a VOC-style root containing
`ImageSets/Main/{trainval,test}.txt`, `Annotations/` and `JPEGImages/`. Edit the paths in
`configs/*.yaml` — they currently point at a relative `data/VOCdevkit/VOC20xx`, and the
original absolute paths from the source machine are recorded in a comment at the top of
each file.

Images with more than 25 objects are skipped, because `num_queries` is 25. On VOC this
drops about 15 images.

## Train / evaluate / visualise

```bash
python scripts/train.py    --config configs/mor_voc.yaml --model mor
python scripts/evaluate.py --config configs/mor_voc.yaml --model mor
python scripts/infer.py    --config configs/mor_voc.yaml --model mor --num-samples 5
```

Swap in `--config configs/voc.yaml --model detr` for the baseline. Useful flags:
`--device cuda:1`, `--no-resume` (ignore an existing checkpoint), `--no-pretrained`,
`--ckpt path/to/other.pth`.

Training writes `{task_name}/{ckpt_name}` after every epoch and appends a line per epoch
to `{task_name}/train_results.txt`. Re-running `train.py` resumes from that checkpoint
automatically, restoring model, optimizer, scheduler, epoch and step count.

## How MoR differs

A plain transformer stack has `N` distinct layers. A MoR stack has `num_blocks` recursion
blocks, each applying **its own weights `num_recursions` times** — so with the shipped
config, 2 blocks × 2 recursions gives an effective depth of 4 at the parameter cost of 2
layers. Between blocks, a per-token `MoRExpertRouter` scores every still-active token with
a sigmoid and keeps the top-k (`k` shrinking as `(num_blocks - i) / num_blocks`). Only
those tokens go through the next block; the rest keep their previous value, and the
selected tokens' updates are scattered back into the full sequence.

The decoder emits one output per recursion (`num_blocks * num_recursions` in total), and
every one of them is supervised by the Hungarian loss — the same deep supervision plain
DETR applies per decoder layer.

## Known quirks preserved from the notebook

The MoR forward passes are a deliberate line-by-line port. The following look
unintentional but are **kept as-is** so training runs stay comparable with the original
results. Each is flagged with a `NOTE(nb-fidelity)` comment at the relevant line in
`detr_mor/models/mor.py`. A regression check confirmed the ported `DETR` and `MoRDETR`
reproduce the notebook's losses and detections to within 1e-6 in both train and eval mode.

1. **`MoREncoder` discards block 0's output.** Block 0 runs densely and updates a local
   `out`, but block 0 performs no `scatter_add_` (there are no router weights yet), and
   the `scatter_add_` at the end of block 1 writes into the *original* input `x`. Block
   0's dense work is read once by block 1's gather and then thrown away.
2. **`MoREncoder` returns the norm of `x`, not `out`.** With `num_blocks == 1` no routing
   ever happens, so the encoder returns its unmodified input. Keep `num_blocks >= 2`.
3. **No attention weights from the MoR stacks.** The notebook commented out the weight
   stacking, so `MoRDETR`'s output dict has no `enc_attn` / `dec_attn` keys — unlike
   `DETR`, which still returns both. Anything consuming attention maps must special-case
   this.
4. **`MoRDecoder` does not apply the router weights.** The encoder multiplies the block
   output by the router scores before scattering; the decoder scatters the raw
   `selected_patches`. It also scatters *inside* the recursion loop into a repeatedly
   re-cloned `query_objects`, so updates accumulate across recursions within a block.
5. **`if_middle_cycle` is accepted and ignored** by both MoR stacks.

Changing any of these changes the numbers. Re-run the baselines if you do.

## Notes on the port

Beyond restructuring, the only behavioural changes are in plumbing, not the model:

- The Hungarian matcher and set-prediction loss lived as two byte-identical ~90-line
  copies inside `DETR.forward` and `MoR_DETR.forward`. They are now
  `HungarianMatcher` / `SetCriterion` in `detr_mor/models/criterion.py`, used by both.
  Likewise the inference block is now `postprocess_detections`.
- The optimizer is built **once**, before the checkpoint is loaded. The notebook rebuilt
  it afterwards, which silently discarded the restored optimizer state on every resume.
- `{task_name}` is created before the checkpoint-existence check, not after.
- Checkpoints save `epoch + 1`, so a resumed run continues at the next epoch rather than
  repeating the one it just finished.
- `MoRDETR`'s `device` argument now defaults to `None`, in which case index tensors are
  built on the input's own device. That is what lets the model run on CPU without being
  told. Passing an explicit device still works.
- `load_checkpoint` accepts both the full training dict and a bare `state_dict`, so
  checkpoints from either notebook path still load.
- The NaN-loss guard raises `NaNLossError` (logged, then propagated) instead of calling
  `exit(0)`, which in a notebook killed the kernel and reported success to any caller.
