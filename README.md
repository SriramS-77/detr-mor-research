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
  find_batch_size.py  largest batch that fits, measured on synthetic data
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

`evaluate.py` writes `{task_name}/eval_results.json` (mAP, per-class AP, and the
settings used); `--out other.json` moves it, `--out none` skips it.

Before a long run on an unfamiliar GPU, size the batch by measurement rather than guess:

```bash
python scripts/find_batch_size.py --config configs/mor_voc.yaml --model mor
```

It runs a real forward/backward/step at the config's own `im_size` and model dims,
doubling the batch until peak allocation passes `--target-util` (default 0.75 of VRAM),
then refining upward in smaller steps.

Training writes `{task_name}/{ckpt_name}` after every epoch and appends a line per epoch
to `{task_name}/train_results.txt`. Re-running `train.py` resumes from that checkpoint
automatically, restoring model, optimizer, scheduler, epoch and step count.

## How MoR differs

A plain transformer stack has `N` distinct layers. A MoR stack reuses one set of weights
several times, so effective depth exceeds parameter count. A per-token `MoRExpertRouter`
scores every still-active token with a sigmoid and keeps the top-k; only those tokens make
the next pass, the rest keep their previous value, and the selected tokens' gated updates
are scattered back into the full sequence.

### Middle-Cycle / Middle-Sequence

**Both stacks** follow the MoR paper's **Middle-\*** sharing layout. The first and last of
`num_blocks` blocks keep unique weights and run densely over every token; the
`num_blocks - 2` blocks in between are shared and re-applied `num_recursions` times.

```
parameter cost  = num_blocks layers
effective depth = 2 + (num_blocks - 2) * num_recursions
```

Keeping the first and last blocks unshared is what makes the variant work: the first gives
every token a full representation before any routing decision is made, and the last
re-mixes the sequence after routed tokens have been scattered back in, so tokens that
exited early are not left stale. In the decoder the first dense block also lifts
`query_objects` off its all-zeros initialisation before anything is gathered from it.

`encoder_recursion_type` / `decoder_recursion_type` pick how the middle group is traversed:

| value | visit order (M middle blocks, R recursions) | routing decisions |
|---|---|---|
| `cyclic` (Middle-Cycle, the paper's best) | `0,1,..,M-1` repeated R times | `R - 1`, one before each extra pass |
| `sequential` (Middle-Sequence) | `0` R times, then `1` R times, … | `M*R - 1`, one per block application |

Under `cyclic` a routing decision covers a **whole pass** through the middle group, which
is the granularity MoR is defined at — a token either recurses again in full or stops.
Raise `num_recursions` (not `num_blocks`) to route more often.

`*_num_blocks` must be **≥ 3**, otherwise the middle group is empty; both stacks raise a
`ValueError` rather than silently degrading to a plain 2-layer transformer.

The gated recombination is `h ← h + g · f(h)`: the router score multiplies only what the
block *added*, so `g → 0` leaves a token untouched and activations do not grow down the
stack.

In the decoder, only the **query side** is routed — cross-attention always sees the full
encoder output.

### Deep supervision

The decoder returns `2 + stages` states, all supervised by the Hungarian loss: one after
each unique block, plus one per middle stage. It is per *stage* rather than per block
application because a full-sequence query state only exists at stage boundaries —
mid-cycle, only the selected queries have been advanced. So the number of supervised
outputs depends on the schedule:

| `decoder_recursion_type` | supervised outputs |
|---|---|
| `cyclic` | `2 + decoder_num_recursions` |
| `sequential` | `2 + (decoder_num_blocks - 2) * decoder_num_recursions` |

`MoRDETR.num_decoder_layers` is read back from the decoder rather than recomputed.

## Fidelity to the notebook

Plain `DETR` is still a verbatim port: a regression check confirms it reproduces the
notebook's losses and detections to within 1e-6 in both train and eval mode.

`MoRDETR`'s **encoder and decoder no longer match the notebook** — both were rewritten to
the paper's Middle-Cycle layout (see above). The notebook versions had three problems the
rework removes:

- Routing happened per *layer* rather than per recursion, so a token could be updated by
  some blocks of a cycle but not others, and the active set shrank `M*R-1` times instead
  of `R-1`.
- The scatter added `g · h` on top of a sequence that already held `h`, giving
  `(1+g)·h + g·delta`. Activations grew ~`(1+g)` per stage — measured 4.4 → 1003 over 12
  stages.
- The decoder scattered into `query_objects`, which block 0 never wrote to, so it was
  still all zeros. Non-selected queries became exactly 0 and `LayerNorm(0)` is a constant:
  13 of 25 final-layer queries collapsed to one identical detection. Now 25 queries give
  25 distinct detections, same as plain DETR.

### Known issues

1. **`MoRExpertRouter` reads un-normalised hidden states.** Its sigmoid can saturate and
   the router stop learning if activations grow. Observed with a randomly-initialised
   backbone (feature std ≈ 100), where some encoder routers get exactly zero gradient;
   with the real ImageNet backbone (std ≈ 1) every router trains. There is no auxiliary
   router loss or z-loss keeping logits in range, unlike the paper.
2. **Routing is not differentiable.** Top-k selection carries no gradient; only the
   scalar gate `g` does. Same as the paper's expert-choice router.
3. **No attention weights from the MoR stacks.** The notebook commented out the weight
   stacking, so `MoRDETR`'s output dict has no `enc_attn` / `dec_attn` keys — unlike
   `DETR`, which still returns both. Anything consuming attention maps must special-case
   this.
4. **`if_middle_cycle` is accepted and ignored.** Superseded by `*_recursion_type`; kept
   only so existing configs keep loading.
5. **`MoRRecursionBlock`** is the notebook's reference implementation and is not used by
   either stack.

Not a MoR issue, but worth knowing: on the **first** optimizer step four decoder tensors
get exactly zero gradient (`decoder.attns.0.in_proj_weight`, `.out_proj.weight`,
`decoder.attn_norms.0.weight`, `decoder.cross_attn_norms.0.weight`), because
`query_objects` starts as exact zeros and `LayerNorm(0) == 0` at init. It clears after one
step. Plain `DETR` behaves identically — verified side by side.

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
