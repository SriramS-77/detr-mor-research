# DETR-MoR

DETR and DETR with **Mixture-of-Recursions (MoR)** blocks, for object detection on
Pascal VOC.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Pick the torch index URL that matches the machine's CUDA version.

<!-- ## Check it works before touching a GPU

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
the real ImageNet backbone in the loop. -->

## Data Loading

```bash
python scripts/load_dataset.py
```

Downloads the 2007 trainval, 2007 test and 2012 trainval datasets into `data/VOCdevkit/` folder.

Images with more than 25 objects are skipped, because `num_queries` is 25. On VOC this
drops about 15 images.

# Cyclic Recursion Method

Under `cyclic` method, a routing decision covers a **whole pass** through the middle group (all the middle encoder/decoder blocks), which is the granularity MoR is defined at — a token either recurses again in full or stops. Raise `num_recursions` (not `num_blocks`) to route more often.

## Train

```bash
python scripts/train.py    --config configs/mor_cyclic.yaml --model mor --device cuda
```

Flags:
`--device cuda:1`, `--no-resume` (ignore an existing checkpoint), `--no-pretrained`,
`--ckpt path/to/other.pth`. Change device to the correct cuda index.

Training writes `{task_name}/{ckpt_name}` after every epoch and appends a line per epoch
to `{task_name}/train_results.txt`. Re-running `train.py` resumes from that checkpoint
automatically, restoring model, optimizer, scheduler, epoch and step count.

## Evaluate

```bash
python scripts/evaluate.py    --config configs/mor_cyclic.yaml --model mor --device cuda
```

`evaluate.py` writes `{task_name}/eval_results.json` (mAP, per-class AP, and the
settings used); `--out other.json` moves it, `--out none` skips it.


# Sequential Recursion Method

Under `sequential` method, a routing decision covers a single recursion through one middle block. So, granularity is defined at `block x recursion pass`. The tokens pass through all `num_recursions` of a block before going to the next block.

## Train

```bash
python scripts/train.py    --config configs/mor_sequential.yaml --model mor --device cuda
```

Flags:
`--device cuda:1`, `--no-resume` (ignore an existing checkpoint), `--no-pretrained`,
`--ckpt path/to/other.pth`. Change device to the correct cuda index.

Training writes `{task_name}/{ckpt_name}` after every epoch and appends a line per epoch
to `{task_name}/train_results.txt`. Re-running `train.py` resumes from that checkpoint
automatically, restoring model, optimizer, scheduler, epoch and step count.

## Evaluate

```bash
python scripts/evaluate.py    --config configs/mor_sequential.yaml --model mor --device cuda
```

`evaluate.py` writes `{task_name}/eval_results.json` (mAP, per-class AP, and the
settings used); `--out other.json` moves it, `--out none` skips it.
