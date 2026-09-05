# NaFlex two-GPU DDP, global batch 256

## Reproduction contract

- launcher: PyTorch DDP via `torch.distributed.run`
- world size: 2
- `SOLVER.IMS_PER_BATCH`: 256 (global)
- effective per-rank batch: 128
- `DATALOADER.NUM_INSTANCE`: 4
- losses: rank-local ID, Triplet, Caption-C, Attribute Codebook, and Relation
- synchronization: parameter gradients only
- evaluation, logs, and checkpoints: rank 0 only

This matches the historical two-card training semantics. It does not gather
features or captions across ranks, because doing so would change the
contrastive and batch-hard objectives rather than merely restore batch size.

## Before a run

Check that both selected GPUs are idle. On the shared campus server, do not
start if either GPU has an unknown process or substantial memory use.

The two physical GPU IDs are exposed with `CUDA_VISIBLE_DEVICES`; inside the
job they become local ranks 0 and 1. For example, physical GPUs 0 and 2:

```bash
CUDA_VISIBLE_DEVICES=0,2 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train.py \
  --config_file configs/experiments/base/siglip2_naflex_caption_codebook_relation.yml \
  MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256
```

Always override server-specific dataset, pretrained, Caption, Codebook, and
output paths when the base recipe paths do not match the current host.

For a one-iteration memory check, add `--smoke_iterations 1` before the YACS
overrides. This executes one optimizer step and deliberately performs no
evaluation and writes no checkpoint.

## Matrix runners

Single-source matrix:

```bash
python tools/run_backbone_transfer_matrix.py \
  --matrix configs/experiments/matrices/single_source/siglip2_naflex_caption_pid_30ep.json \
  --mode train \
  --nproc-per-node 2 \
  --global-batch-size 256 \
  --gpu-ids 0,2
```

Protocol-2 matrix:

```bash
python tools/run_protocol2_matrix.py \
  --matrix configs/experiments/matrices/protocol2/siglip2_naflex_caption_pid_30ep.json \
  --mode train \
  --nproc-per-node 2 \
  --global-batch-size 256 \
  --gpu-ids 0,2
```

Do not launch a complete matrix until a one-iteration memory smoke has passed
with the exact model and objectives. Per-rank batch 128 may still exceed a
24-GiB RTX 3090 even with AMP; DDP does not pool GPU memory.

## Scientific comparison

Changing global batch from 64 to 256 changes the optimizer trajectory and the
number of optimizer steps per epoch. A batch-256 rerun is a new controlled
baseline; it must not be mixed with prior batch-64 results as if only hardware
changed. Keep learning rate and schedule fixed for the first parity run, then
test any learning-rate scaling as a separate ablation.
