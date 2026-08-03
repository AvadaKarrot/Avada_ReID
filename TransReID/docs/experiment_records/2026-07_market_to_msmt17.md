# Market-1501 -> MSMT17 experiment record (2026-07-29/30)

This file records completed experiments before Protocol-3 preparation. Metrics
were transcribed from the immutable server logs under
`/root/autodl-tmp/logs/`; checkpoints remain under
`/root/autodl-tmp/experiments/`.

## Shared protocol

- Source: Market-1501 official train split, 12,936 images / 751 identities.
- Target evaluation: MSMT17 query 11,659 + gallery 82,161, image input only.
- Seed: 1234.
- Schedule: 60 epochs; evaluate and save every 10 epochs.
- Input: 256 x 128; batch size 64; 4 instances per identity.
- Metric: Market-style CMC and mAP through `R1_mAP_eval`, feature norm on.
- These are single-source Market -> MSMT17 diagnostic experiments, not P2/P3.

## Legacy visual CLIP-ReID baseline

- Entry: `train_clipreid_base.py`.
- Config: `configs/experiments/clip_market_to_msmt17.yml`.
- Run commit: `6178d2c` (`refactor: harden visual clip baseline workflow`).
- Log: `/root/autodl-tmp/logs/clipreid_market_to_msmt17_full.log`.
- Output: `/root/autodl-tmp/experiments/clipreid/market_to_msmt17/seed_1234`.
- Model: legacy CLIP visual encoder, 768-D and 512-D branches.
- Training loss: two ID losses + three triplet losses, label smoothing on.
- Test feature: raw 768-D + projected 512-D concatenation (1,280-D).

| Epoch | mAP (%) | Rank-1 (%) |
|---:|---:|---:|
| 10 | 18.7 | 44.1 |
| 20 | **20.2** | **45.3** |
| 30 | 18.5 | 42.5 |
| 40 | 18.5 | 42.1 |
| 50 | 18.5 | 41.9 |
| 60 | 18.5 | 41.7 |

Best checkpoint is epoch 20. Epoch 60 is the fixed-schedule final result.

## Unified framework diagnostics

The following three runs use the new unified model path. They are mutually
comparable, but are not yet implementation-parity reproductions of the legacy
baseline: the adapter keeps one 768-D CLIP feature, the head has one BNNeck and
classifier, the loss has one ID + one triplet term, and evaluation uses the
768-D BNNeck embedding. Optimizer and augmentation settings also differ from the
legacy configuration. Therefore the performance gap below cannot be attributed
to Caption supervision alone.

### Image-only

- Entry: `tools/train.py`.
- Config: `configs/experiments/clip_market_to_msmt_image_only.yml`.
- Run commit: `6293045`.
- Log: `/root/autodl-tmp/logs/clip_market_to_msmt_image_only_seed1234.log`.
- Output: `/root/autodl-tmp/experiments/clip_market_to_msmt_image_only_seed1234`.

| Epoch | mAP (%) | Rank-1 (%) |
|---:|---:|---:|
| 10 | 3.274 | 12.977 |
| 20 | 7.186 | 23.484 |
| 30 | **8.534** | 27.052 |
| 40 | 8.377 | 27.258 |
| 50 | 8.333 | 27.266 |
| 60 | 8.457 | **27.781** |

### Caption, PID-aware multi-positive alignment

- Config: `configs/experiments/clip_market_to_msmt_caption_v2_4.yml`.
- Run commit: `38b3e50`.
- Caption file: Prompt V2.4 clean/exported contract, Market train coverage
  12,936/12,936.
- Log: `/root/autodl-tmp/logs/clip_market_to_msmt_caption_v2_4_seed1234.log`.
- Output: `/root/autodl-tmp/experiments/clip_market_to_msmt_caption_v2_4_seed1234`.

| Epoch | mAP (%) | Rank-1 (%) |
|---:|---:|---:|
| 10 | 3.280 | 12.994 |
| 20 | 7.215 | 23.570 |
| 30 | 8.445 | 26.958 |
| 40 | 8.391 | 27.249 |
| 50 | 8.514 | 27.575 |
| 60 | **8.695** | **28.287** |

### Caption, instance-only alignment

- Config: `configs/experiments/clip_market_to_msmt_caption_v2_4_instance.yml`.
- Run commit: `e00be4a`.
- Log:
  `/root/autodl-tmp/logs/clip_market_to_msmt_caption_v2_4_instance_seed1234.log`.
- Output:
  `/root/autodl-tmp/experiments/clip_market_to_msmt_caption_v2_4_instance_seed1234`.

| Epoch | mAP (%) | Rank-1 (%) |
|---:|---:|---:|
| 10 | 3.282 | 13.029 |
| 20 | 7.211 | 23.587 |
| 30 | 8.463 | 27.018 |
| 40 | 8.436 | 27.524 |
| 50 | 8.535 | 27.704 |
| 60 | **8.747** | **28.236** |

## Interpretation and next gate

Within the unified path, both Caption variants are slightly above image-only at
epoch 60, and PID-aware versus instance-only is nearly tied. This does not yet
establish Caption benefit because the stronger legacy CLIP-ReID baseline has not
been reproduced by the unified path. The next model-development gate remains:

1. reproduce legacy visual outputs, dual heads, loss composition, optimizer,
   schedule, augmentation, and 1,280-D test feature inside the unified API;
2. verify Market -> MSMT17 returns near the legacy 20.2% best mAP;
3. add Caption as the only changed variable.

## Caption corpus state before Protocol-3 extension

Prompt V2.4 clean files passed JSON/empty/duplicate checks:

| Dataset | Split coverage | Rows |
|---|---|---:|
| Market-1501 | train only | 12,936 |
| MSMT17_V1 | train 30,248; val 2,373; query 11,659; gallery 82,161 | 126,441 |
| CUHK03 detected/new split | train 7,365; query 1,400; gallery 5,332 | 14,097 |
| CUHK-SYSU | all `cropped_images` (train-only dataset) | 34,574 |

Protocol-3 full-source training additionally requires Market query 3,368 and
gallery 13,115 usable images. Market gallery PID 0 background images (2,798)
and PID -1 junk images (3,819) are excluded to match `Dataset.combine_all`.
After extension, Market contains 29,419 usable full-source images and the four
dataset Caption contract contains 204,531 rows.
