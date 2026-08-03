# Protocol-2 CLIP 30-epoch matrix (2026-08-03)

## Decision

After the original 60-epoch diagnostic matrix finishes, the same six
Protocol-2 experiments will be trained from scratch with a shorter, fixed
schedule. The fixed epoch-30 target result is the primary comparison; the
best target checkpoint is retained as supplementary diagnostic information.

| Sources | Target | Variants |
|---|---|---|
| Market + MSMT17 + CUHK-SYSU | CUHK03 | image-only, Caption alignment |
| Market + CUHK-SYSU + CUHK03 | MSMT17 | image-only, Caption alignment |
| MSMT17 + CUHK-SYSU + CUHK03 | Market | image-only, Caption alignment |

The machine-readable source of truth is
`configs/experiments/protocol2_clip_matrix_30ep.json`.

## Schedule contract

- Train for 30 epochs from a fresh initialization.
- Keep `BASE_LR=5e-6` and all optimizer, batch, loss, augmentation, seed and
  model settings from the verified base configurations.
- Compress warmup from 10 to 5 epochs.
- Compress learning-rate milestones from `[30, 50]` to `[15, 25]`.
- Evaluate and save a complete resumable checkpoint every 5 epochs:
  5, 10, 15, 20, 25 and 30.
- Save `model_last` at epoch 30 and keep `model_best` selected by target mAP.
- Do not initialize these runs from any checkpoint produced by the 60-epoch
  schedule because the learning-rate histories differ.

The six formal outputs are isolated under
`/root/autodl-tmp/experiments/protocol2_30ep`. Runtime state and logs must use
`/root/autodl-tmp/logs/protocol2_clip_30ep`; they must not share the original
60-epoch matrix log directory.

## Execution order

The currently running 60-epoch matrix remains an immutable schedule
diagnostic and must finish first. After its six runs are validated, pull the
commit containing this matrix, validate real data and Caption coverage, run
the isolated six-run smoke matrix, and only then launch the sequential full
matrix. Normal resumability is provided by `checkpoint_latest.pth.tar`.
