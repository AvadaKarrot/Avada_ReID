# Protocol-2 CLIP experiment matrix (2026-08-03)

## Scope

This matrix compares the verified unified CLIP image-only parity model with the
same model plus source-only Prompt V2.4 global Caption alignment. All target
evaluation is image-only. Every run uses seed 1234, 60 epochs, and evaluation
plus a complete checkpoint every 10 epochs.

| Sources | Target | Variants |
|---|---|---|
| Market + MSMT17 + CUHK-SYSU | CUHK03 | image-only, Caption alignment |
| Market + CUHK-SYSU + CUHK03 | MSMT17 | image-only, Caption alignment |
| MSMT17 + CUHK-SYSU + CUHK03 | Market | image-only, Caption alignment |

The machine-readable source of truth is
`configs/experiments/protocol2_clip_matrix.json`.

## Split contract

- `DATASETS.COMBINEALL=False`; Protocol-2 is not the `Full` protocol.
- Market source: official train, 12,936 images / 751 identities.
- MSMT17 source: `list_train.txt` only, 30,248 images / 1,041 identities.
  The 2,373 validation images are excluded.
- CUHK-SYSU source: all 34,574 cropped train-only images / 11,934 identities.
- CUHK03 source: detected/new-protocol split 0 train, 7,365 images / 767
  identities.
- Targets use their official query/gallery sets and never receive Caption.

Expected merged source sizes are:

| Sources | Images | Identities |
|---|---:|---:|
| M + MS + CS | 77,758 | 13,726 |
| M + CS + C3 | 54,875 | 13,452 |
| MS + CS + C3 | 72,187 | 13,742 |

## Reproducibility and execution

Market and CUHK-SYSU file enumeration and PID relabeling are sorted before
these runs, so independent image-only and Caption processes receive identical
source ordering. `RandomIdentitySampler` remains unchanged and supplies four
instances per identity for the three triplet branches and PID-aware Caption
alignment.

`tools/run_protocol2_matrix.py` executes the six runs sequentially. It records
the Git commit, refuses to overwrite a non-resumable non-empty output, resumes
from `checkpoint_latest.pth.tar` when available, and requires
`model_last.pth.tar` before marking a run complete. Each run has an independent
log and complete/failed marker under `/root/autodl-tmp/logs/protocol2_clip`.

Results will be appended only after full checkpoint and metric validation.
