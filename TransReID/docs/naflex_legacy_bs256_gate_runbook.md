# NaFlex legacy-batch M→MS gate

This track isolates the effect of global batch 256 and the legacy 60-epoch
schedule. It must not be mixed with the existing batch-64/30-epoch records.

## Fixed contract

- Market1501 → MSMT17, source-only (`COMBINEALL=false`), seed 1234.
- Two DDP ranks on explicitly selected GPUs; global batch 256, local batch 128.
- Adam, LR 5e-6, 60 epochs, linear warmup 10, milestones 30/50.
- Evaluate and checkpoint every 2 epochs; report epoch 60 as the primary metric.
- Target evaluation is image-only. Caption and attributes are training-only.
- Codebook and Relation begin at epoch 6 when their stages are selected.
- A successful run retains only `model_best.pth.tar` and `train_log.txt`.

The matrix is
`configs/experiments/gates/naflex_m_to_ms_legacy_bs256_60ep.json`.

## Inspect without training

```bash
RUNTIME_ROOT=/home/zhangwq/ReID_2026_runtime/autodl-tmp \
GPU_IDS=0,2 MODE=dry-run RUN_NAME=image_only \
bash tools/run_naflex_legacy_bs256_gate.sh
```

## Start one explicit stage

First verify both selected GPUs are idle. Then activate `ReIDEnv` and run:

```bash
source /home/zhangwq/miniconda3/bin/activate ReIDEnv
RUNTIME_ROOT=/home/zhangwq/ReID_2026_runtime/autodl-tmp \
GPU_IDS=0,2 MODE=train RUN_NAME=image_only \
bash tools/run_naflex_legacy_bs256_gate.sh
```

Valid run names are `image_only`, `caption_pid`,
`caption_codebook_delayed`, and `caption_codebook_relation`. Training refuses to
start without an explicit run name. Run the later stages only after reviewing
the preceding gate result.

## State and monitoring

State and launcher logs are written under
`$RUNTIME_ROOT/logs/naflex_legacy_bs256_gate/m_to_ms/`. The actual resolved
configuration, losses, and evaluation metrics are in each output directory's
`train_log.txt`. A `.running.json` changes to `.complete.json` only after epoch
60, a non-empty best checkpoint, and successful non-best checkpoint cleanup.
Failures retain their checkpoint state for diagnosis/resume.
