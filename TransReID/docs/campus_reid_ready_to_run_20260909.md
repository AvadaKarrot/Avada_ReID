# Campus ReID runbook — 2026-09-09

## Evidence and completed work
Runtime: /home/zhangwq/ReID_2026_runtime/autodl-tmp
Worktree: /home/zhangwq/ReID_2026_runtime/worktrees/ddp-global-loss-gather
Environment: /home/zhangwq/miniconda3/envs/ReIDEnv
Branch: codex/naflex-reid (consolidated 2026-09-10; physical worktree path unchanged)

The Campus log inventory contains the following historical experiment families.
Log presence alone is not proof of successful completion; inspect final epoch and checkpoint before reporting a result.

| Family | Scope / evidence | Next action |
|---|---|---|
| NaFlex image-only, Caption alignment | Six single-source directions; three Protocol-2 directions | Retain historical BS64/30ep cohort; no blanket rerun |
| Caption + Codebook | M->MS fixed/delayed/decay experiments, seeds42/1234/2026; other five directions and three Protocol-2 logs | Audit comparable settings before combining metrics |
| Caption + Codebook + Relation | Campus M->MS log reaches epoch30; best mAP about18.95 at epoch15; final18.6880/43.4600 | Separate single-card cohort; DDP Relation needs queue/domain synchronization audit |
| Legacy BS256 image-only | Epoch60 complete; best mAP about17.37 at epoch10; final14.6770/35.5262 | Historical reference; does not isolate batch alone or match new gather |
| DDP Caption-C, gather | Epoch60 complete; best19.7675/44.6779 at epoch14; final16.6147/39.3173 | Direct control for next Codebook run |
| DDP Caption + Codebook | Five-iteration smoke passed, global256/local128; KL0.3759->0.3609 | Formal run pending GPU availability |

Historical log roots under experiments:
- backbone_transfer_s1/siglip2_naflex
- backbone_transfer_s1/siglip2_naflex_caption_alignment
- attribute_codebook_pr2
- attribute_codebook_delayed_s1
- protocol2_siglip2_naflex_image_only_30ep
- protocol2_siglip2_naflex_caption_alignment_30ep
- protocol2_siglip2_naflex_caption_attribute_codebook_delayed_30ep
- attribute_relation_pr3/m_to_ms/seed_1234
- legacy_bs256_gate/m_to_ms/image_only/seed_1234
- ddp_global_loss_gather/m_to_ms/caption_pid/lr_5e6/seed_1234

## Ready-to-run experiment
Entry: tools/run_naflex_ddp_codebook_gather_lr.sh
Base YAML: configs/experiments/base/siglip2_naflex_caption_codebook.yml
M->MS, seed1234, two GPUs, global256/local128, four instances/PID.
60 epochs, LR5e-6, warmup10, LR milestones30/50, eval every2 epochs.
Caption0.1; Codebook0.05 from epoch6; Relation disabled.
Caption and Triplet gather enabled; target captions excluded; test image-only.
Offline assets: precomputed/attribute_codebooks/market1501_pr1 (Market only; seven buckets,139 codes,768 dimensions).
This matches the completed DDP Caption control except for Codebook supervision.

On Campus:
```bash
cd /home/zhangwq/ReID_2026_runtime/worktrees/ddp-global-loss-gather/TransReID
DRY_RUN=1 GPU_IDS=0,2 bash tools/run_naflex_ddp_codebook_gather_lr.sh 5e-6
# Run only after both selected GPUs become available:
GPU_IDS=0,2 nohup bash tools/run_naflex_ddp_codebook_gather_lr.sh 5e-6 > /home/zhangwq/ReID_2026_runtime/autodl-tmp/logs/codebook_formal_launcher.log 2>&1 < /dev/null &
```

The launcher checks both selected GPUs for processes and memory allocation before starting; query failure aborts.
It does not reserve GPUs against unrelated users: coordinate allocation before launch.
A per-experiment lock blocks duplicate copies of this launcher.
Existing failed state requires inspection; successful runs are not repeated.
Resume arguments are passed to train.py; inspect checkpoint compatibility before any resume.

State/log:
logs/ddp_global_loss_gather/m_to_ms/caption_codebook_delayed/lr_5e6/seed_1234
Output:
experiments/ddp_global_loss_gather/m_to_ms/caption_codebook_delayed/lr_5e6/seed_1234

```bash
tail -f /home/zhangwq/ReID_2026_runtime/autodl-tmp/logs/ddp_global_loss_gather/m_to_ms/caption_codebook_delayed/lr_5e6/seed_1234/train.stdout.log
nvidia-smi -i 0,2 --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv -l 2
```

Check epoch6 Codebook weight0.05 and finite KL; local anchors128/global candidates256.
Success requires training exit0, epoch60, nonempty best checkpoint and completion state.
Successful launcher retains best checkpoint and logs and records deletion of intermediate weights. Failed runs retain evidence.
No shutdown or automatic GPU waiting is configured.

## Resource diagnosis and priorities
Five-iteration Codebook smoke measured about8.78GiB allocated /9.32GiB reserved per GPU.
This is not a full-run peak and does not measure compute utilization.
A representative Caption-C log interval (epoch55 iterations20->40) took7.782s: about0.389s/step,
or roughly658 global training samples/s over that interval. This is an estimate, not a synchronized benchmark.
MSMT validation around epochs56/58/60 took approximately4min per evaluation.
Validation and checkpoint pauses can dominate a run with only roughly50 steps/epoch and evaluation every2 epochs.
There is no historical continuous GPU-utilization trace here; do not conclude the GPUs were compute-idle from memory alone.

After the matched formal run, profile training separately from validation:
- Record data wait, forward/backward, optimizer, communication and evaluation time; CUDA timing requires synchronization/events.
- Check evaluation code for rank0-only inference and idle waiting ranks before designing distributed evaluation.
- Compare one GPU batch256 against two GPUs128 each only with measured memory headroom and equivalent objectives.
- Tune loaders and evaluation throughput before changing scientific batch/token settings.
- Keep global batch256 and NaFlex128 patches for matched comparisons. More memory allocation is not a performance goal.
- Increasing training batch or token limit changes the experiment; label these as new ablations.
- LR1e-5 and2e-5 are optional source-validation-selected ablations, each requiring matched Caption and Codebook controls.
- Do not choose LR, stopping epoch or architecture from target-test mAP; report fixed final metrics alongside descriptive best metrics.

## Subsequent experiment gates
1. Complete the prepared M->MS Caption+Codebook run; compare to existing identical DDP Caption recipe.
2. If improvement warrants validation, repeat paired controls at seeds42 and2026.
3. Extend the agreed DDP recipe to the remaining five directions, rebuilding/selecting source-only assets per direction.
4. Extend to three Protocol-2 directions using assets constructed only from their three training sources.
5. Add DDP Relation only after checking queue, domain coverage, gradients and rank synchronization.
The historical single-card results remain a separate cohort rather than being relabeled as DDP results.
