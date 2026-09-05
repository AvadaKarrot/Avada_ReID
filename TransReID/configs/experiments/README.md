# Experiment configuration layout

This directory separates reusable training recipes from experiment matrices.
The split is organizational only: the cleanup did not change solver, model, data,
or objective values in the retained configurations.

## Canonical directories

- `base/`: one reusable YAML recipe per model/objective combination.
- `matrices/single_source/`: JSON run matrices for one-source-to-one-target
  evaluation.
- `matrices/protocol2/`: JSON run matrices for the standard three-source
  Protocol-2 evaluation.
- `gates/`: small, explicitly ordered decision gates. These are not full
  protocol matrices and must be run one named stage at a time.

The source, target, caption path, and output path inside a base YAML are defaults
for direct smoke or manual runs. Matrix runners override those fields for each
direction. A resolved configuration is recorded in each experiment's
`train_log.txt`; generated per-run YAML/JSON files do not belong here.

## Active NaFlex method chain

| Stage | Canonical base recipe |
|---|---|
| Image-only | `base/siglip2_naflex_image_only.yml` |
| Caption-C (PID-positive alignment) | `base/siglip2_naflex_caption_pid.yml` |
| Attribute Codebook only | `base/siglip2_naflex_codebook_only.yml` |
| Caption-C + Attribute Codebook | `base/siglip2_naflex_caption_codebook.yml` |
| Caption-C + Codebook + Relation | `base/siglip2_naflex_caption_codebook_relation.yml` |

The Codebook-only recipe is intentionally retained as an ablation. Text and
caption supervision are training-only; target evaluation remains image-only.

## Naming rules

- Base YAML names describe architecture and objective, not one transfer
  direction.
- Matrix JSON names describe protocol and schedule.
- Use `caption_pid` for Caption-C to make the PID-positive contract explicit.
- Do not add dates, seeds, server names, or output paths to base recipe names.

## Running

Direct single recipe smoke/manual run:

```bash
python tools/train.py \
  --config_file configs/experiments/base/siglip2_naflex_caption_codebook_relation.yml
```

Use the matrix runners for complete single-source or Protocol-2 sweeps. The
current delayed Codebook pipeline entry point is:

```bash
bash tools/run_naflex_codebook_delayed_s1_p2.sh
```

For historical two-card/global-batch-256 reproduction, use the DDP options on
the matrix runners and follow
`docs/naflex_two_gpu_ddp_bs256_runbook.md`. The canonical base recipes retain
batch 64 so older experiment records are not silently redefined.

The isolated 60-epoch M→MS decision gate is documented in
`docs/naflex_legacy_bs256_gate_runbook.md`. Its training runner requires an
explicit `--only` stage and therefore never launches all four methods by
accident.

Before starting a formal run, confirm the matrix's source domains, target domain,
`DATASETS.COMBINEALL=false`, target caption disabled, seed, schedule, and output
directory.

## Compatibility files at the directory root

Root-level CLIP, multibranch, DINOv3, and older matrix files remain temporarily
because existing scripts or historical comparisons may still depend on them.
They are not the preferred location for new NaFlex work. Migrate them only after
their callers and records have been audited.

Historical transition-only NaFlex matrices removed by this cleanup are documented
in `docs/experiment_records/2026-08_siglip2_feature_level_ablations.md` and remain
recoverable from Git history.

## Evaluation note

For domain-generalization claims, fixed-epoch results are the protocol-safe
comparison. Historical logs may report a target-domain-selected "best" epoch;
that value is useful for diagnosis but must not be presented as target-free model
selection.
