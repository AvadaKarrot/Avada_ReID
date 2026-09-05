# SigLIP2 feature-level transition ablations (August 2026)

This record preserves the intent and observed results of four transition-only
matrix configurations removed during the experiment-config cleanup. The original
JSON files remain recoverable from Git history, and every completed run retains
its resolved configuration in `train_log.txt`.

These experiments preceded the canonical NaFlex image-only, Caption-C, Attribute
Codebook, and Relation recipes. They should not be used as launch configs for new
runs.

## Removed configurations

| Historical configuration | Purpose | Successor |
|---|---|---|
| `siglip2_map_only_image_only_s1_30ep.json` | Penultimate MAP-pooler image-only transition test | `configs/experiments/matrices/single_source/siglip2_map_image_only_30ep.json` with final/native MAP semantics |
| `siglip2_penultimate_native_image_only_s1_30ep.json` | Penultimate native-pooler image-only test | `configs/experiments/matrices/single_source/siglip2_naflex_image_only_30ep.json` |
| `siglip2_naflex_penultimate_triplet_caption_alignment_s1_30ep.json` | Single-source penultimate Triplet + Caption-C test | `configs/experiments/matrices/single_source/siglip2_naflex_caption_pid_30ep.json` |
| `protocol2_siglip2_naflex_penultimate_triplet_caption_alignment_30ep.json` | Protocol-2 penultimate Triplet + Caption-C test | `configs/experiments/matrices/protocol2/siglip2_naflex_caption_pid_30ep.json` |

## Recorded results

Values are `mAP / Rank-1` in percent. “Best” is copied from historical target
evaluation logs and is diagnostic only. Fixed final/last-available values are the
safer reproducibility reference. “Incomplete” means the available log stopped
before epoch 30.

### MAP-only image-only transition

| Direction | Final or last available | Historical best |
|---|---:|---:|
| M→MS | 16.24 / 40.31 (epoch 30) | 16.24 (epoch 25) |
| MS→M | 41.87 / 65.35 (epoch 30) | 42.04 (epoch 10) |
| MS→C3 | 31.64 / 32.86 (epoch 30) | 31.80 (epoch 20) |
| C3→MS | 12.63 / 34.35 (epoch 5, incomplete) | 12.63 (epoch 5) |
| M→C3 | no log located | — |
| C3→M | no log located | — |

### Penultimate native image-only

| Direction | Final or last available | Historical best |
|---|---:|---:|
| M→MS | 13.62 / 36.66 (epoch 30) | 13.83 (epoch 20) |
| MS→M | 40.23 / 64.37 (epoch 30) | 40.23 (epoch 30) |
| MS→C3 | 28.04 / 28.21 (epoch 30) | 28.15 (epoch 20) |
| C3→MS | 12.34 / 33.80 (epoch 30) | 12.48 (epoch 20) |
| C3→M | 34.54 / 58.70 (epoch 30) | 34.54 (epoch 30) |
| M→C3 | 22.79 / 23.50 (epoch 25, incomplete) | 23.03 (epoch 20) |

### Penultimate Triplet + Caption-C, single source

| Direction | Epoch 30 | Historical best |
|---|---:|---:|
| M→MS | 19.66 / 44.77 | 19.67 (epoch 25) |
| MS→M | 46.03 / 70.40 | 48.02 (epoch 10) |
| MS→C3 | 31.34 / 32.93 | 32.09 (epoch 15) |
| C3→MS | 23.75 / 49.29 | 23.90 (epoch 25) |
| C3→M | 51.97 / 71.32 | 51.97 (epoch 30) |
| M→C3 | 29.29 / 29.71 | 29.45 (epoch 20) |

### Penultimate Triplet + Caption-C, Protocol-2

| Sources→target | Epoch 30 | Historical best |
|---|---:|---:|
| M+MS+CS→C3 | 36.77 / 37.93 | 38.20 (epoch 10) |
| M+CS+C3→MS | 26.13 / 53.52 | 26.16 (epoch 25) |
| MS+CS+C3→M | 59.30 / 78.89 | 59.30 (epoch 30) |

## Interpretation

The transition matrices mixed feature-level questions with protocol scheduling,
which made the root config directory difficult to understand. Their scientific
evidence is retained here, while new experiments should start from the canonical
base recipes and protocol matrices. Missing or incomplete directions must not be
silently treated as completed results.
