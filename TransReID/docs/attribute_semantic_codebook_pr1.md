# Attribute Semantic Codebook PR1

PR1 builds source-only, bucket-specific semantic anchors from the existing
Caption v2.4 JSONL. It does not regenerate attributes and does not connect a
new loss to training yet.

## Data contract

The original JSONL is the source of truth. The loader reads these buckets:

- `upper_clothing`
- `lower_clothing`
- `footwear`
- `carried_items`
- `accessories`
- `hair`
- `distinctive_features`

For the three clothing dictionaries, `not_visible`, `unknown`, `unclear`, and
`cannot_determine` are masked while `clear` and `partial` remain valid. Empty
list buckets are masked. Camera IDs are joined from the authoritative ReID
dataset parser; they are never guessed from Caption filenames.

## Build

Run from `TransReID`:

```bash
python tools/build_attribute_codebook.py \
  --config-file configs/experiments/base/siglip2_naflex_caption_pid.yml \
  --output-dir /root/autodl-tmp/precomputed/attribute_codebooks/market1501 \
  --domain-mode auto \
  --device cuda
```

`auto` uses camera pseudo-domains for one source dataset and dataset domains
for multiple source datasets. Only `DATASETS.SOURCES` and the allowed source
splits are loaded. The target dataset is not read into the phrase bank or
codebook.

Outputs:

```text
phrase_bank.pt
codebook.pt
manifest.json
```

`manifest.json` records source datasets, split policy, domain evidence,
Caption SHA256, text-model path, construction parameters, and artifact hashes.

## Validate

```bash
python tools/validate_attribute_codebook.py \
  /root/autodl-tmp/precomputed/attribute_codebooks/market1501 \
  --forbid-dataset msmt17 \
  --verify-caption-file
```

Validation fails for target leakage, artifact hash mismatch, Caption hash
mismatch, invalid prototype norms, confidence outside `[0, 1]`, incompatible
embedding dimensions, or out-of-range phrase assignments.

## Reproducibility

The primary spherical K-Means seed defaults to `1234`. Additional stability
seeds default to `2345` and `3456`. Override them by repeating:

```bash
--stability-seed 4567 --stability-seed 5678
```

The merge is accepted only when prototype similarity passes the configured
threshold and the merged cluster's compactness degradation stays below
`--max-compactness-drop`.
