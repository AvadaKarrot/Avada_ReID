# ReID Framework Architecture

## Invariants

1. Target-domain inference accepts images only.
2. CLIP, DINOv3, and SigLIP2 implement the same visual-backbone contract.
3. ReID heads and losses are shared across backbones for fair comparison.
4. Captions are optional source-training supervision and never enter
   `ReIDModel.forward`.
5. Trainers and evaluators do not branch on a backbone name.

## Runtime flow

```text
tools/train.py
  -> data.build_datamanager
  -> modeling.build_model
       -> BackboneAdapter
       -> ReIDHead
  -> objectives.build_objective
  -> engine.Trainer
       -> ReIDModel(images)
       -> ReIDObjective(outputs, batch)
```

Target-domain evaluation uses:

```text
tools/evaluate.py
  -> ReIDModel(images)
  -> normalized embeddings
  -> CMC / mAP
```

## Stable contracts

### Backbone

Every visual backbone returns `BackboneOutput`:

- `global_feature`: required `[batch, channels]`
- `patch_features`: optional `[batch, tokens, channels]`
- `spatial_shape`: optional `(height, width)`

### Model

`ReIDModel.forward(images)` returns `ReIDOutput`:

- `embedding`: BNNeck feature used by the evaluator
- `raw_feature`: pre-BN feature used by metric learning
- `logits`: training-only identity logits
- `patch_features`: optional local features

### Batch

New data loaders should return a dictionary containing:

- `images`
- `pids`
- `camids`
- `dataset_ids`
- `image_paths`
- optional `captions`
- optional `caption_mask`

`engine.normalize_batch` temporarily supports legacy five- and six-item tuples.

## Package responsibilities

- `data`: dataset records, transforms, samplers, caption lookup, loaders
- `modeling`: visual backbones, shared ReID head, image-only model
- `objectives`: ID, triplet, and optional source-only text supervision
- `engine`: generic optimization and evaluation loops
- `evaluation`: metrics output and result serialization
- `optim`: optimizer construction
- `tools`: unified command-line entry points

## Migration status

Implemented:

- common model/output contracts
- CLIP, DINOv3, and SigLIP2 adapters
- common ReID head
- ID/triplet objective
- PID-aware optional caption-alignment objective
- frozen legacy CLIP text-tower builder for source Caption supervision
- generic trainer and image-only evaluator
- complete latest/best/last checkpoints and exact training resume
- unified train/evaluate entry points
- lightweight source-caption JSONL store

Temporarily retained:

- legacy `model`, `loss`, `processor`, and training scripts
- legacy `ImageDataManager`
- legacy CMC/mAP implementation

The old files must remain until the refactored CLIP baseline has been compared
against an existing checkpoint. They can then be removed one subsystem at a
time.

`train_caption.py` is now a compatibility alias for `tools/train.py`.
`train_clipreid_base.py` remains a separate legacy parity entry point.

The Caption text encoder is named `clip_legacy`: it loads the same CLIP
checkpoint family through the old MaPLe-capable loader, but does not itself
claim to implement MaPLe. Full MaPLe injects learnable shallow/deep prompts
into both visual and textual Transformer blocks and must be represented as a
separate model feature. It must not reintroduce per-image target captions.

## Caption JSONL contract

Caption files may preserve all official splits for auditing, but only source
training records are indexed by the training-side `CaptionStore`:

```json
{
  "dataset": "market1501",
  "split": "train",
  "image_path": "bounding_box_train/0002_c1s1_example.jpg",
  "pid": 2,
  "captions": ["a person wearing a dark jacket"],
  "quality_score": 0.91,
  "generator": "caption-model-name",
  "prompt_version": "v2",
  "attributes": {
    "upper_clothing": {
      "visibility": "clear",
      "attributes": ["dark jacket"]
    }
  }
}
```

`CaptionStore` filters query/gallery records before binding and scopes keys by
dataset name. It resolves an absolute dataset path against the relative
`image_path` contract, then attaches the complete caption tuple only to source
training records. At sampling time, `random`, `first`, or `concat` selects the
text view; missing source captions either fail fast (`error`, the default) or
produce a masked sample (`mask`). Target DataLoaders always retain the original
image-only tuple contract.

`prompt_version` is provenance metadata; legacy records without
it are interpreted as V1. `attributes` is required in V2 generation artifacts
and retained for auditing, while training continues to consume `captions`.
V2.1 retains the V2 schema but enforces complete noun-headed item phrases and
removes base-garment restatement from distinctive features.
V2.2 adds confidence-aware footwear naming and uses one canonical full
description in `captions`; overlapping text views can be regenerated from the
retained `attributes` for later ablations. Training consumes only postprocessed
and exported clean JSONL, never raw generator responses. Clean records also
carry `postprocess_version` and `renderer_version` provenance.
V2.3 keeps `graphic` as a safe category for unidentifiable pictorial designs,
prioritizes more specific visible feature classes, and rejects graphics without
color, size, or shape qualifiers.
V2.4 makes semantic feature labels confidence-aware: `logo`, `text`, and
`patch` require corresponding visual evidence, while a visible but ambiguous
compact feature is represented conservatively as `mark`.

## Current gates

The unified entry point deliberately rejects:

- distributed training, until single-device CLIP parity is verified
- legacy `MODEL.CAPTION` fusion, because it requires target text
- source-camera/view embeddings, because they are not portable to held-out
  target domains

These gates prevent partially migrated paths from silently producing
non-comparable experimental results.
