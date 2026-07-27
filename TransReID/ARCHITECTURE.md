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
- optional caption-alignment interface
- generic trainer and image-only evaluator
- unified train/evaluate entry points
- lightweight source-caption JSONL store

Temporarily retained:

- legacy `model`, `loss`, `processor`, and training scripts
- legacy `ImageDataManager`
- legacy CMC/mAP implementation

The old files must remain until the refactored CLIP baseline has been compared
against an existing checkpoint. They can then be removed one subsystem at a
time.

## Caption JSONL contract

Only source training records are valid:

```json
{
  "dataset": "market1501",
  "split": "train",
  "image_path": "bounding_box_train/0002_c1s1_example.jpg",
  "pid": 2,
  "captions": ["a person wearing a dark jacket"],
  "quality_score": 0.91,
  "generator": "caption-model-name"
}
```

`CaptionStore` rejects query/gallery records to make target-text leakage an
explicit error.

## Current gates

The unified entry point deliberately rejects:

- distributed training, until single-device CLIP parity is verified
- caption training, until a concrete text encoder and generated JSONL file are
  selected

These gates prevent partially migrated paths from silently producing
non-comparable experimental results.
