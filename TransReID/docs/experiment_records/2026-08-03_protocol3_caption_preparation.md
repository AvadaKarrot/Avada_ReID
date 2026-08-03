# Protocol-3 Caption preparation record (2026-08-03)

This record freezes the Caption corpus and code state before extending
Market-1501 from train-only coverage to full-source coverage. It complements
`2026-07_market_to_msmt17.md`, which contains the completed single-source
Market-to-MSMT17 diagnostic metrics.

## Protocol boundary

- Protocol-3 trains on the complete official source datasets.
- Caption is source-only training supervision.
- The target query/gallery loaders remain image-only in every rotation.
- A dataset's query/gallery Caption is eligible only when that dataset is a
  source with `DATASETS.COMBINEALL=True`; it is never consumed when the dataset
  is the target.

The planned leave-one-domain-out rotations are:

| Target | Full source domains | Caption status before this extension |
|---|---|---|
| CUHK03 | Market-1501 + MSMT17 + CUHK-SYSU | Market full-source incomplete |
| MSMT17 | Market-1501 + CUHK-SYSU + CUHK03 | Market full-source incomplete |
| Market-1501 | MSMT17 + CUHK-SYSU + CUHK03 | Complete |

## Verified Prompt V2.4 corpus

The clean files on the experiment server passed JSON parsing, non-empty
`raw_response`, unique `image_path`, dataset name, prompt-version, and split
count checks.

| Dataset | Covered splits | Rows | Protocol-3 source status |
|---|---|---:|---|
| Market-1501 | train 12,936 | 12,936 | Missing query/gallery |
| MSMT17_V1 | train 30,248; val 2,373; query 11,659; gallery 82,161 | 126,441 | Complete |
| CUHK03 detected/new split | train 7,365; query 1,400; gallery 5,332 | 14,097 | Complete |
| CUHK-SYSU | all cropped images (train-only dataset) | 34,574 | Complete |

Current four-dataset contract: 188,048 rows.

## Market-1501 full-source definition

The server dataset was enumerated with the same exclusions used by
`Dataset.combine_all`:

| Split | Disk/official candidates | Exclusion | Caption rows required |
|---|---:|---|---:|
| train | 12,936 | none | 12,936 |
| query | 3,368 | none | 3,368 |
| gallery | 19,732 files | PID 0 background and PID -1 junk | 13,115 |
| **full source** | | | **29,419** |

The missing generation workload is therefore 16,483 images. After it is
generated, cleaned, and exported, the four-dataset contract must contain
204,531 unique rows.

## Code gate prepared before GPU generation

- `caption_tools/datasets.py` supports Market `--split-scope full` and applies
  the same PID 0/-1 exclusions as training.
- `DATASETS.COMBINEALL` is forwarded into `ImageDataManager`.
- Comma-separated source/target configuration is normalized for multi-source
  runs.
- `CaptionStore` remains train-only by default and widens its index to official
  source splits only when `combineall=True`.
- Target loaders retain their original image-only tuple contract.
- The generator resumes the existing Market raw JSONL by `image_path`, so the
  12,936 completed train records are not regenerated.

The real server filesystem enumeration produced exactly 29,419 unique paths:
train 12,936, query 3,368, gallery 13,115.

## Completion acceptance criteria

Before Protocol-3 training, all of the following must hold:

1. Market raw and clean JSONL each contain 29,419 rows with the exact split
   counts above.
2. Invalid JSON, empty response, duplicate path, and non-V2.4 records are all
   zero.
3. The merged contract contains 204,531 unique rows.
4. A real multi-source `ImageDataManager` run with `COMBINEALL=True` reports
   complete source Caption coverage.
5. Target query/gallery batches contain images and labels only, with no Caption
   field.
6. The updated contract is uploaded to Hugging Face only after local/server
   download verification; the dataset card must continue to state that
   MSMT17_V2 is not included.

## Runtime and storage expectation

The previous steady Market generation rate was approximately 3.59 images/s.
The remaining 16,483 images require about 76.5 minutes of inference, plus model
startup and final validation. The operational estimate is 1 hour 18 minutes to
1 hour 30 minutes on the RTX PRO 6000.
