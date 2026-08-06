"""Validate real source-caption coverage and image-only target batches."""

import argparse
import gc
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.datamanager import ImageDataManager
from engine.batch import normalize_batch


EXPECTED_TRAIN = {
    "market1501": 12936,
    "msmt17": 30248,
    "cuhk03": 7365,
    "cuhksysu": 34574,
}


def _target_for(source):
    return "msmt17" if source == "market1501" else "market1501"


def validate_dataset(source, data_root, caption_file):
    common_options = dict(
        root=data_root,
        sources=source,
        targets=_target_for(source),
        transforms="random_flip",
        batch_size_train=8,
        batch_size_test=8,
        workers=0,
        train_sampler="SequentialSampler",
        use_gpu=False,
    )
    manager = ImageDataManager(
        **common_options,
        caption=True,
        caption_file=caption_file,
        caption_selection="first",
        caption_missing_policy="error",
    )

    records = manager.train_loader.dataset.train
    expected = EXPECTED_TRAIN[source]
    if len(records) != expected:
        raise AssertionError(
            f"{source}: expected {expected} train records, got {len(records)}"
        )
    covered = sum(bool(record[4]) for record in records)
    if covered != expected:
        raise AssertionError(
            f"{source}: expected complete coverage, got {covered}/{expected}"
        )

    train_batch = next(iter(manager.train_loader))
    if not isinstance(train_batch, dict):
        raise AssertionError(f"{source}: caption train batch is not named")
    if not train_batch["caption_mask"].all():
        raise AssertionError(f"{source}: train batch contains empty captions")
    if len(train_batch["dataset_ids"]) != train_batch["images"].shape[0]:
        raise AssertionError(f"{source}: dataset ids were not preserved")
    if manager._source_eval_loader is not None:
        raise AssertionError(
            f"{source}: lazy source visualization loader was built eagerly"
        )

    target_batch = normalize_batch(next(iter(manager.test_loader)))
    if target_batch["captions"] is not None:
        raise AssertionError(
            f"{source}: target evaluation batch leaked captions"
        )
    if target_batch["caption_mask"] is not None:
        raise AssertionError(
            f"{source}: target evaluation batch leaked caption masks"
        )

    image_only_manager = ImageDataManager(
        **common_options,
        caption=False,
    )
    image_only_batch = normalize_batch(
        next(iter(image_only_manager.train_loader))
    )
    if image_only_batch["captions"] is not None:
        raise AssertionError(
            f"{source}: caption=False source batch contains captions"
        )
    if image_only_batch["caption_mask"] is not None:
        raise AssertionError(
            f"{source}: caption=False source batch contains caption masks"
        )

    result = {
        "source": source,
        "target": _target_for(source),
        "train_records": len(records),
        "caption_covered": covered,
        "train_batch_named": True,
        "caption_false_source_image_only": True,
        "target_batch_image_only": True,
        "source_eval_loader_lazy": True,
    }
    del (
        image_only_batch,
        image_only_manager,
        target_batch,
        train_batch,
        records,
        manager,
    )
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(EXPECTED_TRAIN),
        default=sorted(EXPECTED_TRAIN),
    )
    args = parser.parse_args()

    results = [
        validate_dataset(name, args.data_root, args.caption_file)
        for name in args.datasets
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
