from typing import Any, Dict

import torch


def normalize_batch(batch: Any) -> Dict[str, Any]:
    """Convert legacy tuples to the new named batch contract.

    This compatibility function is temporary. New datasets should return
    dictionaries directly.
    """

    if isinstance(batch, dict):
        required = {"images", "pids"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Batch is missing required keys: {sorted(missing)}")
        return batch

    if not isinstance(batch, (tuple, list)) or len(batch) not in (5, 6):
        raise TypeError(
            "Expected a named batch or a legacy 5/6-item ReID tuple"
        )

    if len(batch) == 5:
        # Legacy source-train collate:
        # images, pids, camids, image_paths, dataset_ids
        images, pids, camids, image_paths, dataset_ids = batch
    else:
        # Legacy query+gallery collate:
        # images, pids, raw_camids, camid_tensor, image_paths, dataset_ids
        images, pids, raw_camids, camids, image_paths, dataset_ids = batch

    return {
        "images": images,
        "pids": pids,
        "camids": camids,
        "dataset_ids": dataset_ids,
        "image_paths": image_paths,
        "captions": None,
        "caption_mask": None,
    }
