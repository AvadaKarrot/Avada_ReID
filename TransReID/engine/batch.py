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

    images, pids, camids, viewids, image_paths = batch[:5]
    captions = batch[5] if len(batch) == 6 else None
    if captions is None:
        caption_mask = None
    else:
        caption_mask = torch.tensor(
            [bool(caption) for caption in captions], dtype=torch.bool
        )

    return {
        "images": images,
        "pids": pids,
        "camids": camids,
        "viewids": viewids,
        "image_paths": image_paths,
        "captions": captions,
        "caption_mask": caption_mask,
    }
