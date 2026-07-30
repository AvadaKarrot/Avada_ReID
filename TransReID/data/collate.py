import torch


def collate_fn(batch):
    imgs, pids, camids, impaths, dsetids = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        impaths,
        dsetids,
    )


def caption_collate_fn(batch):
    imgs, pids, camids, impaths, dsetids, captions = zip(*batch)
    return {
        "images": torch.stack(imgs, dim=0),
        "pids": torch.tensor(pids, dtype=torch.int64),
        "camids": torch.tensor(camids, dtype=torch.int64),
        "image_paths": impaths,
        "dataset_ids": dsetids,
        "captions": captions,
        "caption_mask": torch.tensor(
            [bool(caption) for caption in captions],
            dtype=torch.bool,
        ),
    }


def val_collate_fn(batch):
    imgs, pids, raw_camids, impaths, dsetids = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        raw_camids,
        torch.tensor(raw_camids, dtype=torch.int64),
        impaths,
        dsetids,
    )
