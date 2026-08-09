import torch


class NaFlexCollator:
    """Convert variable-size PIL images to the official NaFlex batch contract."""

    def __init__(self, processor, max_num_patches=128, caption=False):
        self.processor = processor
        self.max_num_patches = int(max_num_patches)
        self.caption = bool(caption)
        if self.max_num_patches <= 0:
            raise ValueError('max_num_patches must be positive')

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        *,
        max_num_patches=128,
        caption=False,
    ):
        try:
            from transformers import Siglip2ImageProcessor
        except ImportError as exc:
            raise ImportError(
                'SigLIP2 NaFlex collation requires transformers'
            ) from exc
        processor = Siglip2ImageProcessor.from_pretrained(model_name)
        return cls(
            processor,
            max_num_patches=max_num_patches,
            caption=caption,
        )

    def __call__(self, batch):
        if self.caption:
            imgs, pids, camids, impaths, dsetids, captions = zip(*batch)
        else:
            imgs, pids, camids, impaths, dsetids = zip(*batch)
            captions = None

        encoded = self.processor(
            images=list(imgs),
            return_tensors='pt',
            max_num_patches=self.max_num_patches,
        )
        required = {
            'pixel_values',
            'pixel_attention_mask',
            'spatial_shapes',
        }
        missing = required.difference(encoded)
        if missing:
            raise RuntimeError(
                'NaFlex processor omitted required visual inputs: '
                f'{sorted(missing)}'
            )
        valid_patches = encoded['pixel_attention_mask'].to(torch.int64).sum(dim=1)
        declared_patches = encoded['spatial_shapes'].to(torch.int64).prod(dim=1)
        if not torch.equal(valid_patches, declared_patches):
            raise ValueError(
                'NaFlex processor returned a mask whose valid-token count '
                'does not equal height*width in spatial_shapes'
            )

        result = {
            'images': {name: encoded[name] for name in sorted(required)},
            'pids': torch.tensor(pids, dtype=torch.int64),
            'camids': torch.tensor(camids, dtype=torch.int64),
            'image_paths': impaths,
            'dataset_ids': dsetids,
            'captions': captions,
            'caption_mask': None,
        }
        if captions is not None:
            result['caption_mask'] = torch.tensor(
                [bool(caption) for caption in captions], dtype=torch.bool
            )
        return result


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
