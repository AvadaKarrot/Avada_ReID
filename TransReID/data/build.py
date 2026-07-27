def imagedata_kwargs(cfg):
    """Translate project configuration to the legacy ImageDataManager API."""

    caption_enabled = bool(
        getattr(getattr(cfg, "OBJECTIVE", object()), "CAPTION", False)
        and getattr(cfg.OBJECTIVE.CAPTION, "ENABLED", False)
    )
    return {
        "root": cfg.DATASETS.ROOT_DIR,
        "sources": cfg.DATASETS.SOURCES,
        "targets": cfg.DATASETS.TARGETS,
        "height": cfg.INPUT.SIZE_TRAIN[0],
        "width": cfg.INPUT.SIZE_TRAIN[1],
        "transforms": cfg.DATASETS.TRANSFORMS,
        "norm_mean": cfg.INPUT.PIXEL_MEAN,
        "norm_std": cfg.INPUT.PIXEL_STD,
        "batch_size_train": cfg.SOLVER.IMS_PER_BATCH,
        "batch_size_test": cfg.TEST.IMS_PER_BATCH,
        "workers": cfg.DATALOADER.NUM_WORKERS,
        "num_instances": cfg.DATALOADER.NUM_INSTANCE,
        "train_sampler": cfg.DATALOADER.SAMPLER,
        "dist_train": cfg.MODEL.DIST_TRAIN,
        "randomerase_prob": cfg.INPUT.RE_PROB,
        "padding": cfg.INPUT.PADDING,
        "sobel_prob": cfg.INPUT.SOBEL_PROB,
        "hdrnet": cfg.HDRNET.HDR_NET,
        "caption": caption_enabled,
    }


def build_datamanager(cfg):
    from .datamanager import ImageDataManager

    return ImageDataManager(**imagedata_kwargs(cfg))
