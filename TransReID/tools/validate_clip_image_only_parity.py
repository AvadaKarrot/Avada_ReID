"""Numerically compare legacy and unified CLIP image-only implementations."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg as default_cfg
from data.build import build_datamanager
from engine.batch import normalize_batch
from loss.make_loss_clipreid import make_loss
from model.make_model_clipreid_base import make_model as build_legacy_model
from modeling import build_model as build_unified_model
from objectives import build_objective


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def max_difference(left, right):
    return float((left.float() - right.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-config",
        default="configs/experiments/clip_market_to_msmt17.yml",
    )
    parser.add_argument(
        "--unified-config",
        default="configs/experiments/clip_market_to_msmt_image_only.yml",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CLIP parity validation")

    legacy_cfg = default_cfg.clone()
    legacy_cfg.merge_from_file(args.legacy_config)
    legacy_cfg.defrost()
    legacy_cfg.DATALOADER.NUM_WORKERS = args.workers
    legacy_cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    legacy_cfg.freeze()

    unified_cfg = default_cfg.clone()
    unified_cfg.merge_from_file(args.unified_config)
    unified_cfg.defrost()
    unified_cfg.DATALOADER.NUM_WORKERS = args.workers
    unified_cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    unified_cfg.freeze()

    comparable_fields = (
        "DATASETS.SOURCES",
        "DATASETS.TARGETS",
        "DATASETS.TRANSFORMS",
        "INPUT.SIZE_TRAIN",
        "INPUT.SIZE_TEST",
        "INPUT.PIXEL_MEAN",
        "INPUT.PIXEL_STD",
        "SOLVER.BASE_LR",
        "SOLVER.WEIGHT_DECAY",
        "SOLVER.WEIGHT_DECAY_BIAS",
        "SOLVER.STEPS",
        "SOLVER.GAMMA",
        "SOLVER.WARMUP_FACTOR",
        "SOLVER.WARMUP_ITERS",
        "TEST.NECK_FEAT",
    )
    mismatches = []
    for path in comparable_fields:
        legacy_value = legacy_cfg
        unified_value = unified_cfg
        for component in path.split("."):
            legacy_value = getattr(legacy_value, component)
            unified_value = getattr(unified_value, component)
        if legacy_value != unified_value:
            mismatches.append(
                {"field": path, "legacy": legacy_value, "unified": unified_value}
            )
    if mismatches:
        raise RuntimeError(f"Parity configuration mismatch: {mismatches}")

    set_seed(legacy_cfg.SOLVER.SEED)
    manager = build_datamanager(unified_cfg)
    model_rng = capture_rng_state()

    legacy_model = build_legacy_model(
        legacy_cfg,
        num_class=manager._num_train_pids,
        camera_num=manager._num_train_cams,
        view_num=0,
    ).cuda()
    restore_rng_state(model_rng)
    unified_model = build_unified_model(
        unified_cfg, num_classes=manager._num_train_pids
    ).cuda()

    parameter_differences = {
        "classifier": max_difference(
            legacy_model.classifier.weight,
            unified_model.head.classifier.weight,
        ),
        "classifier_proj": max_difference(
            legacy_model.classifier_proj.weight,
            unified_model.head.classifier_proj.weight,
        ),
        "bnneck": max_difference(
            legacy_model.bottleneck.weight,
            unified_model.head.bnneck.weight,
        ),
        "bnneck_proj": max_difference(
            legacy_model.bottleneck_proj.weight,
            unified_model.head.bnneck_proj.weight,
        ),
    }

    source_batch = normalize_batch(next(iter(manager.train_loader)))
    images = source_batch["images"].cuda(non_blocking=True)
    pids = source_batch["pids"].cuda(non_blocking=True)
    legacy_model.train()
    unified_model.train()
    legacy_scores, legacy_features = legacy_model(images, pids)
    unified_outputs = unified_model(images)

    branch_differences = {
        "id_768": max_difference(
            legacy_scores[0], unified_outputs.id_logits[0]
        ),
        "id_512": max_difference(
            legacy_scores[1], unified_outputs.id_logits[1]
        ),
        "triplet_last": max_difference(
            legacy_features[0], unified_outputs.metric_features[0]
        ),
        "triplet_768": max_difference(
            legacy_features[1], unified_outputs.metric_features[1]
        ),
        "triplet_512": max_difference(
            legacy_features[2], unified_outputs.metric_features[2]
        ),
    }

    legacy_loss_fn, _ = make_loss(
        legacy_cfg, num_classes=manager._num_train_pids
    )
    legacy_loss = legacy_loss_fn(
        legacy_scores, legacy_features, pids, None
    )
    unified_objective = build_objective(unified_cfg).cuda()
    unified_losses = unified_objective(
        unified_outputs, {"pids": source_batch["pids"]}
    )
    loss_difference = abs(
        float(legacy_loss.item()) - float(unified_losses["total"].item())
    )

    target_batch = normalize_batch(next(iter(manager.test_loader)))
    target_images = target_batch["images"].cuda(non_blocking=True)
    legacy_model.eval()
    unified_model.eval()
    with torch.no_grad():
        legacy_embedding = legacy_model(target_images)
        unified_embedding = unified_model(target_images).embedding
    embedding_difference = max_difference(
        legacy_embedding, unified_embedding
    )

    tolerance = 1e-5
    result = {
        "config_mismatches": mismatches,
        "parameter_max_abs_diff": parameter_differences,
        "branch_max_abs_diff": branch_differences,
        "legacy_loss": float(legacy_loss.item()),
        "unified_loss": float(unified_losses["total"].item()),
        "loss_abs_diff": loss_difference,
        "target_embedding_shape": list(unified_embedding.shape),
        "target_embedding_max_abs_diff": embedding_difference,
        "tolerance": tolerance,
    }
    print(json.dumps(result, indent=2))
    all_differences = (
        list(parameter_differences.values())
        + list(branch_differences.values())
        + [loss_difference, embedding_difference]
    )
    if max(all_differences) > tolerance:
        raise RuntimeError("Legacy and unified CLIP paths are not numerically equal")
    print("CLIP_IMAGE_ONLY_PARITY_OK=YES")


if __name__ == "__main__":
    main()
