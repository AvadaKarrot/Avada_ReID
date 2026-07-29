"""Short, reproducible smoke test for the legacy visual CLIP-ReID baseline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import cfg as default_cfg
from data.build import build_datamanager
from engine.batch import normalize_batch
from loss.make_loss_clipreid import make_loss
from model.make_model_clipreid_base import make_model
from solver.make_optimizer_clipreid import make_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Market1501 training and MSMT17 image-only inference"
    )
    parser.add_argument(
        "--config-file",
        default=str(PROJECT_DIR / "configs" / "visual" / "clip_base.yml"),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--target-iters", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = default_cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.DATASETS.ROOT_DIR = args.data_root
    cfg.DATASETS.SOURCES = ["market1501"]
    cfg.DATASETS.TARGETS = ["msmt17"]
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.TEST.IMS_PER_BATCH = args.test_batch_size
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.DEVICE_ID = "0"
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.CAPTION = False
    cfg.OBJECTIVE.CAPTION.ENABLED = False
    cfg.TEST.GRADCAM = False
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    return cfg


def gradients_are_finite(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("No trainable parameter received a gradient")
    return all(torch.isfinite(gradient).all().item() for gradient in gradients)


def main() -> None:
    args = parse_args()
    if args.train_iters < 1 or args.target_iters < 1:
        raise ValueError("train-iters and target-iters must both be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this smoke test")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = build_cfg(args)
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = build_datamanager(cfg)
    model = make_model(
        cfg,
        num_class=manager._num_train_pids,
        camera_num=manager._num_train_cams,
        view_num=0,
    ).cuda()
    loss_fn, center_criterion = make_loss(
        cfg, num_classes=manager._num_train_pids
    )
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scaler = torch.amp.GradScaler("cuda")

    losses = []
    skipped_steps = 0
    max_attempts = max(args.train_iters * 4, args.train_iters + 10)
    model.train()
    for attempt, legacy_batch in enumerate(manager.train_loader, start=1):
        batch = normalize_batch(legacy_batch)
        if batch["captions"] is not None or batch["caption_mask"] is not None:
            raise RuntimeError("Caption data entered the image-only baseline")

        images = batch["images"].cuda(non_blocking=True)
        pids = batch["pids"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            scores, features = model(images, pids)
            loss = loss_fn(scores, features, pids, None)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss at attempt {attempt}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients = gradients_are_finite(model)
        scaler.step(optimizer)
        scaler.update()
        if not finite_gradients:
            skipped_steps += 1
            print(
                f"TRAIN_ATTEMPT={attempt} SKIPPED_NONFINITE_GRADIENT=YES "
                f"NEXT_SCALE={scaler.get_scale():.1f}",
                flush=True,
            )
            if attempt >= max_attempts:
                raise RuntimeError(
                    "Dynamic loss scaling did not recover within "
                    f"{max_attempts} attempts"
                )
            continue

        losses.append(loss.item())
        print(
            f"TRAIN_ITER={len(losses)} ATTEMPT={attempt} "
            f"LOSS={loss.item():.6f} SCALE={scaler.get_scale():.1f}",
            flush=True,
        )
        if len(losses) >= args.train_iters:
            break

    if len(losses) != args.train_iters:
        raise RuntimeError(
            f"Only completed {len(losses)} of {args.train_iters} iterations"
        )

    checkpoint_path = output_dir / "clipreid_smoke_checkpoint.pth.tar"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": len(losses),
            "caption_enabled": False,
            "source": "market1501",
            "target": "msmt17",
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint restore mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    model.eval()
    target_batches = 0
    target_images = 0
    with torch.no_grad():
        for legacy_batch in manager.test_loader:
            batch = normalize_batch(legacy_batch)
            images = batch["images"].cuda(non_blocking=True)
            embeddings = model(images)
            if embeddings.ndim != 2 or embeddings.shape[1] != 1280:
                raise RuntimeError(
                    f"Expected 1280-D embeddings, got {tuple(embeddings.shape)}"
                )
            if not torch.isfinite(embeddings).all().item():
                raise RuntimeError("Non-finite target-domain embedding")
            target_batches += 1
            target_images += images.shape[0]
            if target_batches >= args.target_iters:
                break

    print(f"SOURCE_DATASET=market1501")
    print(f"TARGET_DATASET=msmt17")
    print(f"CAPTION_ENABLED={cfg.OBJECTIVE.CAPTION.ENABLED}")
    print(f"TRAIN_ITERATIONS={len(losses)}")
    print(f"SKIPPED_AMP_STEPS={skipped_steps}")
    print(f"INITIAL_LOSS={losses[0]:.6f}")
    print(f"FINAL_LOSS={losses[-1]:.6f}")
    print(f"TARGET_BATCHES={target_batches}")
    print(f"TARGET_IMAGES={target_images}")
    print(f"TARGET_EMBEDDING_DIM=1280")
    print(f"CHECKPOINT={checkpoint_path}")
    print(
        "MAX_GPU_MEMORY_MIB="
        f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}"
    )
    print("CLIPREID_BASELINE_SMOKE_OK=YES")


if __name__ == "__main__":
    main()
