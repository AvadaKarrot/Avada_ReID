"""Evaluate SigLIP2 retrieval branches from an existing checkpoint."""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.build import build_datamanager
from engine.batch import normalize_batch
from evaluation.siglip2_branches import extract_siglip2_branch_features
from modeling import build_model
from utils.metrics import R1_mAP_eval


def _make_metric(num_query, feat_norm, max_rank):
    return R1_mAP_eval(
        num_query,
        max_rank=max_rank,
        feat_norm="yes" if feat_norm else "no",
    )


@torch.no_grad()
def evaluate_branches(model, loader, num_query, device, feat_norm, max_rank):
    model.eval()
    metrics = None
    dimensions = OrderedDict()

    for raw_batch in loader:
        batch = normalize_batch(raw_batch)
        images = batch["images"].to(device, non_blocking=True)
        features = extract_siglip2_branch_features(model, images)

        if metrics is None:
            metrics = OrderedDict(
                (name, _make_metric(num_query, feat_norm, max_rank))
                for name in features
            )
            for metric in metrics.values():
                metric.reset()
            dimensions.update(
                (name, int(feature.shape[1]))
                for name, feature in features.items()
            )

        for name, feature in features.items():
            metrics[name].update(
                (feature, batch["pids"], batch["camids"])
            )

    if metrics is None:
        raise RuntimeError("Target DataLoader is empty")

    results = OrderedDict()
    for name, metric in metrics.items():
        cmc, mAP, *_ = metric.compute()
        results[name] = {
            "dimension": dimensions[name],
            "mAP": float(mAP),
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]) if len(cmc) >= 5 else None,
            "rank10": float(cmc[9]) if len(cmc) >= 10 else None,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Zero-cost SigLIP2 retrieval-branch evaluation"
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data_manager = build_datamanager(cfg)
    model = build_model(cfg, num_classes=data_manager._num_train_pids)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_mAP": checkpoint.get("best_mAP"),
        "source": str(cfg.DATASETS.SOURCES),
        "target": str(cfg.DATASETS.TARGETS),
        "num_query": int(data_manager.num_query),
        "feature_normalization": str(cfg.TEST.FEAT_NORM),
        "branches": evaluate_branches(
            model,
            data_manager.test_loader,
            data_manager.num_query,
            device,
            feat_norm=cfg.TEST.FEAT_NORM == "yes",
            max_rank=args.max_rank,
        ),
    }

    payload = json.dumps(results, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
