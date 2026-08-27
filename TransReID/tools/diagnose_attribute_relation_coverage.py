"""Diagnose whether real training batches support Code-to-Code relations.

This tool deliberately reuses the production DataManager and sampler.  It
does not run a model: only the frozen Attribute targets already attached to
source records are inspected.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.attribute_store import ATTRIBUTE_BUCKETS
from data.build import build_datamanager
from engine.batch import normalize_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Attribute relation coverage in real batches"
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--min-hard-support", type=int, default=2)
    parser.add_argument("--min-soft-mass", type=float, default=1.0)
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Configuration overrides, identical to tools/train.py",
    )
    return parser.parse_args()


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _quantile(values, q):
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def _summary(values):
    if not values:
        return {"mean": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "p10": _quantile(values, 0.10),
        "median": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
    }


def _domain_values(batch, mode):
    if mode == "dataset":
        return [str(value) for value in batch["dataset_ids"]]
    return [str(int(value)) for value in batch["camids"].tolist()]


def _coverage(distribution, valid, quality, min_hard, min_mass):
    code_count = distribution.shape[1]
    if not bool(valid.any()):
        return {
            "valid_samples": 0,
            "hard_active_codes": 0,
            "soft_active_codes": 0,
            "hard_active_pairs": 0,
            "soft_active_pairs": 0,
        }
    selected = distribution[valid]
    hard = selected.argmax(dim=1)
    hard_support = torch.bincount(hard, minlength=code_count)
    soft_mass = (selected * quality[valid, None]).sum(dim=0)
    hard_active = int((hard_support >= min_hard).sum())
    soft_active = int((soft_mass >= min_mass).sum())
    return {
        "valid_samples": int(valid.sum()),
        "hard_active_codes": hard_active,
        "soft_active_codes": soft_active,
        "hard_active_pairs": hard_active * (hard_active - 1) // 2,
        "soft_active_pairs": soft_active * (soft_active - 1) // 2,
    }


def main():
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("max-batches must be positive")
    if args.min_hard_support <= 0 or args.min_soft_mass <= 0:
        raise ValueError("coverage thresholds must be positive")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    _seed_everything(int(cfg.SOLVER.SEED))
    manager = build_datamanager(cfg)

    bucket_rows = defaultdict(lambda: defaultdict(list))
    domain_rows = {
        "dataset": defaultdict(lambda: defaultdict(list)),
        "camera": defaultdict(lambda: defaultdict(list)),
    }
    batch_domain_counts = {"dataset": [], "camera": []}
    batches = 0

    for raw_batch in manager.train_loader:
        batch = normalize_batch(raw_batch)
        targets = batch.get("attribute_targets")
        if targets is None:
            raise RuntimeError("Attribute targets are not enabled")
        quality = targets["quality"].float()
        domain_values = {
            mode: _domain_values(batch, mode)
            for mode in ("dataset", "camera")
        }
        for mode, values in domain_values.items():
            batch_domain_counts[mode].append(len(set(values)))

        for bucket in ATTRIBUTE_BUCKETS:
            distribution = targets["distributions"][bucket].float()
            valid = targets["mask"][bucket].bool()
            row = _coverage(
                distribution, valid, quality,
                args.min_hard_support, args.min_soft_mass,
            )
            for name, value in row.items():
                bucket_rows[bucket][name].append(value)

            for mode, values in domain_values.items():
                for domain in sorted(set(values)):
                    group = torch.tensor(
                        [value == domain for value in values], dtype=torch.bool
                    )
                    group_row = _coverage(
                        distribution,
                        valid & group,
                        quality,
                        args.min_hard_support,
                        args.min_soft_mass,
                    )
                    key = f"{bucket}:{domain}"
                    for name, value in group_row.items():
                        domain_rows[mode][key][name].append(value)

        batches += 1
        if batches >= args.max_batches:
            break

    report = {
        "version": 1,
        "config_file": str(args.config_file),
        "sources": str(cfg.DATASETS.SOURCES),
        "targets": str(cfg.DATASETS.TARGETS),
        "sampler": str(cfg.DATALOADER.SAMPLER),
        "batch_size": int(cfg.SOLVER.IMS_PER_BATCH),
        "batches": batches,
        "thresholds": {
            "min_hard_support": args.min_hard_support,
            "min_soft_mass": args.min_soft_mass,
        },
        "batch_domain_counts": {
            mode: _summary(values)
            for mode, values in batch_domain_counts.items()
        },
        "buckets": {
            bucket: {
                name: _summary(values)
                for name, values in metrics.items()
            }
            for bucket, metrics in bucket_rows.items()
        },
        "domains": {
            mode: {
                key: {
                    name: _summary(values)
                    for name, values in metrics.items()
                }
                for key, metrics in rows.items()
            }
            for mode, rows in domain_rows.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
