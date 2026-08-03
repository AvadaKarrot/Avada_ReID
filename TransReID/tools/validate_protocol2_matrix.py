"""Validate real source splits and Caption coverage for Protocol-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg as default_cfg
from data.build import build_datamanager
from utils.protocol2_matrix import config_overrides, load_protocol2_matrix


EXPECTED_SOURCE = {
    "market1501": {"images": 12936, "pids": 751},
    "msmt17": {"images": 30248, "pids": 1041},
    "cuhksysu": {"images": 34574, "pids": 11934},
    "cuhk03": {"images": 7365, "pids": 767},
}
EXPECTED_TARGET = {
    "market1501": {"query": 3368, "gallery": 15913},
    "msmt17": {"query": 11659, "gallery": 82161},
    "cuhk03": {"query": 1400, "gallery": 5332},
}


def _resolved_config(matrix, run):
    current = default_cfg.clone()
    current.merge_from_file(str(PROJECT_DIR / run["base_config"]))
    current.merge_from_list(config_overrides(matrix, run))
    current.freeze()
    return current


def _validate_run(matrix, run):
    current = _resolved_config(matrix, run)
    manager = build_datamanager(current)
    train = manager.train_loader.dataset.train
    expected_images = sum(
        EXPECTED_SOURCE[name]["images"] for name in run["sources"]
    )
    expected_pids = sum(
        EXPECTED_SOURCE[name]["pids"] for name in run["sources"]
    )
    if len(train) != expected_images:
        raise RuntimeError(
            f"{run['name']}: train images {len(train)} != {expected_images}"
        )
    if manager.num_train_pids != expected_pids:
        raise RuntimeError(
            f"{run['name']}: train PIDs {manager.num_train_pids} "
            f"!= {expected_pids}"
        )
    if len({item[3] for item in train}) != 3:
        raise RuntimeError(f"{run['name']}: expected three source domains")

    caption_enabled = bool(current.OBJECTIVE.CAPTION.ENABLED)
    if caption_enabled:
        covered = sum(len(item) == 5 and bool(item[4]) for item in train)
        if covered != expected_images:
            raise RuntimeError(
                f"{run['name']}: Caption coverage {covered}/{expected_images}"
            )
    elif any(len(item) != 4 for item in train):
        raise RuntimeError(
            f"{run['name']}: image-only source contains Caption fields"
        )

    target_dataset = manager.test_loader.dataset
    target_expected = EXPECTED_TARGET[run["target"]]
    if len(target_dataset.query) != target_expected["query"]:
        raise RuntimeError(f"{run['name']}: target query count mismatch")
    if len(target_dataset.gallery) != target_expected["gallery"]:
        raise RuntimeError(f"{run['name']}: target gallery count mismatch")
    if any(len(item) != 4 for item in target_dataset.test):
        raise RuntimeError(f"{run['name']}: target unexpectedly has Caption")

    return {
        "name": run["name"],
        "method": run["method"],
        "sources": run["sources"],
        "target": run["target"],
        "train_images": len(train),
        "train_pids": manager.num_train_pids,
        "caption_covered": expected_images if caption_enabled else 0,
        "target_query": len(target_dataset.query),
        "target_gallery": len(target_dataset.gallery),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate the real Protocol-2 data matrix"
    )
    parser.add_argument(
        "--matrix",
        default=str(
            PROJECT_DIR
            / "configs"
            / "experiments"
            / "protocol2_clip_matrix.json"
        ),
    )
    args = parser.parse_args()
    matrix = load_protocol2_matrix(args.matrix)
    results = [_validate_run(matrix, run) for run in matrix["runs"]]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
