"""Contracts for the one-source-to-one-target backbone matrix."""

from __future__ import annotations

import json
from pathlib import Path


BACKBONE_CONFIGS = {
    "clip": "configs/experiments/clip_market_to_msmt_image_only.yml",
    "dinov3": "configs/experiments/dinov3_multibranch_image_only.yml",
    "siglip2": "configs/experiments/siglip2_multibranch_image_only.yml",
}

EXPECTED_DIRECTIONS = {
    ("msmt17", "market1501"),
    ("market1501", "msmt17"),
    ("msmt17", "cuhk03"),
    ("cuhk03", "msmt17"),
    ("cuhk03", "market1501"),
    ("market1501", "cuhk03"),
}

ALLOWED_SOLVER_OVERRIDES = {
    "MAX_EPOCHS",
    "BASE_LR",
    "WARMUP_ITERS",
    "WARMUP_FACTOR",
    "WARMUP_METHOD",
    "STEPS",
    "GAMMA",
    "EVAL_PERIOD",
    "CHECKPOINT_PERIOD",
}


def load_backbone_transfer_matrix(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    validate_backbone_transfer_matrix(matrix)
    return matrix


def validate_backbone_transfer_matrix(matrix):
    if matrix.get("protocol") != "one-source-one-target":
        raise ValueError("Unexpected backbone transfer protocol")
    if matrix.get("combineall") is not False:
        raise ValueError("Source-only transfer requires combineall=false")

    solver_overrides = matrix.get("solver_overrides", {})
    unknown = set(solver_overrides) - ALLOWED_SOLVER_OVERRIDES
    if unknown:
        raise ValueError(f"Unsupported solver overrides: {sorted(unknown)}")

    runs = matrix.get("runs", [])
    expected_count = len(EXPECTED_DIRECTIONS) * len(BACKBONE_CONFIGS)
    if len(runs) != expected_count:
        raise ValueError(
            f"Backbone transfer matrix requires {expected_count} runs"
        )
    names = [run.get("name") for run in runs]
    outputs = [run.get("output_dir") for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("Run names must be unique")
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output directories must be unique")

    observed = {}
    for run in runs:
        backbone = run.get("backbone")
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone!r}")
        if run.get("base_config") != BACKBONE_CONFIGS[backbone]:
            raise ValueError(f"Wrong base config for {run.get('name')!r}")
        direction = (run.get("source"), run.get("target"))
        if direction not in EXPECTED_DIRECTIONS:
            raise ValueError(f"Unexpected transfer direction: {direction}")
        observed.setdefault(direction, set()).add(backbone)

    if set(observed) != EXPECTED_DIRECTIONS:
        raise ValueError("Matrix is missing a transfer direction")
    expected_backbones = set(BACKBONE_CONFIGS)
    if any(backbones != expected_backbones for backbones in observed.values()):
        raise ValueError("Each direction requires all three backbones")


def config_overrides(matrix, run):
    overrides = [
        "DATASETS.SOURCES",
        run["source"],
        "DATASETS.TARGETS",
        run["target"],
        "DATASETS.COMBINEALL",
        str(bool(matrix["combineall"])),
        "OUTPUT_DIR",
        run["output_dir"],
    ]
    for key, value in matrix.get("solver_overrides", {}).items():
        serialized = json.dumps(value) if isinstance(value, list) else str(value)
        overrides.extend([f"SOLVER.{key}", serialized])
    return overrides
