"""Protocol-2 matrix contracts shared by validation and execution tools."""

from __future__ import annotations

import json
from pathlib import Path


METHOD_CONFIGS = {
    "image_only": "configs/experiments/clip_market_to_msmt_image_only.yml",
    "caption_alignment": (
        "configs/experiments/"
        "clip_market_to_msmt_caption_alignment_v2_4.yml"
    ),
}

EXPECTED_PROTOCOLS = {
    (
        ("market1501", "msmt17", "cuhksysu"),
        "cuhk03",
    ),
    (
        ("market1501", "cuhksysu", "cuhk03"),
        "msmt17",
    ),
    (
        ("msmt17", "cuhksysu", "cuhk03"),
        "market1501",
    ),
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


def load_protocol2_matrix(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    validate_protocol2_matrix(matrix)
    return matrix


def validate_protocol2_matrix(matrix):
    if matrix.get("protocol") != "Protocol-2":
        raise ValueError("Matrix protocol must be 'Protocol-2'")
    if matrix.get("combineall") is not False:
        raise ValueError("Protocol-2 requires combineall=false")

    solver_overrides = matrix.get("solver_overrides", {})
    unknown_solver_keys = set(solver_overrides) - ALLOWED_SOLVER_OVERRIDES
    if unknown_solver_keys:
        raise ValueError(
            "Unsupported Protocol-2 solver overrides: "
            f"{sorted(unknown_solver_keys)}"
        )

    runs = matrix.get("runs", [])
    if len(runs) != 6:
        raise ValueError("Protocol-2 CLIP matrix must contain six runs")
    names = [run.get("name") for run in runs]
    outputs = [run.get("output_dir") for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("Protocol-2 run names must be unique")
    if len(set(outputs)) != len(outputs):
        raise ValueError("Protocol-2 output directories must be unique")

    observed = {}
    for run in runs:
        method = run.get("method")
        if method not in METHOD_CONFIGS:
            raise ValueError(f"Unknown Protocol-2 method: {method!r}")
        if run.get("base_config") != METHOD_CONFIGS[method]:
            raise ValueError(
                f"Run {run.get('name')!r} uses the wrong base config"
            )
        key = (tuple(run.get("sources", [])), run.get("target"))
        if key not in EXPECTED_PROTOCOLS:
            raise ValueError(f"Unexpected Protocol-2 source/target row: {key}")
        observed.setdefault(key, set()).add(method)

    expected_methods = set(METHOD_CONFIGS)
    if set(observed) != EXPECTED_PROTOCOLS:
        raise ValueError("Protocol-2 matrix is missing a source/target row")
    if any(methods != expected_methods for methods in observed.values()):
        raise ValueError(
            "Every Protocol-2 row requires image-only and Caption runs"
        )


def config_overrides(matrix, run):
    overrides = [
        "DATASETS.SOURCES",
        ",".join(run["sources"]),
        "DATASETS.TARGETS",
        run["target"],
        "DATASETS.COMBINEALL",
        str(bool(matrix["combineall"])),
        "OUTPUT_DIR",
        run["output_dir"],
    ]
    for key, value in matrix.get("solver_overrides", {}).items():
        serialized = json.dumps(value) if isinstance(value, list) else str(value)
        overrides.extend(
            [f"SOLVER.{key}", serialized]
        )
    return overrides
