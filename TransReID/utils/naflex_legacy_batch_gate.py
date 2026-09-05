"""Validation and override construction for the legacy-batch NaFlex gate."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


EXPECTED_RUNS = (
    "image_only",
    "caption_pid",
    "caption_codebook_delayed",
    "caption_codebook_relation",
)


def load_gate(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    validate_gate(gate)
    return gate


def validate_gate(gate):
    if gate.get("protocol") != "m-to-ms-legacy-batch-gate":
        raise ValueError("Unexpected gate protocol")
    if gate.get("source") != "market1501" or gate.get("target") != "msmt17":
        raise ValueError("This gate must remain Market1501 -> MSMT17")
    if gate.get("combineall") is not False:
        raise ValueError("The gate requires source-only combineall=false")
    if gate.get("world_size") != 2 or gate.get("global_batch_size") != 256:
        raise ValueError("The gate requires two ranks and global batch 256")
    solver = gate.get("solver_overrides", {})
    required_solver = {
        "MAX_EPOCHS": 60,
        "BASE_LR": 0.000005,
        "WARMUP_ITERS": 10,
        "STEPS": [30, 50],
        "CHECKPOINT_PERIOD": 2,
        "EVAL_PERIOD": 2,
        "SEED": 1234,
    }
    for key, expected in required_solver.items():
        if solver.get(key) != expected:
            raise ValueError(f"Legacy schedule mismatch: SOLVER.{key}")

    runs = gate.get("runs", [])
    names = tuple(run.get("name") for run in runs)
    if names != EXPECTED_RUNS:
        raise ValueError(f"Gate runs must be ordered as {EXPECTED_RUNS}")
    outputs = [run.get("output_relative") for run in runs]
    if len(outputs) != len(set(outputs)) or any(not value for value in outputs):
        raise ValueError("Gate output paths must be non-empty and unique")

    delayed = runs[2].get("objective_overrides", {})
    relation = runs[3].get("objective_overrides", {})
    if delayed.get("ATTRIBUTE_CODEBOOK.START_EPOCH") != 6:
        raise ValueError("Codebook gate must start at epoch 6")
    if relation.get("ATTRIBUTE_CODEBOOK.START_EPOCH") != 6:
        raise ValueError("Relation gate Codebook must start at epoch 6")
    if relation.get("ATTRIBUTE_RELATION.START_EPOCH") != 6:
        raise ValueError("Relation gate must start at epoch 6")


def _serialized(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def config_overrides(gate, run, runtime_root):
    # Runtime paths are interpreted by the remote Linux process even when a
    # dry-run is generated on Windows.
    root = PurePosixPath(runtime_root)
    overrides = [
        "DATASETS.SOURCES", gate["source"],
        "DATASETS.TARGETS", gate["target"],
        "DATASETS.COMBINEALL", str(gate["combineall"]),
        "DATASETS.ROOT_DIR", str(root / "datasets"),
        "MODEL.BACKBONE.PRETRAINED_NAME",
        str(root / "pretrained/siglip2/siglip2-base-patch16-naflex"),
        "OUTPUT_DIR", str(root / run["output_relative"]),
    ]
    for key, value in gate["solver_overrides"].items():
        overrides.extend([f"SOLVER.{key}", _serialized(value)])
    for key, value in run.get("objective_overrides", {}).items():
        overrides.extend([f"OBJECTIVE.{key}", _serialized(value)])

    if run["method"] != "image_only":
        caption = root / "Avada_ReID/TransReID/caption_tools/output/final/v2.4/captions.jsonl"
        overrides.extend(["OBJECTIVE.CAPTION.FILE", str(caption)])
    if "codebook" in run["method"]:
        codebook = root / "precomputed/attribute_codebooks/market1501_pr1"
        overrides.extend([
            "OBJECTIVE.ATTRIBUTE_CODEBOOK.CAPTION_FILE",
            str(root / "Avada_ReID/TransReID/caption_tools/output/final/v2.4/captions.jsonl"),
            "OBJECTIVE.ATTRIBUTE_CODEBOOK.PHRASE_BANK", str(codebook / "phrase_bank.pt"),
            "OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK", str(codebook / "codebook.pt"),
            "OBJECTIVE.ATTRIBUTE_CODEBOOK.MANIFEST", str(codebook / "manifest.json"),
        ])
    return overrides
