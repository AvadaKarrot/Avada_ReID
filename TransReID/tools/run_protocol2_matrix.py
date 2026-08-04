"""Run a verified Protocol-2 matrix sequentially and resumably."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.protocol2_matrix import config_overrides, load_protocol2_matrix


DEFAULT_MATRIX = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "protocol2_clip_matrix.json"
)

DATA_ROOT = Path("/root/autodl-tmp")
OFFLINE_ENV = {
    "HF_HOME": str(DATA_ROOT / "hf_cache"),
    "HF_HUB_CACHE": str(DATA_ROOT / "hf_cache" / "hub"),
    "TORCH_HOME": str(DATA_ROOT / "hf_cache" / "torch"),
    "XDG_CACHE_HOME": str(DATA_ROOT / "hf_cache" / "xdg"),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        text=True,
    ).strip()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command(matrix, run, mode, *, resume=False):
    entry = (
        "tools/smoke_unified_caption.py"
        if mode == "smoke"
        else "tools/train.py"
    )
    command = [
        sys.executable,
        entry,
        "--config_file",
        run["base_config"],
        *config_overrides(matrix, run),
    ]
    if mode == "smoke":
        smoke_root = matrix.get(
            "smoke_output_root",
            "/root/autodl-tmp/experiments/protocol2/smoke",
        )
        smoke_output = str(Path(smoke_root) / run["name"])
        command.extend(
            [
                "SOLVER.IMS_PER_BATCH",
                "16",
                "TEST.IMS_PER_BATCH",
                "32",
                "DATALOADER.NUM_WORKERS",
                "2",
                "OUTPUT_DIR",
                smoke_output,
            ]
        )
    elif resume:
        command.extend(["SOLVER.RESUME_TRAIN", "True"])
    return command


def _select_runs(matrix, selected):
    if not selected:
        return matrix["runs"]
    requested = set(selected)
    runs = [run for run in matrix["runs"] if run["name"] in requested]
    missing = requested - {run["name"] for run in runs}
    if missing:
        raise ValueError(f"Unknown run names: {sorted(missing)}")
    return runs


def _run_one(matrix, run, mode, logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run['name']}.{mode}.log"
    output_dir = Path(run["output_dir"])
    complete = logs_dir / f"{run['name']}.complete.json"
    failed = logs_dir / f"{run['name']}.failed.json"
    running = logs_dir / f"{run['name']}.running.json"

    if mode == "train" and complete.is_file():
        print(f"SKIP_COMPLETE {run['name']}", flush=True)
        return
    if mode == "train" and (output_dir / "model_last.pth.tar").is_file():
        _write_json(
            complete,
            {"name": run["name"], "completed_at": _timestamp()},
        )
        print(f"SKIP_VERIFIED_LAST {run['name']}", flush=True)
        return

    resume = mode == "train" and (
        output_dir / "checkpoint_latest.pth.tar"
    ).is_file()
    if mode == "train" and output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise RuntimeError(
                f"Refusing non-resumable non-empty output: {output_dir}"
            )

    command = _command(matrix, run, mode, resume=resume)
    metadata = {
        "name": run["name"],
        "backbone": matrix.get("backbone", "clip"),
        "method": run["method"],
        "sources": run["sources"],
        "target": run["target"],
        "mode": mode,
        "resume": resume,
        "started_at": _timestamp(),
        "command": command,
        "git_head": _git_head(),
    }
    _write_json(running, metadata)
    print(f"START {run['name']} resume={resume}", flush=True)

    environment = os.environ.copy()
    environment.update(OFFLINE_ENV)
    with log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        metadata.update(
            returncode=result.returncode,
            failed_at=_timestamp(),
        )
        _write_json(failed, metadata)
        running.unlink(missing_ok=True)
        raise RuntimeError(
            f"Protocol-2 run failed: {run['name']} ({result.returncode})"
        )
    if mode == "train" and not (
        output_dir / "model_last.pth.tar"
    ).is_file():
        metadata.update(failed_at=_timestamp(), reason="missing model_last")
        _write_json(failed, metadata)
        running.unlink(missing_ok=True)
        raise RuntimeError(
            f"Protocol-2 run lacks final checkpoint: {run['name']}"
        )

    metadata.update(completed_at=_timestamp(), returncode=0)
    if mode == "train":
        _write_json(complete, metadata)
    running.unlink(missing_ok=True)
    print(f"COMPLETE {run['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run a verified Protocol-2 matrix"
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument(
        "--mode", choices=("dry-run", "smoke", "train"), default="dry-run"
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--logs-dir",
        default="/root/autodl-tmp/logs/protocol2_clip",
    )
    args = parser.parse_args()

    matrix = load_protocol2_matrix(args.matrix)
    runs = _select_runs(matrix, args.only)
    if args.mode == "dry-run":
        for run in runs:
            print(" ".join(_command(matrix, run, "train")))
        return

    logs_dir = Path(args.logs_dir)
    for run in runs:
        _run_one(matrix, run, args.mode, logs_dir)


if __name__ == "__main__":
    main()
