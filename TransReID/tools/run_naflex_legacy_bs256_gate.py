"""Run the two-GPU/global-batch-256 NaFlex M-to-MS gate safely."""

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

from utils.distributed_launcher import distributed_overrides, validate_gpu_ids, wrap_torchrun
from utils.naflex_legacy_batch_gate import config_overrides, load_gate


DEFAULT_GATE = PROJECT_DIR / "configs/experiments/gates/naflex_m_to_ms_legacy_bs256_60ep.json"


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()


def build_command(gate, run, runtime_root, *, resume=False):
    command = [
        sys.executable,
        "tools/train.py",
        "--config_file",
        run["base_config"],
        *config_overrides(gate, run, runtime_root),
    ]
    if resume:
        command.extend(["SOLVER.RESUME_TRAIN", "True"])
    command.extend(distributed_overrides(
        nproc_per_node=gate["world_size"],
        global_batch_size=gate["global_batch_size"],
    ))
    return wrap_torchrun(command, nproc_per_node=gate["world_size"])


def _select(gate, names):
    requested = set(names)
    runs = [run for run in gate["runs"] if run["name"] in requested]
    missing = requested - {run["name"] for run in runs}
    if missing:
        raise ValueError(f"Unknown gate runs: {sorted(missing)}")
    return runs


def _compact(output_dir):
    best = output_dir / "model_best.pth.tar"
    if not best.is_file() or best.stat().st_size == 0:
        raise RuntimeError(f"Cannot compact without a non-empty model_best: {output_dir}")
    removed = []
    for name in ("checkpoint_latest.pth.tar", "model_last.pth.tar"):
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for path in output_dir.glob("*_epoch*.pth"):
        path.unlink()
        removed.append(path.name)
    return sorted(removed)


def _run_one(gate, run, runtime_root, logs_dir, gpu_ids):
    output_dir = runtime_root / run["output_relative"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    running = logs_dir / f"{run['name']}.running.json"
    complete = logs_dir / f"{run['name']}.complete.json"
    failed = logs_dir / f"{run['name']}.failed.json"
    log_path = logs_dir / f"{run['name']}.train.log"
    if complete.is_file():
        print(f"SKIP_COMPLETE {run['name']}", flush=True)
        return
    resume = (output_dir / "checkpoint_latest.pth.tar").is_file()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise RuntimeError(f"Refusing non-resumable non-empty output: {output_dir}")

    command = build_command(gate, run, runtime_root, resume=resume)
    metadata = {
        "name": run["name"], "method": run["method"], "source": gate["source"],
        "target": gate["target"], "global_batch_size": gate["global_batch_size"],
        "world_size": gate["world_size"], "gpu_ids": gpu_ids, "resume": resume,
        "started_at": _timestamp(), "git_head": _git_head(), "command": command,
    }
    _write_json(running, metadata)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "HF_HOME": str(runtime_root / "hf_cache"),
        "HF_HUB_CACHE": str(runtime_root / "hf_cache/hub"),
        "TORCH_HOME": str(runtime_root / "hf_cache/torch"),
        "XDG_CACHE_HOME": str(runtime_root / "hf_cache/xdg"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=PROJECT_DIR, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"torchrun exited with {result.returncode}")
        last = output_dir / "model_last.pth.tar"
        if not last.is_file() or last.stat().st_size == 0:
            raise RuntimeError("Successful process did not produce model_last")
        removed = _compact(output_dir)
        metadata.update(completed_at=_timestamp(), returncode=0, removed_non_best=removed)
        _write_json(complete, metadata)
        failed.unlink(missing_ok=True)
        print(f"COMPLETE {run['name']} retained=model_best.pth.tar", flush=True)
    except Exception as error:
        metadata.update(failed_at=_timestamp(), error=str(error))
        _write_json(failed, metadata)
        raise
    finally:
        running.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_GATE))
    parser.add_argument("--mode", choices=("dry-run", "train"), default="dry-run")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--runtime-root", default="/root/autodl-tmp")
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("--gpu-ids", default="0,2")
    args = parser.parse_args()

    gate = load_gate(args.matrix)
    validate_gpu_ids(args.gpu_ids, nproc_per_node=gate["world_size"])
    if args.mode == "train" and not args.only:
        parser.error("train mode requires at least one explicit --only run")
    runs = _select(gate, args.only) if args.only else gate["runs"]
    runtime_root = Path(args.runtime_root)
    logs_dir = Path(args.logs_dir) if args.logs_dir else runtime_root / "logs/naflex_legacy_bs256_gate/m_to_ms"
    if args.mode == "dry-run":
        for run in runs:
            print(" ".join(build_command(gate, run, runtime_root)))
        return
    for run in runs:
        _run_one(gate, run, runtime_root, logs_dir, args.gpu_ids)


if __name__ == "__main__":
    main()
