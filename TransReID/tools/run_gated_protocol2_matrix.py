"""Run a Protocol-2 gate pair and expand the matrix only after a gain."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
BEST_MAP_PATTERN = re.compile(r"best_mAP=([0-9]+(?:\.[0-9]+)?)")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def best_map_from_log(path):
    values = [
        float(match.group(1))
        for match in BEST_MAP_PATTERN.finditer(
            Path(path).read_text(encoding="utf-8", errors="replace")
        )
    ]
    if not values:
        raise RuntimeError(f"No best_mAP value found in {path}")
    return max(values)


def _runner_command(matrix_path, logs_dir, selected=None):
    command = [
        sys.executable,
        "tools/run_protocol2_matrix.py",
        "--matrix",
        str(matrix_path),
        "--mode",
        "train",
        "--logs-dir",
        str(logs_dir),
    ]
    for name in selected or []:
        command.extend(["--only", name])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--logs-dir", required=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    gate = matrix.get("gate")
    if not gate:
        raise ValueError("The matrix does not define a gate")
    gate_runs = gate.get("runs", [])
    baselines = gate.get("baseline_best_mAP", {})
    if not gate_runs or set(gate_runs) != set(baselines):
        raise ValueError("Gate runs and baseline_best_mAP must match")

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = logs_dir / "gate_status.json"
    _write_json(
        status_path,
        {"status": "running_gate", "started_at": _timestamp()},
    )
    subprocess.run(
        _runner_command(matrix_path, logs_dir, gate_runs),
        cwd=PROJECT_DIR,
        check=True,
    )

    observed = {
        name: best_map_from_log(logs_dir / f"{name}.train.log")
        for name in gate_runs
    }
    gains = {
        name: observed[name] - float(baselines[name]) for name in gate_runs
    }
    if gate.get("require_all", True):
        passed = all(value > 0.0 for value in gains.values())
    else:
        passed = any(value > 0.0 for value in gains.values())
    payload = {
        "status": "gate_passed" if passed else "gate_not_passed",
        "checked_at": _timestamp(),
        "baseline_best_mAP": baselines,
        "observed_best_mAP": observed,
        "absolute_gains": gains,
        "require_all": bool(gate.get("require_all", True)),
    }
    _write_json(status_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not passed:
        return

    _write_json(
        status_path,
        {**payload, "status": "expanding_full_matrix"},
    )
    subprocess.run(
        _runner_command(matrix_path, logs_dir),
        cwd=PROJECT_DIR,
        check=True,
    )
    _write_json(
        status_path,
        {**payload, "status": "full_matrix_complete", "completed_at": _timestamp()},
    )


if __name__ == "__main__":
    main()
