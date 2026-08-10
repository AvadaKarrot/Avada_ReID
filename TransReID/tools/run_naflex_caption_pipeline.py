"""Finish NaFlex caption S1 runs, then run the Protocol-2 matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/root/autodl-tmp")
S1_MATRIX = PROJECT_DIR / "configs/experiments/siglip2_naflex_caption_alignment_s1_30ep.json"
P2_MATRIX = PROJECT_DIR / "configs/experiments/protocol2_siglip2_naflex_caption_alignment_30ep.json"
CURRENT_LOGS = DATA_ROOT / "logs/siglip2_naflex_caption_pid_nce_m_to_ms"
PIPELINE_LOGS = DATA_ROOT / "logs/siglip2_naflex_caption_pipeline"

CURRENT_RUN = "siglip2_naflex_m_to_ms_caption_alignment_s1"
S1_REMAINING = (
    "siglip2_naflex_ms_to_m_caption_alignment_s1",
    "siglip2_naflex_ms_to_c3_caption_alignment_s1",
    "siglip2_naflex_c3_to_ms_caption_alignment_s1",
    "siglip2_naflex_c3_to_m_caption_alignment_s1",
    "siglip2_naflex_m_to_c3_caption_alignment_s1",
)
P2_RUNS = (
    "siglip2_naflex_m_ms_cs_to_c3_caption_alignment_30ep",
    "siglip2_naflex_m_cs_c3_to_ms_caption_alignment_30ep",
    "siglip2_naflex_ms_cs_c3_to_m_caption_alignment_30ep",
)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
    ).strip()


def _run(command):
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _wait_for_current():
    complete = CURRENT_LOGS / f"{CURRENT_RUN}.complete.json"
    failed = CURRENT_LOGS / f"{CURRENT_RUN}.failed.json"
    pid_path = CURRENT_LOGS / "supervisor.pid"
    while not complete.is_file():
        if failed.is_file():
            raise RuntimeError("Current M->MS run failed")
        if not pid_path.is_file():
            raise RuntimeError("Current M->MS supervisor PID is missing")
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(
                "Current M->MS supervisor stopped without completion"
            ) from error
        time.sleep(30)


def _matrix_output(matrix_path, run_name):
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for run in matrix["runs"]:
        if run["name"] == run_name:
            return Path(run["output_dir"])
    raise KeyError(run_name)


def _compact(output_dir):
    best = output_dir / "model_best.pth.tar"
    if not best.is_file():
        raise RuntimeError(f"Cannot compact without model_best: {output_dir}")
    for name in ("checkpoint_latest.pth.tar", "model_last.pth.tar"):
        (output_dir / name).unlink(missing_ok=True)
    for path in output_dir.glob("*_epoch*.pth"):
        path.unlink()


def _run_selected(runner, matrix, run_name, logs_dir, mode="train"):
    _run(
        [
            sys.executable,
            str(PROJECT_DIR / runner),
            "--matrix",
            str(matrix),
            "--mode",
            mode,
            "--only",
            run_name,
            "--logs-dir",
            str(logs_dir),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    PIPELINE_LOGS.mkdir(parents=True, exist_ok=True)
    running = PIPELINE_LOGS / "pipeline.running.json"
    failed = PIPELINE_LOGS / "pipeline.failed.json"
    complete = PIPELINE_LOGS / "pipeline.complete.json"
    state = {"started_at": _timestamp(), "expected_head": args.expected_head}
    _write_json(running, state)
    try:
        if _git_head() != args.expected_head:
            raise RuntimeError("Git HEAD does not match the pinned pipeline commit")
        _wait_for_current()
        _compact(_matrix_output(S1_MATRIX, CURRENT_RUN))

        s1_logs = PIPELINE_LOGS / "single_domain"
        for run_name in S1_REMAINING:
            _run_selected(
                "tools/run_backbone_transfer_matrix.py",
                S1_MATRIX,
                run_name,
                s1_logs,
            )
            _compact(_matrix_output(S1_MATRIX, run_name))

        p2_logs = PIPELINE_LOGS / "protocol2"
        for run_name in P2_RUNS:
            _run_selected(
                "tools/run_protocol2_matrix.py",
                P2_MATRIX,
                run_name,
                p2_logs,
                mode="smoke",
            )
        for run_name in P2_RUNS:
            _run_selected(
                "tools/run_protocol2_matrix.py",
                P2_MATRIX,
                run_name,
                p2_logs,
            )
            _compact(_matrix_output(P2_MATRIX, run_name))

        state["completed_at"] = _timestamp()
        _write_json(complete, state)
        running.unlink(missing_ok=True)
    except Exception as error:
        state.update(failed_at=_timestamp(), error=str(error))
        _write_json(failed, state)
        running.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
