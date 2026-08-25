"""Validate semantic artifacts and their source-only provenance manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from semantic import (
    load_torch_artifact,
    sha256_file,
    validate_codebook_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate an Attribute semantic codebook"
    )
    parser.add_argument("artifact_dir")
    parser.add_argument(
        "--forbid-dataset", action="append", default=[],
        help="Fail if a target dataset appears in source_datasets",
    )
    parser.add_argument(
        "--verify-caption-file", action="store_true",
        help="Re-hash the original Caption JSONL named by the manifest",
    )
    return parser.parse_args()


def validate_directory(
    artifact_dir,
    *,
    forbidden_datasets=(),
    verify_caption_file=False,
):
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    source_datasets = {
        str(value).strip().lower()
        for value in manifest.get("source_datasets", ())
    }
    overlap = source_datasets.intersection(
        str(value).strip().lower() for value in forbidden_datasets
    )
    if overlap:
        errors.append(
            "forbidden target datasets appear in sources: "
            + ", ".join(sorted(overlap))
        )

    artifacts = manifest.get("artifacts", {})
    loaded = {}
    for name in ("phrase_bank", "codebook"):
        metadata = artifacts.get(name, {})
        path = artifact_dir / metadata.get("path", "")
        if not path.is_file():
            errors.append(f"missing artifact {name}: {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != metadata.get("sha256"):
            errors.append(f"{name} SHA256 mismatch")
            continue
        loaded[name] = load_torch_artifact(path)

    validation = {"valid": False, "errors": [], "summary": {}}
    if "phrase_bank" in loaded and "codebook" in loaded:
        validation = validate_codebook_artifact(
            loaded["codebook"], loaded["phrase_bank"]
        )
        errors.extend(validation["errors"])

    if verify_caption_file:
        caption_path = Path(manifest.get("caption_file", ""))
        if not caption_path.is_file():
            errors.append(f"Caption file is unavailable: {caption_path}")
        elif sha256_file(caption_path) != manifest.get("caption_sha256"):
            errors.append("Caption JSONL SHA256 mismatch")

    return {
        "valid": not errors,
        "errors": errors,
        "summary": validation["summary"],
        "source_datasets": sorted(source_datasets),
        "domain_mode": manifest.get("domain_mode"),
    }


def main():
    args = parse_args()
    result = validate_directory(
        args.artifact_dir,
        forbidden_datasets=args.forbid_dataset,
        verify_caption_file=args.verify_caption_file,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
