"""Build a source-only Attribute semantic codebook for SigLIP2 NaFlex."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build source-only Attribute semantic anchors"
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caption-file", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stability-seed",
        type=int,
        action="append",
        dest="stability_seeds",
        help="Repeat for each auxiliary clustering seed",
    )
    parser.add_argument("--max-codes", type=int, default=64)
    parser.add_argument("--merge-similarity", type=float, default=0.90)
    parser.add_argument(
        "--max-compactness-drop", type=float, default=0.01
    )
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-compactness", type=float, default=0.0)
    parser.add_argument("--min-domain-coverage", type=int, default=1)
    parser.add_argument(
        "--domain-mode",
        choices=("auto", "dataset", "camera"),
        default="auto",
        help="auto uses cameras for one source and datasets for multi-source",
    )
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Configuration overrides",
    )
    return parser.parse_args()


def _dataset_options(configuration):
    datasets = configuration.DATASETS
    return {
        "root": datasets.ROOT_DIR,
        "split_id": int(getattr(datasets, "SPLIT_ID", 0)),
        "cuhk03_labeled": bool(
            getattr(datasets, "CUHK03_LABELED", False)
        ),
        "cuhk03_classic_split": bool(
            getattr(datasets, "CUHK03_CLASSIC_SPLIT", False)
        ),
        "market1501_500k": bool(
            getattr(datasets, "MARKET1501_500K", False)
        ),
    }


def _bind_authoritative_cameras(store, configuration, sources):
    from data.datasets import init_image_dataset

    options = _dataset_options(configuration)
    for source in sources:
        dataset = init_image_dataset(
            source,
            mode="train",
            combineall=bool(configuration.DATASETS.COMBINEALL),
            **options,
        )
        store = store.bind_camera_metadata(
            dataset.train, source, missing_policy="error"
        )
    return store


def _expected_domains(store, sources, domain_mode):
    if domain_mode == "dataset":
        return tuple(sorted(sources))
    return tuple(sorted({
        f"{record.dataset}:camera:{record.camera_id}"
        for record in store.records()
        if record.camera_id is not None
    }))


def main():
    args = parse_args()
    stability_seeds = args.stability_seeds or [2345, 3456]
    from config import cfg
    from data.attribute_store import AttributeStore
    from objectives.text_encoders import SigLIP2TextEncoder
    from semantic import (
        build_adaptive_codebook,
        build_phrase_bank,
        save_torch_artifact,
        sha256_file,
        validate_codebook_artifact,
    )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    raw_sources = cfg.DATASETS.SOURCES
    if isinstance(raw_sources, str):
        raw_sources = raw_sources.split(",")
    sources = tuple(
        str(source).strip().lower()
        for source in raw_sources
        if str(source).strip()
    )
    if not sources:
        raise ValueError("DATASETS.SOURCES must not be empty")
    domain_mode = args.domain_mode
    if domain_mode == "auto":
        domain_mode = "camera" if len(sources) == 1 else "dataset"

    caption_file = args.caption_file or cfg.OBJECTIVE.CAPTION.FILE
    if not caption_file:
        raise ValueError(
            "Provide --caption-file or OBJECTIVE.CAPTION.FILE"
        )
    allowed_splits = (
        ("train", "val", "query", "gallery")
        if cfg.DATASETS.COMBINEALL
        else ("train",)
    )
    store = AttributeStore.from_jsonl(
        caption_file,
        allowed_splits=allowed_splits,
        allowed_datasets=sources,
    )
    if domain_mode == "camera":
        store = _bind_authoritative_cameras(store, cfg, sources)
    expected_domains = _expected_domains(store, sources, domain_mode)

    text_encoder = SigLIP2TextEncoder(
        model_name=cfg.MODEL.BACKBONE.PRETRAINED_NAME,
        trainable=False,
    )
    phrase_bank = build_phrase_bank(
        store,
        text_encoder,
        batch_size=args.batch_size,
        device=args.device,
        domain_mode=domain_mode,
    )
    codebook = build_adaptive_codebook(
        phrase_bank,
        expected_domains=expected_domains,
        seed=args.seed,
        stability_seeds=stability_seeds,
        max_codes=args.max_codes,
        merge_similarity=args.merge_similarity,
        max_compactness_drop=args.max_compactness_drop,
        min_support=args.min_support,
        min_compactness=args.min_compactness,
        min_domain_coverage=args.min_domain_coverage,
    )
    validation = validate_codebook_artifact(codebook, phrase_bank)
    if not validation["valid"]:
        raise RuntimeError(
            "Generated codebook failed validation: "
            + "; ".join(validation["errors"])
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_path = output_dir / "phrase_bank.pt"
    codebook_path = output_dir / "codebook.pt"
    save_torch_artifact(phrase_bank, phrase_path)
    save_torch_artifact(codebook, codebook_path)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_datasets": list(sources),
        "allowed_splits": list(allowed_splits),
        "domain_mode": domain_mode,
        "expected_domains": list(expected_domains),
        "domain_evidence": (
            "authoritative_dataset_parser_camera_ids"
            if domain_mode == "camera"
            else "dataset_ids"
        ),
        "caption_file": str(Path(caption_file).resolve()),
        "caption_sha256": sha256_file(caption_file),
        "text_model": str(cfg.MODEL.BACKBONE.PRETRAINED_NAME),
        "config_file": str(Path(args.config_file).resolve()),
        "parameters": {
            "seed": args.seed,
            "stability_seeds": stability_seeds,
            "max_codes": args.max_codes,
            "merge_similarity": args.merge_similarity,
            "max_compactness_drop": args.max_compactness_drop,
            "min_support": args.min_support,
            "min_compactness": args.min_compactness,
            "min_domain_coverage": args.min_domain_coverage,
        },
        "attribute_summary": store.summary(),
        "codebook_summary": validation["summary"],
        "artifacts": {
            "phrase_bank": {
                "path": phrase_path.name,
                "sha256": sha256_file(phrase_path),
            },
            "codebook": {
                "path": codebook_path.name,
                "sha256": sha256_file(codebook_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
