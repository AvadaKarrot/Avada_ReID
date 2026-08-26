"""Offline Attribute-codebook targets attached only to source records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from data.attribute_store import ATTRIBUTE_BUCKETS, AttributeStore
from semantic.artifact import load_torch_artifact, sha256_file


@dataclass(frozen=True)
class AttributeSemanticTarget:
    distributions: dict
    valid: dict
    quality: float


class AttributeSemanticTargetStore:
    """Map structured source Attributes to frozen soft code assignments."""

    def __init__(self, targets, code_sizes):
        self._targets = dict(targets)
        self.code_sizes = dict(code_sizes)

    @classmethod
    def from_artifacts(
        cls,
        caption_file,
        phrase_bank_file,
        codebook_file,
        *,
        source_datasets,
        text_temperature=0.07,
        manifest_file="",
    ):
        if text_temperature <= 0:
            raise ValueError("text_temperature must be positive")
        sources = tuple(str(value).strip().lower() for value in source_datasets)
        if manifest_file:
            with open(manifest_file, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest_sources = {
                str(value).strip().lower()
                for value in manifest.get("source_datasets", ())
            }
            if manifest_sources != set(sources):
                raise ValueError(
                    "Manifest source datasets do not match configured sources"
                )
            root = Path(manifest_file).parent
            expected_files = {
                "phrase_bank": Path(phrase_bank_file),
                "codebook": Path(codebook_file),
            }
            for name, actual_path in expected_files.items():
                entry = manifest.get("artifacts", {}).get(name, {})
                declared_path = root / entry.get("path", "")
                if declared_path.resolve() != actual_path.resolve():
                    raise ValueError(f"Manifest {name} path does not match config")
                if entry.get("sha256") != sha256_file(actual_path):
                    raise ValueError(f"Manifest {name} SHA256 verification failed")
        attributes = AttributeStore.from_jsonl(
            caption_file,
            allowed_splits=("train",),
            allowed_datasets=sources,
        )
        phrase_bank = load_torch_artifact(phrase_bank_file)
        codebook = load_torch_artifact(codebook_file)
        artifact_datasets = {
            str(domain).split(":camera:", 1)[0].strip().lower()
            for domain in codebook.get("expected_domains", ())
        }
        unexpected = artifact_datasets.difference(sources)
        absent = set(sources).difference(artifact_datasets)
        if unexpected or absent:
            raise ValueError(
                "Codebook source domains do not match configured sources: "
                f"artifact={sorted(artifact_datasets)}, "
                f"configured={sorted(sources)}"
            )
        phrase_lookup = {}
        code_sizes = {}
        for bucket in ATTRIBUTE_BUCKETS:
            phrase_payload = phrase_bank["buckets"][bucket]
            code_payload = codebook["buckets"][bucket]
            embeddings = F.normalize(
                phrase_payload["embeddings"].float(), dim=1
            )
            prototypes = F.normalize(
                code_payload["prototypes"].float(), dim=1
            )
            distributions = F.softmax(
                embeddings @ prototypes.t() / float(text_temperature), dim=1
            )
            phrase_lookup[bucket] = {
                phrase: distributions[index]
                for index, phrase in enumerate(phrase_payload["phrases"])
            }
            code_sizes[bucket] = int(prototypes.shape[0])

        targets = {}
        for record in attributes.records():
            distributions = {}
            valid = {}
            for bucket in ATTRIBUTE_BUCKETS:
                values = [
                    phrase_lookup[bucket][phrase]
                    for phrase in record.phrases(bucket)
                    if phrase in phrase_lookup[bucket]
                ]
                valid[bucket] = bool(values)
                distributions[bucket] = (
                    torch.stack(values).mean(dim=0)
                    if values
                    else torch.zeros(code_sizes[bucket], dtype=torch.float32)
                )
            targets[(record.dataset, record.image_path)] = AttributeSemanticTarget(
                distributions=distributions,
                valid=valid,
                quality=record.quality_score,
            )
        return cls(targets, code_sizes)

    def get(self, image_path, dataset):
        dataset = str(dataset).strip().lower()
        for candidate in AttributeStore._path_candidates(image_path):
            target = self._targets.get((dataset, candidate))
            if target is not None:
                return target
        return None

    def _masked_target(self):
        return AttributeSemanticTarget(
            distributions={
                bucket: torch.zeros(size, dtype=torch.float32)
                for bucket, size in self.code_sizes.items()
            },
            valid={bucket: False for bucket in ATTRIBUTE_BUCKETS},
            quality=0.0,
        )

    def bind(self, records, dataset, *, missing_policy="error"):
        if missing_policy not in {"error", "mask"}:
            raise ValueError("missing_policy must be 'error' or 'mask'")
        bound = []
        missing = []
        for item in records:
            target = self.get(item[0], dataset)
            if target is None:
                missing.append(str(item[0]))
                target = self._masked_target()
            bound.append(tuple(item) + (target,))
        if missing and missing_policy == "error":
            preview = "\n".join(f"  - {path}" for path in missing[:5])
            raise ValueError(
                f"Attribute target coverage failed for {dataset!r}: "
                f"{len(missing)}/{len(records)} records missing.\n{preview}"
            )
        return bound


def collate_attribute_targets(targets):
    return {
        "distributions": {
            bucket: torch.stack(
                [target.distributions[bucket] for target in targets]
            )
            for bucket in ATTRIBUTE_BUCKETS
        },
        "mask": {
            bucket: torch.tensor(
                [target.valid[bucket] for target in targets], dtype=torch.bool
            )
            for bucket in ATTRIBUTE_BUCKETS
        },
        "quality": torch.tensor(
            [target.quality for target in targets], dtype=torch.float32
        ),
    }
