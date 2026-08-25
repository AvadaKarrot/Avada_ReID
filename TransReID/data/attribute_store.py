"""Source-only structured Attribute records from the Caption JSONL asset."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence, Tuple


ATTRIBUTE_BUCKETS = (
    "upper_clothing",
    "lower_clothing",
    "footwear",
    "carried_items",
    "accessories",
    "hair",
    "distinctive_features",
)

VISIBILITY_BUCKETS = frozenset(
    {"upper_clothing", "lower_clothing", "footwear"}
)

INVALID_VISIBILITY = frozenset(
    {
        "",
        "not visible",
        "not_visible",
        "unknown",
        "cannot determine",
        "cannot_determine",
        "unclear",
    }
)

_WHITESPACE = re.compile(r"\s+")


def normalize_attribute_phrase(value: str) -> str:
    """Return the canonical lookup spelling without rewriting semantics."""

    return _WHITESPACE.sub(" ", str(value).strip().lower())


def _normalize_path(path) -> str:
    value = str(path).replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return str(PurePosixPath(value))


def _normalize_visibility(value) -> str:
    return normalize_attribute_phrase(str(value).replace("-", " "))


def _clean_phrase_values(values) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        phrase = normalize_attribute_phrase(value)
        if phrase and phrase not in seen:
            seen.add(phrase)
            result.append(phrase)
    return tuple(result)


@dataclass(frozen=True)
class AttributeRecord:
    dataset: str
    split: str
    image_path: str
    pid: object
    quality_score: float
    attributes: Mapping[str, Tuple[str, ...]]
    visibility: Mapping[str, str]
    camera_id: object | None = None

    def phrases(self, bucket: str) -> Tuple[str, ...]:
        return self.attributes.get(bucket, ())

    def is_valid(self, bucket: str) -> bool:
        return bool(self.phrases(bucket))


class AttributeStore:
    """Read structured attributes once and apply visibility at load time.

    The original JSONL remains the source of truth.  Invalid or invisible
    attributes are represented by empty tuples, so callers do not need a
    second sidecar containing seven duplicated classifications.
    """

    def __init__(
        self,
        records: Mapping[Tuple[str, str], AttributeRecord],
    ):
        self._records = dict(records)

    @classmethod
    def from_jsonl(
        cls,
        path,
        *,
        allowed_splits: Iterable[str] = ("train",),
        allowed_datasets: Iterable[str] | None = None,
    ) -> "AttributeStore":
        splits = {
            str(split).strip().lower() for split in allowed_splits
        }
        if not splits:
            raise ValueError("allowed_splits must not be empty")
        datasets = None
        if allowed_datasets is not None:
            datasets = {
                str(dataset).strip().lower()
                for dataset in allowed_datasets
            }
            if not datasets:
                raise ValueError("allowed_datasets must not be empty")

        records = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc

                split = str(raw.get("split", "train")).strip().lower()
                if split not in splits:
                    continue
                dataset = str(raw.get("dataset", "")).strip().lower()
                if not dataset:
                    raise ValueError(
                        f"{path}:{line_number} is missing dataset"
                    )
                if datasets is not None and dataset not in datasets:
                    continue

                image_path = _normalize_path(raw["image_path"])
                quality = raw.get("quality_score", 1.0)
                try:
                    quality = float(quality)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number} has invalid quality_score"
                    ) from exc
                if not math.isfinite(quality):
                    raise ValueError(
                        f"{path}:{line_number} has non-finite quality_score"
                    )
                quality = min(1.0, max(0.0, quality))

                raw_attributes = raw.get("attributes") or {}
                attributes = {}
                visibility = {}
                for bucket in ATTRIBUTE_BUCKETS:
                    value = raw_attributes.get(bucket)
                    if bucket in VISIBILITY_BUCKETS:
                        if not isinstance(value, dict):
                            visibility[bucket] = ""
                            attributes[bucket] = ()
                            continue
                        state = _normalize_visibility(
                            value.get("visibility", "")
                        )
                        visibility[bucket] = state
                        if state in INVALID_VISIBILITY:
                            attributes[bucket] = ()
                        else:
                            attributes[bucket] = _clean_phrase_values(
                                value.get("attributes", ())
                            )
                    else:
                        visibility[bucket] = "present" if value else ""
                        attributes[bucket] = _clean_phrase_values(value)

                key = (dataset, image_path)
                if key in records:
                    raise ValueError(
                        f"{path}:{line_number} duplicates "
                        f"{dataset}/{image_path}"
                    )
                records[key] = AttributeRecord(
                    dataset=dataset,
                    split=split,
                    image_path=image_path,
                    pid=raw.get("pid"),
                    quality_score=quality,
                    attributes=MappingProxyType(attributes),
                    visibility=MappingProxyType(visibility),
                )
        return cls(records)

    @staticmethod
    def _path_candidates(image_path: str) -> Iterator[str]:
        normalized = _normalize_path(image_path)
        parts = PurePosixPath(normalized).parts
        yield normalized
        for index in range(1, len(parts)):
            yield str(PurePosixPath(*parts[index:]))

    def get(self, image_path: str, dataset: str) -> AttributeRecord | None:
        dataset = str(dataset).strip().lower()
        for candidate in self._path_candidates(image_path):
            record = self._records.get((dataset, candidate))
            if record is not None:
                return record
        return None

    def records(self) -> Iterator[AttributeRecord]:
        return iter(self._records.values())

    def bind_camera_metadata(
        self,
        records: Sequence[tuple],
        dataset: str,
        *,
        missing_policy: str = "error",
        restrict_to_records: bool = True,
    ) -> "AttributeStore":
        """Attach camera IDs supplied by the authoritative ReID parser.

        Caption JSON deliberately has no camera field.  Camera pseudo-domains
        must therefore come from dataset records rather than filename guesses.
        """

        if missing_policy not in {"error", "ignore"}:
            raise ValueError("missing_policy must be 'error' or 'ignore'")
        dataset = str(dataset).strip().lower()
        updated = dict(self._records)
        missing = []
        matched_keys = set()
        for item in records:
            if len(item) < 3:
                raise ValueError(
                    "camera binding expects (path, pid, camid, ...) records"
                )
            image_path, _, camera_id = item[:3]
            matched_key = None
            for candidate in self._path_candidates(image_path):
                key = (dataset, candidate)
                if key in updated:
                    matched_key = key
                    break
            if matched_key is None:
                missing.append(str(image_path))
                continue
            matched_keys.add(matched_key)
            updated[matched_key] = replace(
                updated[matched_key], camera_id=camera_id
            )
        if missing and missing_policy == "error":
            preview = "\n".join(f"  - {path}" for path in missing[:5])
            raise ValueError(
                f"Camera metadata coverage failed for {dataset!r}: "
                f"{len(missing)}/{len(records)} records are missing. "
                f"First missing paths:\n{preview}"
            )
        if restrict_to_records:
            updated = {
                key: value
                for key, value in updated.items()
                if key[0] != dataset or key in matched_keys
            }
        return type(self)(updated)

    def phrase_statistics(
        self,
        bucket: str,
        *,
        domain_mode: str = "dataset",
    ) -> dict[str, dict]:
        if bucket not in ATTRIBUTE_BUCKETS:
            raise KeyError(f"Unknown Attribute bucket {bucket!r}")
        if domain_mode not in {"dataset", "camera"}:
            raise ValueError("domain_mode must be 'dataset' or 'camera'")
        counts = Counter()
        quality_sum = defaultdict(float)
        domains = defaultdict(Counter)
        for record in self._records.values():
            if domain_mode == "camera":
                if record.camera_id is None and record.is_valid(bucket):
                    raise ValueError(
                        "camera domain mode requires camera metadata for "
                        f"{record.dataset}/{record.image_path}"
                    )
                domain = (
                    f"{record.dataset}:camera:{record.camera_id}"
                )
            else:
                domain = record.dataset
            for phrase in record.phrases(bucket):
                counts[phrase] += 1
                quality_sum[phrase] += record.quality_score
                domains[phrase][domain] += 1
        return {
            phrase: {
                "count": counts[phrase],
                "mean_quality": quality_sum[phrase] / counts[phrase],
                "domains": dict(sorted(domains[phrase].items())),
            }
            for phrase in sorted(counts)
        }

    def summary(self) -> dict:
        datasets = Counter(record.dataset for record in self._records.values())
        valid = {
            bucket: sum(
                record.is_valid(bucket) for record in self._records.values()
            )
            for bucket in ATTRIBUTE_BUCKETS
        }
        unique = {
            bucket: len(self.phrase_statistics(bucket))
            for bucket in ATTRIBUTE_BUCKETS
        }
        return {
            "records": len(self),
            "datasets": dict(sorted(datasets.items())),
            "valid_records_by_bucket": valid,
            "unique_phrases_by_bucket": unique,
        }

    def __len__(self) -> int:
        return len(self._records)

    def keys(self) -> Sequence[Tuple[str, str]]:
        return tuple(self._records)
