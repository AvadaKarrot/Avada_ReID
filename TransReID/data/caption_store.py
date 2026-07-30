import json
from pathlib import PurePosixPath
from typing import Dict, Iterable, Sequence, Tuple


def _normalize_path(path) -> str:
    value = str(path).replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return str(PurePosixPath(value))


class CaptionStore:
    """Read-only caption sidecar used only for source-domain training.

    Caption files may contain every official split. Only ``train`` records are
    indexed, so query/gallery text can never enter the ReID evaluation path.
    Keys are scoped by dataset name to prevent collisions in multi-source
    training.
    """

    def __init__(
        self,
        captions: Dict[Tuple[str, str], Tuple[str, ...]],
    ):
        self._captions = captions

    @classmethod
    def from_jsonl(cls, path) -> "CaptionStore":
        captions = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc

                if str(record.get("split", "train")).lower() != "train":
                    continue

                dataset = str(record.get("dataset", "")).strip().lower()
                if not dataset:
                    raise ValueError(
                        f"{path}:{line_number} is missing dataset"
                    )
                image_path = _normalize_path(record["image_path"])
                values = tuple(
                    text.strip()
                    for text in record.get("captions", ())
                    if isinstance(text, str) and text.strip()
                )
                key = (dataset, image_path)
                if key in captions:
                    raise ValueError(
                        f"{path}:{line_number} duplicates {dataset}/{image_path}"
                    )
                captions[key] = values
        return cls(captions)

    @staticmethod
    def _path_candidates(image_path: str) -> Iterable[str]:
        normalized = _normalize_path(image_path)
        parts = PurePosixPath(normalized).parts
        yield normalized
        for index in range(1, len(parts)):
            yield str(PurePosixPath(*parts[index:]))

    def get(
        self,
        image_path: str,
        dataset: str,
    ) -> Tuple[str, ...]:
        dataset = str(dataset).strip().lower()
        for candidate in self._path_candidates(image_path):
            captions = self._captions.get((dataset, candidate))
            if captions is not None:
                return captions
        return ()

    def bind(
        self,
        records: Sequence[tuple],
        dataset: str,
        missing_policy: str = "error",
    ) -> list:
        """Attach caption tuples without changing the image dataset parser.

        ``missing_policy='error'`` protects experiment integrity. ``mask``
        keeps uncovered images and lets the objective ignore their empty text.
        """

        if missing_policy not in {"error", "mask"}:
            raise ValueError(
                "caption missing_policy must be 'error' or 'mask'"
            )

        bound = []
        missing = []
        for record in records:
            if len(record) != 4:
                raise ValueError(
                    "CaptionStore.bind expects "
                    "(path, pid, camid, dataset_id) records"
                )
            image_path, pid, camid, dataset_id = record
            values = self.get(image_path, dataset)
            if not values:
                missing.append(str(image_path))
            bound.append(
                (image_path, pid, camid, dataset_id, values)
            )

        if missing and missing_policy == "error":
            preview = "\n".join(f"  - {path}" for path in missing[:5])
            raise ValueError(
                f"Caption coverage failed for dataset {dataset!r}: "
                f"{len(missing)}/{len(records)} source-train images are missing. "
                f"First missing paths:\n{preview}"
            )
        return bound

    def __contains__(self, key) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        dataset, image_path = key
        return bool(self.get(image_path, dataset))

    def __len__(self) -> int:
        return len(self._captions)

    def keys(self) -> Iterable[Tuple[str, str]]:
        return self._captions.keys()
