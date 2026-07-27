import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


class CaptionStore:
    """Source-training caption index backed by JSONL.

    Every line must contain ``image_path`` and ``captions``. Target-domain
    datasets should never construct this object.
    """

    def __init__(self, captions: Dict[str, Tuple[str, ...]]):
        self._captions = captions

    @classmethod
    def from_jsonl(cls, path) -> "CaptionStore":
        path = Path(path)
        captions = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") not in (None, "train"):
                    raise ValueError(
                        f"{path}:{line_number} contains a non-training caption"
                    )
                image_path = str(record["image_path"])
                values = tuple(
                    text.strip()
                    for text in record.get("captions", ())
                    if text and text.strip()
                )
                captions[image_path] = values
        return cls(captions)

    def get(self, image_path: str) -> Tuple[str, ...]:
        return self._captions.get(str(image_path), ())

    def __contains__(self, image_path: str) -> bool:
        return str(image_path) in self._captions

    def __len__(self) -> int:
        return len(self._captions)

    def keys(self) -> Iterable[str]:
        return self._captions.keys()
