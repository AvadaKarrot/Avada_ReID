from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReIDSample:
    """Unambiguous sample metadata shared by single- and multi-source data."""

    image_path: str
    pid: int
    camid: int
    dataset_id: int
    captions: Tuple[str, ...] = ()

    @property
    def has_caption(self) -> bool:
        return bool(self.captions)


def image_only_sample(
    image_path: str, pid: int, camid: int, dataset_id: int
) -> ReIDSample:
    """Build a query/gallery sample that cannot carry target-domain text."""

    return ReIDSample(
        image_path=image_path,
        pid=pid,
        camid=camid,
        dataset_id=dataset_id,
        captions=(),
    )
