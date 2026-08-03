"""源域训练集图片遍历模块（caption 生成管线 - 数据层）。

职责：
    为 5 个源域 ReID 数据集（market1501 / msmt17 / cuhk03 / cuhksysu / cuhk02）
    提供统一且协议感知的图片遍历接口，产出 ImageRecord 迭代器。

训练侧契约（来自 TransReID/ARCHITECTURE.md）：
    - image_path 必须是"相对数据集目录"的路径，如 bounding_box_train/0002_c1s1_000451_03.jpg；
    - Protocol-2 只遍历源域 train；Protocol-3 可遍历完整源域；
    - 目标域 query/gallery 永远不绑定 Caption，推理保持 image-only。

兼容性说明：
    各数据集在磁盘上的目录布局存在历史变体（如 MSMT17 V1/V2、CUHK03 detected/labeled），
    每个数据集函数内部做目录探测，并在找不到任何候选目录时抛出带提示的 FileNotFoundError。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# 当前 Caption 生成任务覆盖这 5 个源域数据集的协议所需 split。
SUPPORTED_DATASETS: list[str] = ["market1501", "msmt17", "cuhk03", "cuhksysu", "cuhk02"]

# 图片扩展名白名单（大小写不敏感）
_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_SCOPES: tuple[str, ...] = ("train", "trainval", "full", "all")


@dataclass(frozen=True)
class ImageRecord:
    """单张协议图片的元信息。

    Attributes:
        dataset: 数据集名（如 "market1501"）。
        split: 官方 split 名称（train/val/query/gallery）。
        image_path: 相对数据集目录的路径，如 "bounding_box_train/0002_c1s1_000451_03.jpg"。
        abs_path: 图片在磁盘上的绝对路径（供读取/送推理使用）。
        pid: 行人 ID（整数）。
    """

    dataset: str
    split: str
    image_path: str
    abs_path: str
    pid: int


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------

def _iter_images_in_dir(directory: Path) -> Iterator[Path]:
    """按文件名排序遍历目录下的图片文件（排序保证可复现的顺序）。"""
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p


def _pick_existing_dir(candidates: list[Path], dataset: str) -> Path:
    """从候选目录列表中选出第一个存在的目录。

    全部不存在时抛出带候选清单的 FileNotFoundError，便于快速定位挂载路径问题。
    """
    for c in candidates:
        if c.is_dir():
            return c
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"[{dataset}] 未找到训练图片目录，已尝试以下候选路径：\n  - {tried}\n"
        f"请检查数据集是否已下载并解压到 data_root 下。"
    )


# ---------------------------------------------------------------------------
# market1501
# ---------------------------------------------------------------------------

def _iter_market1501(data_root: str) -> Iterator[ImageRecord]:
    """Market-1501：bounding_box_train/*.jpg，文件名形如 0002_c1s1_000451_03.jpg。

    pid = 文件名第一个 '_' 前的整数；以 0000（垃圾图）和 -1（干扰图）开头的跳过。
    """
    root = Path(data_root)
    train_dir = _pick_existing_dir(
        [
            root / "market" / "bounding_box_train",
            root / "Market-1501-v15.09.15" / "bounding_box_train",
        ],
        "market1501",
    )
    for img in _iter_images_in_dir(train_dir):
        stem = img.stem
        first_token = stem.split("_")[0]
        if first_token in ("0000", "-1"):
            continue  # 跳过垃圾图与干扰图
        pid = int(first_token)
        yield ImageRecord(
            dataset="market1501",
            split="train",
            image_path=f"{train_dir.name}/{img.name}",
            abs_path=str(img.resolve()),
            pid=pid,
        )


def _iter_market1501_splits(
    data_root: str,
    split_names: tuple[str, ...],
) -> Iterator[ImageRecord]:
    """Enumerate the Market-1501 splits used by DG full-source training.

    ``bounding_box_test`` contains both ``-1`` junk detections and PID ``0``
    background distractors.  The training-side ``Dataset.combine_all`` drops
    both categories, so caption generation applies the same rule and does not
    spend GPU time on images that can never enter Protocol-3 training.
    """

    root = Path(data_root)
    train_dir = _pick_existing_dir(
        [
            root / "market" / "bounding_box_train",
            root / "Market-1501-v15.09.15" / "bounding_box_train",
        ],
        "market1501",
    )
    dataset_dir = train_dir.parent
    split_dirs = {
        "train": dataset_dir / "bounding_box_train",
        "query": dataset_dir / "query",
        "gallery": dataset_dir / "bounding_box_test",
    }
    seen_paths: set[str] = set()
    for split in split_names:
        directory = split_dirs.get(split)
        if directory is None:
            raise ValueError(f"[market1501] unsupported split: {split!r}")
        if not directory.is_dir():
            raise FileNotFoundError(
                f"[market1501] missing official {split} directory: {directory}"
            )
        for image in _iter_images_in_dir(directory):
            first_token = image.stem.split("_")[0]
            if first_token in ("0000", "-1"):
                continue
            image_path = f"{directory.name}/{image.name}"
            if image_path in seen_paths:
                raise ValueError(
                    f"[market1501] duplicate image across splits: {image_path}"
                )
            seen_paths.add(image_path)
            yield ImageRecord(
                dataset="market1501",
                split=split,
                image_path=image_path,
                abs_path=str(image.resolve()),
                pid=int(first_token),
            )


# ---------------------------------------------------------------------------
# msmt17
# ---------------------------------------------------------------------------

def _iter_msmt17(data_root: str) -> Iterator[ImageRecord]:
    """MSMT17：按官方 ``list_train.txt`` 枚举标准 train split。

    V1 的图片位于 ``train/<pid>/<image>``，V2 位于 ``mask_train_v2/``；
    两个版本都由版本目录下的 ``list_train.txt`` 给出图片相对路径和 pid。
    读取 split 文件而不是递归扫描目录，可以避免把 ``list_val.txt`` 中的
    2,373 张验证图片混入标准训练集。
    """
    root = Path(data_root)
    train_dir = _pick_existing_dir(
        [
            root / "msmt17" / "MSMT17_V2" / "mask_train_v2",
            root / "msmt17" / "mask_train_v2",
            root / "msmt17" / "MSMT17_V1" / "train",
            root / "msmt17" / "train",
        ],
        "msmt17",
    )
    dataset_dir = root / "msmt17"
    list_train_path = train_dir.parent / "list_train.txt"
    if not list_train_path.is_file():
        raise FileNotFoundError(
            f"[msmt17] 未找到官方 train split 文件：{list_train_path}"
        )

    train_root = train_dir.resolve()
    dataset_root = dataset_dir.resolve()
    seen_paths: set[str] = set()
    with list_train_path.open("r", encoding="utf-8") as split_file:
        for line_no, raw_line in enumerate(split_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"[msmt17] {list_train_path}:{line_no} 格式错误，"
                    "应为 '<相对图片路径> <pid>'"
                )
            relative_image, pid_text = fields
            try:
                pid = int(pid_text)
            except ValueError as exc:
                raise ValueError(
                    f"[msmt17] {list_train_path}:{line_no} pid 不是整数："
                    f"{pid_text!r}"
                ) from exc

            img = (train_dir / relative_image).resolve()
            try:
                img.relative_to(train_root)
            except ValueError as exc:
                raise ValueError(
                    f"[msmt17] {list_train_path}:{line_no} 图片路径越界："
                    f"{relative_image!r}"
                ) from exc
            if img.suffix.lower() not in _IMAGE_EXTS:
                raise ValueError(
                    f"[msmt17] {list_train_path}:{line_no} 不是支持的图片格式："
                    f"{relative_image!r}"
                )
            if not img.is_file():
                raise FileNotFoundError(
                    f"[msmt17] {list_train_path}:{line_no} 图片不存在：{img}"
                )

            image_path = img.relative_to(dataset_root).as_posix()
            if image_path in seen_paths:
                raise ValueError(
                    f"[msmt17] {list_train_path}:{line_no} 图片重复："
                    f"{relative_image!r}"
                )
            seen_paths.add(image_path)

            yield ImageRecord(
                dataset="msmt17",
                split="train",
                image_path=image_path,
                abs_path=str(img),
                pid=pid,
            )


def _iter_msmt17_manifest(
    data_root: str,
    split_names: tuple[str, ...],
) -> Iterator[ImageRecord]:
    """Enumerate MSMT17 strictly from its official split manifests.

    ``train`` and ``val`` entries live below the version's train directory;
    ``query`` and ``gallery`` entries live below its test directory. Files that
    happen to exist on disk but are absent from all manifests are intentionally
    excluded.
    """
    root = Path(data_root)
    version_candidates = (
        ("MSMT17_V2", "mask_train_v2", "mask_test_v2"),
        ("MSMT17_V1", "train", "test"),
    )
    version_dir: Path | None = None
    train_dir: Path | None = None
    test_dir: Path | None = None
    for version_name, train_name, test_name in version_candidates:
        candidate = root / "msmt17" / version_name
        if (candidate / train_name).is_dir() and (candidate / test_name).is_dir():
            version_dir = candidate
            train_dir = candidate / train_name
            test_dir = candidate / test_name
            break
    if version_dir is None or train_dir is None or test_dir is None:
        raise FileNotFoundError(
            "[msmt17] 未找到完整的 MSMT17_V1 或 MSMT17_V2 目录。"
        )

    dataset_root = (root / "msmt17").resolve()
    seen_paths: set[str] = set()
    for split in split_names:
        split_file = version_dir / f"list_{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"[msmt17] 未找到官方 {split} split 文件：{split_file}"
            )
        image_dir = train_dir if split in ("train", "val") else test_dir
        image_root = image_dir.resolve()
        with split_file.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(
                        f"[msmt17] {split_file}:{line_no} 格式错误，"
                        "应为 '<相对图片路径> <pid>'"
                    )
                relative_image, pid_text = fields
                try:
                    pid = int(pid_text)
                except ValueError as exc:
                    raise ValueError(
                        f"[msmt17] {split_file}:{line_no} pid 不是整数："
                        f"{pid_text!r}"
                    ) from exc

                img = (image_dir / relative_image).resolve()
                try:
                    img.relative_to(image_root)
                except ValueError as exc:
                    raise ValueError(
                        f"[msmt17] {split_file}:{line_no} 图片路径越界："
                        f"{relative_image!r}"
                    ) from exc
                if img.suffix.lower() not in _IMAGE_EXTS:
                    raise ValueError(
                        f"[msmt17] {split_file}:{line_no} 不是支持的图片格式："
                        f"{relative_image!r}"
                    )
                if not img.is_file():
                    raise FileNotFoundError(
                        f"[msmt17] {split_file}:{line_no} 图片不存在：{img}"
                    )
                image_path = img.relative_to(dataset_root).as_posix()
                if image_path in seen_paths:
                    raise ValueError(
                        f"[msmt17] {split_file}:{line_no} 图片跨 split 重复："
                        f"{relative_image!r}"
                    )
                seen_paths.add(image_path)
                yield ImageRecord(
                    dataset="msmt17",
                    split=split,
                    image_path=image_path,
                    abs_path=str(img),
                    pid=pid,
                )


# ---------------------------------------------------------------------------
# cuhk03
# ---------------------------------------------------------------------------

def _iter_cuhk03(data_root: str) -> Iterator[ImageRecord]:
    """CUHK03-NP new protocol, detected crop, official train split."""
    yield from _iter_cuhk03_splits(data_root, ("train",))


def _iter_cuhk03_splits(
    data_root: str,
    split_names: tuple[str, ...],
) -> Iterator[ImageRecord]:
    """Enumerate CUHK03 new-protocol detected images from its split JSON.

    The JSON generated by the historical loader may contain stale absolute
    prefixes. Only each path's basename is trusted and resolved under the
    current ``images_detected`` directory.
    """
    root = Path(data_root)
    dataset_dir = root / "cuhk03"
    images_dir = dataset_dir / "images_detected"
    split_file = dataset_dir / "splits_new_detected.json"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"[cuhk03] 未找到 detected 图片目录：{images_dir}"
        )
    if not split_file.is_file():
        raise FileNotFoundError(
            f"[cuhk03] 未找到 new detected split 文件：{split_file}"
        )

    with split_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"[cuhk03] split 文件格式错误：{split_file}")
    split = payload[0]
    seen_paths: set[str] = set()
    for split_name in split_names:
        entries = split.get(split_name)
        if not isinstance(entries, list):
            raise ValueError(
                f"[cuhk03] {split_file} 缺少列表字段：{split_name}"
            )
        for entry_no, entry in enumerate(entries, start=1):
            if not isinstance(entry, list) or len(entry) < 2:
                raise ValueError(
                    f"[cuhk03] {split_file}:{split_name}[{entry_no}] 格式错误"
                )
            source_path, pid_raw = entry[0], entry[1]
            filename = Path(str(source_path)).name
            img = (images_dir / filename).resolve()
            if not img.is_file():
                raise FileNotFoundError(
                    f"[cuhk03] {split_name}[{entry_no}] 图片不存在：{img}"
                )
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[cuhk03] {split_name}[{entry_no}] pid 非整数：{pid_raw!r}"
                ) from exc
            image_path = img.relative_to(dataset_dir.resolve()).as_posix()
            if image_path in seen_paths:
                raise ValueError(
                    f"[cuhk03] 图片跨 split 重复：{image_path}"
                )
            seen_paths.add(image_path)
            yield ImageRecord(
                dataset="cuhk03",
                split=split_name,
                image_path=image_path,
                abs_path=str(img),
                pid=pid,
            )


# ---------------------------------------------------------------------------
# cuhksysu
# ---------------------------------------------------------------------------

# CUHKSYSU 文件名形如 p11422_s16929_1.jpg，pid 取 'p' 后的数字
_CUHKSYSU_PID_RE = re.compile(r"^p(\d+)", re.IGNORECASE)


def _iter_cuhksysu(data_root: str) -> Iterator[ImageRecord]:
    """CUHK-SYSU：与项目训练侧 loader（data/datasets/image/cuhksysu.py）对齐——
    cropped_images/ 下全部图片直接作为训练数据（train-only，无 train/test 划分，
    Protocol-1/2 中 CUHK-SYSU 只作源域不作目标域）。

    文件名形如 p11422_s16929_1.jpg，pid 取 'p' 后的数字。
    文件名无法解析 pid 的图片会打印警告并跳过（防御性处理）。
    """
    root = Path(data_root)
    train_dir = _pick_existing_dir(
        [
            root / "cuhksysu" / "cropped_images",
            root / "cuhksysu" / "cropped_image",
        ],
        "cuhksysu",
    )
    skipped = 0
    for img in _iter_images_in_dir(train_dir):
        m = _CUHKSYSU_PID_RE.match(img.stem)
        if m is None:
            skipped += 1
            print(f"[cuhksysu] 警告：无法从文件名解析 pid，已跳过：{img.name}")
            continue
        pid = int(m.group(1))
        image_path = f"{train_dir.name}/{img.name}"
        yield ImageRecord(
            dataset="cuhksysu",
            split="train",
            image_path=image_path,
            abs_path=str(img.resolve()),
            pid=pid,
        )
    if skipped:
        print(f"[cuhksysu] 共跳过 {skipped} 张无法解析 pid 的图片。")


# ---------------------------------------------------------------------------
# cuhk02
# ---------------------------------------------------------------------------

def _iter_cuhk02(data_root: str) -> Iterator[ImageRecord]:
    """CUHK02：与项目训练侧 loader（data/datasets/image/cuhk02.py）对齐——
    P1-P4 作训练集（P5 为测试集，不生成 caption）。

    目录布局（两种都兼容）：
        cuhk02/Dataset/P1/cam1/*.png   （官方发布结构，项目 loader 使用）
        cuhk02/P1/cam1/*.png           （MetaBIN 风格扁平结构）
    文件名形如 001_1.png，pid 取第一个 '_' 前的部分；
    pid 按 pair 内排序去重后重新编号并逐 pair 累加偏移（与训练侧 loader 一致）。
    image_path 统一为 'P1/cam1/001_1.png'（相对 Dataset/ 层）。
    """
    root = Path(data_root)
    base = root / "cuhk02"
    if (base / "Dataset").is_dir():
        base = base / "Dataset"

    train_pairs = ["P1", "P2", "P3", "P4"]  # P5 为测试集，见训练侧 loader
    found_any = False
    pid_offset = 0
    for pair in train_pairs:
        pair_dir = base / pair
        if not pair_dir.is_dir():
            continue
        found_any = True
        # 先收集本 pair 的原始 pid 集合，重新编号（对齐 loader 的 relabel 逻辑）
        raw_records: list[tuple[Path, str, str]] = []  # (img, raw_pid, rel_dir)
        for cam in ("cam1", "cam2"):
            cam_dir = pair_dir / cam
            if not cam_dir.is_dir():
                continue
            for img in _iter_images_in_dir(cam_dir):
                raw_pid = img.stem.split("_")[0]
                raw_records.append((img, raw_pid, f"{pair}/{cam}"))
        unique_pids = sorted({r[1] for r in raw_records})
        pid2label = {p: i + pid_offset for i, p in enumerate(unique_pids)}
        pid_offset += len(unique_pids)
        for img, raw_pid, rel_dir in raw_records:
            yield ImageRecord(
                dataset="cuhk02",
                split="train",
                image_path=f"{rel_dir}/{img.name}",
                abs_path=str(img.resolve()),
                pid=pid2label[raw_pid],
            )
    if not found_any:
        raise FileNotFoundError(
            f"[cuhk02] 未找到 P1-P4 目录。已尝试：{base}/P1..P4"
            f"（兼容 {base.parent}/Dataset/P1..P4）。请检查数据集目录结构。"
        )


# ---------------------------------------------------------------------------
# 注册表 + 统一入口
# ---------------------------------------------------------------------------

# 数据集名 -> 遍历函数 的注册表；新增数据集时在此登记即可
_DATASET_REGISTRY: dict[str, Callable[[str], Iterator[ImageRecord]]] = {
    "market1501": _iter_market1501,
    "msmt17": _iter_msmt17,
    "cuhk03": _iter_cuhk03,
    "cuhksysu": _iter_cuhksysu,
    "cuhk02": _iter_cuhk02,
}


def iter_source_train_images(dataset: str, data_root: str) -> Iterator[ImageRecord]:
    """遍历指定源域数据集的 train split 图片。

    Args:
        dataset: 数据集名，必须在 SUPPORTED_DATASETS 中。
        data_root: 数据根目录（各数据集目录的共同父目录）。

    Yields:
        ImageRecord，按文件名排序，顺序可复现。

    Raises:
        ValueError: 数据集名不受支持时抛出，并列出支持项。
        FileNotFoundError: 数据集目录不存在时抛出（带候选路径提示）。
    """
    key = dataset.strip().lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(
            f"不支持的数据集名：{dataset!r}。"
            f"当前支持：{', '.join(SUPPORTED_DATASETS)}"
        )
    yield from _DATASET_REGISTRY[key](data_root)


def iter_caption_images(
    dataset: str,
    data_root: str,
    split_scope: str = "train",
) -> Iterator[ImageRecord]:
    """Enumerate protocol-aware images for caption generation.

    Scopes:
      - ``train``: official training split only.
      - ``trainval``: train plus validation when the dataset defines one.
      - ``full``: DG full-source view (train + query + gallery).
      - ``all``: every official manifest split, including validation.

    Existing callers should keep using :func:`iter_source_train_images` when
    they require the strict historical train-only contract.
    """
    key = dataset.strip().lower()
    scope = split_scope.strip().lower()
    if scope not in SPLIT_SCOPES:
        raise ValueError(
            f"不支持的 split_scope：{split_scope!r}；"
            f"可选值：{', '.join(SPLIT_SCOPES)}"
        )
    if scope == "train":
        yield from iter_source_train_images(key, data_root)
        return
    if key == "market1501":
        names = (
            ("train",)
            if scope == "trainval"
            else ("train", "query", "gallery")
        )
        yield from _iter_market1501_splits(data_root, names)
        return
    if key == "cuhk03":
        names = ("train", "query", "gallery")
        yield from _iter_cuhk03_splits(data_root, names)
        return
    if key == "msmt17":
        if scope == "trainval":
            names = ("train", "val")
        elif scope == "full":
            names = ("train", "query", "gallery")
        else:
            names = ("train", "val", "query", "gallery")
        yield from _iter_msmt17_manifest(data_root, names)
        return
    raise ValueError(
        f"[{key}] 当前尚未实现 split_scope={scope!r}；请使用 train。"
    )
