"""源域训练集图片遍历模块（caption 生成管线 - 数据层）。

职责：
    为 5 个源域 ReID 数据集（market1501 / msmt17 / cuhk03 / cuhksysu / cuhk02）
    提供统一的 train split 图片遍历接口，产出 ImageRecord 迭代器。

训练侧契约（来自 TransReID/ARCHITECTURE.md）：
    - image_path 必须是"相对数据集目录"的路径，如 bounding_box_train/0002_c1s1_000451_03.jpg；
    - 只允许 train split（目标域 query/gallery 的文本会被 CaptionStore 拒绝，防泄漏）；
    - 本模块只遍历源域 train 目录，天然满足该约束。

兼容性说明：
    各数据集在磁盘上的目录布局存在历史变体（如 MSMT17 V1/V2、CUHK03 detected/labeled），
    每个数据集函数内部做目录探测，并在找不到任何候选目录时抛出带提示的 FileNotFoundError。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# 当前 caption 生成任务仅覆盖这 5 个源域数据集的 train 部分
SUPPORTED_DATASETS: list[str] = ["market1501", "msmt17", "cuhk03", "cuhksysu", "cuhk02"]

# 图片扩展名白名单（大小写不敏感）
_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    """单张训练图片的元信息。

    Attributes:
        dataset: 数据集名（如 "market1501"）。
        split: 固定为 "train"（契约要求，禁止其他 split）。
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


# ---------------------------------------------------------------------------
# msmt17
# ---------------------------------------------------------------------------

def _iter_msmt17(data_root: str) -> Iterator[ImageRecord]:
    """MSMT17：优先 V2 的 mask_train_v2，兼容扁平布局与 V1 的 train/ 目录。

    文件名同样形如 0002_c1s1_000451_03.jpg；image_path 用实际目录名拼接，
    保证与磁盘真实结构一致（训练侧用 image_path 直接 join 数据集目录）。
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
    for img in _iter_images_in_dir(train_dir):
        first_token = img.stem.split("_")[0]
        if first_token in ("0000", "-1"):
            continue
        pid = int(first_token)
        yield ImageRecord(
            dataset="msmt17",
            split="train",
            image_path=f"{train_dir.name}/{img.name}",
            abs_path=str(img.resolve()),
            pid=pid,
        )


# ---------------------------------------------------------------------------
# cuhk03
# ---------------------------------------------------------------------------

def _iter_cuhk03(data_root: str) -> Iterator[ImageRecord]:
    """CUHK03-NP：detected 与 labeled 两个子集的 bounding_box_train 都遍历。

    两个子集在官方定义中互不重叠（各自独立的划分），因此统一遍历不会重复。
    image_path 分别带 detected/ 或 labeled/ 前缀。
    """
    root = Path(data_root)
    found_any = False
    for subset in ("detected", "labeled"):
        train_dir = root / "cuhk03-np" / subset / "bounding_box_train"
        if not train_dir.is_dir():
            continue  # 某个子集缺失时容忍，继续遍历另一个
        found_any = True
        for img in _iter_images_in_dir(train_dir):
            first_token = img.stem.split("_")[0]
            if first_token in ("0000", "-1"):
                continue
            pid = int(first_token)
            yield ImageRecord(
                dataset="cuhk03",
                split="train",
                image_path=f"{subset}/{train_dir.name}/{img.name}",
                abs_path=str(img.resolve()),
                pid=pid,
            )
    if not found_any:
        raise FileNotFoundError(
            f"[cuhk03] 未找到训练图片目录，已尝试：\n"
            f"  - {root / 'cuhk03-np' / 'detected' / 'bounding_box_train'}\n"
            f"  - {root / 'cuhk03-np' / 'labeled' / 'bounding_box_train'}\n"
            f"请确认 cuhk03-np 数据集已按 detected/labeled 结构放置。"
        )


# ---------------------------------------------------------------------------
# cuhksysu
# ---------------------------------------------------------------------------

# CUHKSYSU 文件名形如 p001_s1_1.png / p001_s1_1.jpg，pid 取 'p' 后的数字
_CUHKSYSU_PID_RE = re.compile(r"^p(\d+)", re.IGNORECASE)


def _iter_cuhksysu(data_root: str) -> Iterator[ImageRecord]:
    """CUHK-SYSU：训练图目录常见为 train/ 或 cropped_image/train/。

    文件名形如 p001_s1_1.png（或 .jpg），pid 取 'p' 后的数字。
    文件名无法解析 pid 的图片会打印警告并跳过（防御性处理）。
    """
    root = Path(data_root)
    train_dir = _pick_existing_dir(
        [
            root / "cuhksysu" / "train",
            root / "cuhksysu" / "cropped_image" / "train",
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
        # image_path 保持与磁盘结构一致：cropped_image/train/ 布局带两层前缀
        if train_dir.parent.name == "cropped_image":
            image_path = f"cropped_image/{train_dir.name}/{img.name}"
        else:
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
    """CUHK02：训练图目录常见为 cam1/ + cam2/（两个相机目录都遍历）或 Train/。

    文件名样式与 Market 系列一致（pid 为第一个 '_' 前的整数）。
    cam1/cam2 布局下 image_path 带相机目录前缀；Train/ 布局下直接拼接文件名。
    """
    root = Path(data_root)
    base = root / "cuhk02"
    cam_dirs = [base / "cam1", base / "cam2"]
    existing_cam_dirs = [d for d in cam_dirs if d.is_dir()]

    if existing_cam_dirs:
        # 双相机目录布局：两个目录都属于 train，合并遍历
        for cam_dir in existing_cam_dirs:
            for img in _iter_images_in_dir(cam_dir):
                first_token = img.stem.split("_")[0]
                if first_token in ("0000", "-1"):
                    continue
                pid = int(first_token)
                yield ImageRecord(
                    dataset="cuhk02",
                    split="train",
                    image_path=f"{cam_dir.name}/{img.name}",
                    abs_path=str(img.resolve()),
                    pid=pid,
                )
        return

    # 退化为单一 Train/ 目录布局
    train_dir = _pick_existing_dir([base / "Train"], "cuhk02")
    for img in _iter_images_in_dir(train_dir):
        first_token = img.stem.split("_")[0]
        if first_token in ("0000", "-1"):
            continue
        pid = int(first_token)
        yield ImageRecord(
            dataset="cuhk02",
            split="train",
            image_path=f"{train_dir.name}/{img.name}",
            abs_path=str(img.resolve()),
            pid=pid,
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
