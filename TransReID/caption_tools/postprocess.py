"""caption 原始输出清洗模块（caption 生成管线 - 后处理层）。

职责：
    读取推理阶段的 raw jsonl（每行包含 dataset / split / image_path / pid /
    raw_response / generator 等字段），执行：
      1. 用同目录 prompts.py 的 parse_response 解析 raw_response 得到 caption 列表；
      2. 基于 token 级 Jaccard 相似度去重；
      3. 生成启发式质量分（caption 数量 + 平均词数，0-1 区间）；
      4. 以训练侧契约格式写出 clean jsonl。

CLI：
    python postprocess.py --input raw.jsonl --output clean.jsonl \
        [--min-captions 2] [--dedup-threshold 0.9]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 与同目录 prompts.py 的接口契约：parse_response(text: str) -> list[str]
try:
    from prompts import parse_response
except ImportError:  # 兼容作为包模块导入的场景
    from .prompts import parse_response  # type: ignore[no-redef]

# 质量分启发式参数（见 compute_quality_score 注释）
_IDEAL_CAPTION_COUNT = 4   # caption 数达到该值时数量项记满分
_IDEAL_AVG_WORDS = 12.0    # 平均词数达到该值时长度项记满分
_MIN_AVG_WORDS = 3.0       # 平均词数低于该值认为 caption 过短，长度项趋近 0

# 英文 token 化：仅保留字母/数字/撇号，转小写
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> set[str]:
    """将 caption 转为小写 token 集合，用于 Jaccard 相似度计算。"""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """两个 token 集合的 Jaccard 相似度；两者皆空时视为完全相同（1.0）。"""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def dedup_captions(captions: list[str], threshold: float = 0.9) -> list[str]:
    """基于 token 级 Jaccard 相似度去重，保留首次出现的 caption。

    策略：逐条与"已保留集合"比较，与任一已保留 caption 的相似度 >= threshold
    即视为重复并丢弃。O(n^2)，但单图 caption 数通常 < 10，开销可忽略。
    """
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for cap in captions:
        cap = cap.strip()
        if not cap:
            continue  # 丢弃空 caption
        tokens = tokenize(cap)
        if any(jaccard_similarity(tokens, kt) >= threshold for kt in kept_tokens):
            continue  # 与已保留 caption 过于相似，判重
        kept.append(cap)
        kept_tokens.append(tokens)
    return kept


def compute_quality_score(captions: list[str]) -> float:
    """启发式质量分（0-1）：caption 数量项与平均词数项各占 50%。

    - 数量项：min(len / _IDEAL_CAPTION_COUNT, 1.0)，caption 越多越好（到 4 条封顶）；
    - 长度项：平均词数从 _MIN_AVG_WORDS 线性增长到 _IDEAL_AVG_WORDS 封顶，
      过短的 caption（如只有 1-2 个词）长度项趋近 0；
    - 无 caption 时直接返回 0.0。
    """
    if not captions:
        return 0.0
    count_term = min(len(captions) / _IDEAL_CAPTION_COUNT, 1.0)
    avg_words = sum(len(cap.split()) for cap in captions) / len(captions)
    length_term = (avg_words - _MIN_AVG_WORDS) / (_IDEAL_AVG_WORDS - _MIN_AVG_WORDS)
    length_term = max(0.0, min(length_term, 1.0))
    return round(0.5 * count_term + 0.5 * length_term, 4)


def process_record(
    raw: dict[str, Any],
    min_captions: int,
    dedup_threshold: float,
) -> dict[str, Any] | None:
    """将一条 raw 记录清洗为契约格式记录；无法解析出任何 caption 时返回 None。

    输入 raw 记录预期字段：dataset, split, image_path, pid, raw_response, generator。
    输出契约字段：dataset, split("train"), image_path, pid, captions, quality_score, generator。
    """
    raw_response = str(raw.get("raw_response", ""))
    captions = parse_response(raw_response)
    captions = dedup_captions(captions, threshold=dedup_threshold)
    if not captions:
        return None  # 解析失败 / 全部被去重，交由调用方统计丢弃数

    quality_score = compute_quality_score(captions)
    if len(captions) < min_captions:
        quality_score = round(quality_score * 0.5, 4)  # caption 数不足，质量分减半

    # pid 防御性转换：源数据可能是 int 或 str
    pid_raw = raw.get("pid", -1)
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = -1

    return {
        "dataset": raw.get("dataset", ""),
        "split": "train",  # 契约硬约束：只允许 train split
        "image_path": raw.get("image_path", ""),
        "pid": pid,
        "captions": captions,
        "quality_score": quality_score,
        "generator": raw.get("generator", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="清洗 caption 推理原始输出，产出训练侧契约格式的 clean jsonl。"
    )
    parser.add_argument("--input", required=True, help="raw jsonl 输入文件路径")
    parser.add_argument("--output", required=True, help="clean jsonl 输出文件路径")
    parser.add_argument(
        "--min-captions",
        type=int,
        default=2,
        help="每张图最少 caption 数，少于此数的记录 quality_score 减半（默认 2）",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.9,
        help="token 级 Jaccard 去重阈值，相似度 >= 该值判重（默认 0.9）",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"错误：输入文件不存在：{input_path}", file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0          # raw 总行数
    written = 0        # 成功写出的记录数
    dropped = 0        # 解析失败/全部被去重的记录数
    bad_lines = 0      # JSON 解析失败的行数
    low_caption = 0    # caption 数低于 min-captions 的记录数

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"警告：第 {line_no} 行 JSON 解析失败，已跳过：{e}", file=sys.stderr)
                continue
            record = process_record(raw, args.min_captions, args.dedup_threshold)
            if record is None:
                dropped += 1
                continue
            if len(record["captions"]) < args.min_captions:
                low_caption += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    # 处理汇总，便于在日志中快速判断清洗质量
    print("=" * 56)
    print(f"输入文件        : {input_path}")
    print(f"输出文件        : {output_path}")
    print(f"raw 记录总数    : {total}")
    print(f"成功写出        : {written}")
    print(f"无有效 caption  : {dropped}")
    print(f"JSON 坏行       : {bad_lines}")
    print(f"caption 数不足  : {low_caption}（quality_score 已减半）")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
