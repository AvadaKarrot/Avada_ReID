"""clean jsonl 合并导出模块（caption 生成管线 - 导出层）。

职责：
    将一个目录下的所有 clean jsonl（postprocess.py 的产物）合并为最终的
    captions.jsonl；可选地按旧格式 train_caption_dict.json 同步导出
    （{image_path: [captions]}，每个数据集一份，写进各数据集目录）。

CLI：
    python export_jsonl.py --input-dir clean/ --output captions.jsonl \
        [--legacy-json-dir /path/to/data_root]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 训练侧契约要求的字段（用于合并时做完整性校验）
_CONTRACT_FIELDS = ("dataset", "split", "image_path", "pid", "captions", "quality_score", "generator")
_PROMPT_VERSIONS = ("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4")


def _iter_clean_records(input_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    """遍历 input_dir 下所有 *.jsonl，返回 (全部记录, 坏行数, 缺字段行数)。

    文件按文件名排序处理，保证合并结果顺序可复现。
    """
    records: list[dict[str, Any]] = []
    bad_lines = 0
    incomplete = 0
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"输入目录中没有找到任何 *.jsonl 文件：{input_dir}\n"
            f"请确认 postprocess.py 的 clean 输出已写入该目录。"
        )
    for jf in jsonl_files:
        with jf.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    print(f"警告：{jf.name} 第 {line_no} 行 JSON 解析失败，已跳过。", file=sys.stderr)
                    continue
                missing = [k for k in _CONTRACT_FIELDS if k not in rec]
                if missing:
                    incomplete += 1
                    print(
                        f"警告：{jf.name} 第 {line_no} 行缺少契约字段 {missing}，已跳过。",
                        file=sys.stderr,
                    )
                    continue
                if rec["split"] != "train":
                    # 防泄漏兜底：非 train split 一律拒绝（契约硬约束）
                    print(
                        f"警告：{jf.name} 第 {line_no} 行 split={rec['split']!r} 非 train，已拒绝。",
                        file=sys.stderr,
                    )
                    incomplete += 1
                    continue
                prompt_version = str(rec.get("prompt_version", "v1")).lower()
                if prompt_version not in _PROMPT_VERSIONS:
                    print(
                        f"警告：{jf.name} 第 {line_no} 行 prompt_version="
                        f"{prompt_version!r} 非法，已跳过。",
                        file=sys.stderr,
                    )
                    incomplete += 1
                    continue
                if prompt_version in ("v2", "v2.1", "v2.2", "v2.3", "v2.4") and not isinstance(
                    rec.get("attributes"), dict
                ):
                    print(
                        f"警告：{jf.name} 第 {line_no} 行为 V2 但缺少 attributes，已跳过。",
                        file=sys.stderr,
                    )
                    incomplete += 1
                    continue
                rec["prompt_version"] = prompt_version
                records.append(rec)
    return records, bad_lines, incomplete


def _export_legacy_json(
    records: list[dict[str, Any]], legacy_json_dir: Path
) -> dict[str, int]:
    """按旧格式导出 train_caption_dict.json：{image_path: [captions]}，每个数据集一份。

    文件写入 legacy_json_dir/<dataset>/train_caption_dict.json，
    与训练侧旧版 caption 加载逻辑兼容。返回各数据集写出的条目数。
    """
    # dataset -> {image_path: captions}
    grouped: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for rec in records:
        grouped[rec["dataset"]][rec["image_path"]] = rec["captions"]

    counts: dict[str, int] = {}
    for dataset, mapping in sorted(grouped.items()):
        out_dir = legacy_json_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "train_caption_dict.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        counts[dataset] = len(mapping)
        print(f"旧格式已导出：{out_path}（{len(mapping)} 条）")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="合并 clean jsonl 目录为最终 captions.jsonl，可选导出旧格式 train_caption_dict.json。"
    )
    parser.add_argument("--input-dir", required=True, help="clean jsonl 所在目录")
    parser.add_argument("--output", required=True, help="最终 captions.jsonl 输出路径")
    parser.add_argument(
        "--legacy-json-dir",
        default=None,
        help="可选：数据根目录。提供时按旧格式 train_caption_dict.json 导出到各数据集目录",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"错误：输入目录不存在：{input_dir}", file=sys.stderr)
        return 1

    try:
        records, bad_lines, incomplete = _iter_clean_records(input_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    if not records:
        print("错误：合并后没有任何有效记录，未生成输出文件。", file=sys.stderr)
        return 1

    # 写最终 captions.jsonl（保持契约字段顺序，便于人工检查）
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            ordered = {k: rec[k] for k in _CONTRACT_FIELDS}
            ordered["prompt_version"] = rec["prompt_version"]
            if "postprocess_version" in rec:
                ordered["postprocess_version"] = rec["postprocess_version"]
            if "renderer_version" in rec:
                ordered["renderer_version"] = rec["renderer_version"]
            if rec["prompt_version"] in ("v2", "v2.1", "v2.2", "v2.3", "v2.4"):
                ordered["attributes"] = rec["attributes"]
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")

    # 按 dataset 统计条数与平均质量分，打印汇总表
    stats: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        stats[rec["dataset"]].append(float(rec["quality_score"]))

    print("=" * 64)
    print(f"合并完成：{output_path}")
    print(f"有效记录总数：{len(records)}"
          f"（JSON 坏行 {bad_lines}，契约不完整/非 train {incomplete}）")
    print("-" * 64)
    print(f"{'dataset':<14}{'records':>10}{'avg_quality':>14}")
    print("-" * 64)
    for dataset in sorted(stats):
        scores = stats[dataset]
        print(f"{dataset:<14}{len(scores):>10}{sum(scores) / len(scores):>14.4f}")
    print("-" * 64)
    print(f"{'TOTAL':<14}{len(records):>10}"
          f"{sum(sum(s) for s in stats.values()) / len(records):>14.4f}")
    print("=" * 64)

    # 可选：旧格式导出
    if args.legacy_json_dir:
        legacy_dir = Path(args.legacy_json_dir)
        legacy_dir.mkdir(parents=True, exist_ok=True)
        _export_legacy_json(records, legacy_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
