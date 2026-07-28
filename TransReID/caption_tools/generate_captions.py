"""Offline batch caption generation for source-domain ReID train images.

基于 vLLM 离线推理（Qwen3-VL-32B-Instruct-FP8）为 5 个源域数据集的
train split 批量生成原始 caption。输出为 JSONL（每行一条原始记录，
含 raw_response），由下游脚本解析/过滤为训练契约格式。

运行示例：
    python generate_captions.py \
        --dataset all \
        --data-root /root/autodl-tmp/data \
        --output-dir ./output/raw

    # 调试：只跑 market1501 前 50 张
    python generate_captions.py --dataset market1501 \
        --data-root /root/autodl-tmp/data --limit 50

    # 使用配置文件（CLI 参数优先于配置文件）
    python generate_captions.py --config configs/default.yaml

续跑：默认开启。重新执行同一命令时，脚本会读取已有输出文件并跳过
已完成的 image_path，因此中断后可直接重跑。

目标环境：Ubuntu 22.04 / Python 3.12 / PyTorch 2.8.0+cu128 /
vLLM / RTX PRO 6000 96GB (sm_120)。仅依赖 vllm、Pillow、PyYAML。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml
from PIL import Image

from datasets import iter_source_train_images  # 由数据遍历工程师提供
from prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_V1,
    SUPPORTED_PROMPT_VERSIONS,
    build_prompt,
    get_system_prompt,
    normalize_prompt_version,
)

# 全部源域数据集（与 configs/default.yaml 中 datasets 列表保持一致）。
ALL_DATASETS: tuple[str, ...] = (
    "market1501",
    "msmt17",
    "cuhk03",
    "cuhksysu",
    "cuhk02",
)

# 每块处理的图片数：一块完成后立即增量写盘并 flush，保证中断不丢进度。
DEFAULT_CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# 配置与 CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；--config 指定的 YAML 作为默认值，CLI 显式参数优先。"""
    parser = argparse.ArgumentParser(
        description="vLLM offline batch caption generation for source ReID train sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（可选）；CLI 显式传入的参数优先于配置文件。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"单个数据集名或 all；all 展开为 {list(ALL_DATASETS)}。",
    )
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录。")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="原始 JSONL 输出目录（默认 caption_tools/output/raw/<prompt-version>）。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="生成模型（HuggingFace 名或本地路径）。",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        choices=SUPPORTED_PROMPT_VERSIONS,
        default=None,
        help="Prompt 版本；v1 保留原六行基线，v2 使用结构化 ReID 属性。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="调试用：每个数据集只处理前 N 张图。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="断点续跑：读取已有输出并跳过已完成的 image_path。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="每条 caption 的最大生成 token 数。",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization。",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM 引擎最大上下文长度；caption 任务无需使用模型原生超长上下文。",
    )
    parser.add_argument(
        "--use-flashinfer-sampler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="使用 FlashInfer top-k/top-p 采样器；不兼容时关闭。",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="传给 mm_processor_kwargs 的 min_pixels。",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="传给 mm_processor_kwargs 的 max_pixels。",
    )

    args = parser.parse_args(argv)

    # 配置文件提供默认值；CLI 显式值覆盖配置文件。
    file_cfg: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            parser.error(f"配置文件不存在: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if not isinstance(file_cfg, dict):
            parser.error(f"配置文件格式错误（顶层必须是 mapping）: {cfg_path}")

    defaults = {
        "dataset": "all",
        "data_root": None,
        "output_dir": None,
        "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "limit": None,
        "resume": True,
        "max_new_tokens": 256,
        "gpu_mem": 0.9,
        "max_model_len": 4096,
        "use_flashinfer_sampler": False,
        "min_pixels": 128 * 28 * 28,
        "max_pixels": 1024 * 28 * 28,
    }
    # 配置文件键名兼容 CLI 风格（data_root / data-root 均可）。
    key_map = {
        "data_root": ("data_root", "data-root"),
        "output_dir": ("output_dir", "output-dir"),
        "prompt_version": ("prompt_version", "prompt-version"),
        "max_new_tokens": ("max_new_tokens", "max-new-tokens"),
        "gpu_mem": ("gpu_mem", "gpu-mem", "gpu_memory_utilization"),
        "max_model_len": ("max_model_len", "max-model-len"),
        "use_flashinfer_sampler": (
            "use_flashinfer_sampler",
            "use-flashinfer-sampler",
        ),
        "min_pixels": ("min_pixels", "min-pixels"),
        "max_pixels": ("max_pixels", "max-pixels"),
    }

    merged: dict[str, Any] = dict(defaults)
    for key, val in file_cfg.items():
        norm = val
        canonical = key
        for dst, aliases in key_map.items():
            if key in aliases:
                canonical = dst
                break
        merged[canonical] = norm
    for key in defaults:
        cli_val = getattr(args, key)
        if cli_val is not None:
            merged[key] = cli_val

    for key, val in merged.items():
        setattr(args, key, val)

    args.prompt_version = normalize_prompt_version(args.prompt_version)
    if not args.output_dir:
        args.output_dir = str(
            Path(__file__).resolve().parent
            / "output"
            / "raw"
            / args.prompt_version
        )
    if not args.data_root:
        parser.error("必须提供 --data-root（或在配置文件中设置 data_root）。")

    return args


def resolve_datasets(dataset_arg: str, cfg_datasets: Sequence[str] | None) -> list[str]:
    """把 --dataset 参数解析为数据集列表；配置文件可覆盖 all 的展开范围。"""
    if dataset_arg.lower() == "all":
        datasets = list(cfg_datasets) if cfg_datasets else list(ALL_DATASETS)
    else:
        datasets = [dataset_arg.lower()]
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        raise ValueError(
            f"未知数据集: {unknown}；支持: {list(ALL_DATASETS)} 或 all。"
        )
    return datasets


# ---------------------------------------------------------------------------
# 续跑支持
# ---------------------------------------------------------------------------

def load_done_paths(output_file: Path, prompt_version: str) -> set[str]:
    """读取断点并拒绝在同一 JSONL 中混写不同 Prompt 版本。"""
    done: set[str] = set()
    if not output_file.is_file():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[warn] {output_file.name} 第 {lineno} 行 JSON 损坏，已忽略",
                    file=sys.stderr,
                )
                continue
            existing_version = normalize_prompt_version(
                str(rec.get("prompt_version", PROMPT_V1))
            )
            if existing_version != prompt_version:
                raise ValueError(
                    f"{output_file} 已包含 prompt_version={existing_version}，"
                    f"当前请求为 {prompt_version}；请使用独立输出目录。"
                )
            path = rec.get("image_path")
            if isinstance(path, str):
                done.add(path)
    return done


# ---------------------------------------------------------------------------
# vLLM 推理
# ---------------------------------------------------------------------------

def build_llm(args: argparse.Namespace):
    """初始化 vLLM 引擎。延迟导入 vllm，使 --help / 参数校验可在无 GPU 环境运行。"""
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.use_flashinfer_sampler else "0"
    )
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError(
            "未找到 vllm，请在目标环境（Ubuntu 22.04, PyTorch 2.8.0+cu128）中安装 vLLM。"
        ) from e

    return LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
    )


def run_chunk(
    llm: Any,
    sampling_params: Any,
    chunk: Sequence[Any],
    system_prompt: str,
    prompt_text: str,
) -> list[tuple[Any, str]]:
    """对一个分块执行批量多模态推理。

    Returns:
        (record, raw_response) 列表；打开失败或推理失败的样本被跳过并告警。
    """
    conversations: list[list[dict[str, Any]]] = []
    kept_records: list[Any] = []

    for rec in chunk:
        try:
            # 统一转 RGB，避免 RGBA/灰度图导致多模态处理器异常。
            image = Image.open(rec.abs_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            print(f"[warn] 无法打开图片，跳过: {rec.abs_path} ({e})", file=sys.stderr)
            continue
        conversations.append(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
        )
        kept_records.append(rec)

    if not conversations:
        return []

    outputs = llm.chat(conversations, sampling_params)

    results: list[tuple[Any, str]] = []
    for rec, out in zip(kept_records, outputs):
        if not out.outputs:
            print(f"[warn] 无生成结果，跳过: {rec.image_path}", file=sys.stderr)
            continue
        results.append((rec, out.outputs[0].text.strip()))
    return results


def write_records(
    fh: Any,
    results: Sequence[tuple[Any, str]],
    model_name: str,
    prompt_version: str,
) -> int:
    """把一块结果增量追加写入 JSONL 并 flush+fsync，保证中断不丢进度。"""
    import os

    n = 0
    for rec, raw_response in results:
        row = {
            "dataset": rec.dataset,
            "split": "train",  # 契约：只允许 train split
            "image_path": rec.image_path,
            "pid": rec.pid,
            "raw_response": raw_response,
            "generator": model_name,
            "prompt_version": prompt_version,
        }
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    fh.flush()
    os.fsync(fh.fileno())
    return n


# ---------------------------------------------------------------------------
# 单数据集主流程
# ---------------------------------------------------------------------------

def process_dataset(
    llm: Any,
    sampling_params: Any,
    dataset: str,
    args: argparse.Namespace,
    system_prompt: str,
    prompt_text: str,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}.jsonl"

    # 物化迭代器以便统计总数与分块（train 集规模约万级，内存可承受）。
    print(f"[{dataset}] 枚举图片: root={args.data_root}")
    records = list(iter_source_train_images(dataset, args.data_root))
    if args.limit is not None:
        records = records[: args.limit]
    total = len(records)
    if total == 0:
        print(f"[{dataset}] 没有可处理的图片，跳过。")
        return

    done: set[str] = set()
    if args.resume:
        done = load_done_paths(output_file, args.prompt_version)
        if done:
            print(f"[{dataset}] 续跑：已完成 {len(done)} 张，跳过。")

    todo = [r for r in records if r.image_path not in done]
    if not todo:
        print(f"[{dataset}] 全部 {total} 张已完成，无需处理。")
        return

    print(f"[{dataset}] 待处理 {len(todo)}/{total} 张 -> {output_file}")

    processed = len(done)
    t_start = time.perf_counter()
    t_done = processed  # 续跑时已完成的数量也计入进度展示

    with output_file.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), DEFAULT_CHUNK_SIZE):
            chunk = todo[start : start + DEFAULT_CHUNK_SIZE]
            results = run_chunk(
                llm,
                sampling_params,
                chunk,
                system_prompt,
                prompt_text,
            )
            written = write_records(
                fh,
                results,
                args.model,
                args.prompt_version,
            )

            t_done += written
            elapsed = time.perf_counter() - t_start
            # 速度按本次运行实际写入数估算；续跑时总量含历史进度。
            speed = (t_done - processed) / elapsed if elapsed > 0 else 0.0
            eta_s = (total - t_done) / speed if speed > 0 else float("inf")
            print(
                f"[{dataset}] {t_done}/{total} "
                f"({speed:.2f} img/s, 本块 +{written}, ETA {eta_s / 60:.1f} min)"
            )

    print(f"[{dataset}] 完成：{t_done}/{total} -> {output_file}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_datasets = None
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg_datasets = (yaml.safe_load(f) or {}).get("datasets")
    try:
        datasets = resolve_datasets(args.dataset, cfg_datasets)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    print(f"模型: {args.model}")
    print(f"Prompt: {args.prompt_version}")
    print(f"数据集: {datasets}")
    print(f"输出目录: {args.output_dir}")
    print(
        f"采样: max_new_tokens={args.max_new_tokens}; "
        f"mm: min_pixels={args.min_pixels}, max_pixels={args.max_pixels}; "
        f"gpu_memory_utilization={args.gpu_mem}; "
        f"max_model_len={args.max_model_len}; "
        f"use_flashinfer_sampler={args.use_flashinfer_sampler}"
    )

    llm = build_llm(args)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,  # 贪心解码：caption 需要稳定可复现
        max_tokens=args.max_new_tokens,
    )

    system_prompt = get_system_prompt(args.prompt_version)
    prompt_text = build_prompt(args.prompt_version)

    t_all = time.perf_counter()
    for dataset in datasets:
        process_dataset(
            llm,
            sampling_params,
            dataset,
            args,
            system_prompt,
            prompt_text,
        )

    print(f"全部完成，总耗时 {(time.perf_counter() - t_all) / 60:.1f} min。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
