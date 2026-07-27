# caption_tools — Caption 数据重建管线

为 DG-ReID 源域训练集批量生成英文 caption（Qwen3-VL-32B-Instruct-FP8 + vLLM），
输出符合 `TransReID/ARCHITECTURE.md` 的 **Caption JSONL 契约**：

```json
{"dataset": "market1501", "split": "train", "image_path": "bounding_box_train/0002_c1s1_000451_03.jpg",
 "pid": 2, "captions": ["a person wearing a dark jacket"], "quality_score": 0.91, "generator": "..."}
```

> ⚠️ 契约约束：**只生成源域 train split**。目标域 query/gallery 的文本会被
> `CaptionStore` 显式拒绝（防止目标域文本泄漏），不要为它们生成 caption。

## 覆盖数据集（Protocol-1 + Protocol-2 源域）

| 数据集 | 目录约定（`--data-root` 下） | 规模 |
|---|---|---|
| market1501 | `market/bounding_box_train/` | ~1.3 万 |
| msmt17 | `msmt17/MSMT17_V2/mask_train_v2/` | ~3.3 万 |
| cuhk03 | `cuhk03-np/{detected,labeled}/bounding_box_train/` | ~1.5 万 |
| cuhksysu | `cuhksysu/train/`（或 `cropped_image/train/`） | ~1.2 万 |
| cuhk02 | `cuhk02/{cam1,cam2}/` | ~0.7 万 |

合计约 7 万张图，每图 6 条主题短句（上衣 / 下装+鞋 / 性别年龄发型 / 携带物 / 动作朝向 / 场景）。

## 服务器环境（Ubuntu 22.04 / Python 3.12 / RTX PRO 6000 96GB）

```bash
# 1. 安装依赖（torch 2.8.0+cu128 由镜像自带；否则走官方 cu128 索引安装）
pip install -r requirements-server.txt

# 2. 环境自检（GPU / 算力 sm_120 / timm dinov3 / vllm）
python check_env.py
```

## 使用流程

```bash
cd TransReID/caption_tools

# 步骤 0（强烈建议）：先跑 200 张冒烟测试，人工抽查质量
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --output-dir ./output/smoke --limit 200

# 步骤 1：批量生成（全部 5 个数据集；支持中断续跑，重跑同一命令即可）
python generate_captions.py --dataset all --data-root /root/autodl-tmp/data \
    --output-dir ./output/raw --config configs/default.yaml

# 步骤 2：清洗（解析 6 行短句、去 not visible、去重、质量分）
for d in market1501 msmt17 cuhk03 cuhksysu cuhk02; do
  python postprocess.py --input ./output/raw/$d.jsonl --output ./output/clean/$d.jsonl
done

# 步骤 3：合并导出最终 captions.jsonl（可选同时导出旧格式 train_caption_dict.json）
python export_jsonl.py --input-dir ./output/clean --output ./output/captions.jsonl
```

## 文件说明

| 文件 | 职责 |
|---|---|
| `prompts.py` | 6 主题 prompt 模板、颜色白名单、响应解析（剥序号 / 截句号 / 滤 not visible） |
| `datasets.py` | 5 个源域数据集 train split 遍历，输出 `ImageRecord` |
| `generate_captions.py` | vLLM 批量推理主脚本（分块、增量落盘、断点续跑、进度/ETA） |
| `postprocess.py` | raw → clean：解析、去重、质量分 |
| `export_jsonl.py` | clean → 契约 JSONL 合并导出（含 split 防泄漏兜底校验） |
| `check_env.py` | 服务器环境自检 |
| `configs/default.yaml` | 默认参数（模型、min/max_pixels、batch、显存利用率） |

## 关键参数

- `--min-pixels / --max-pixels`：控制低分辨率行人图的上采样幅度（默认 128×28² ~ 1024×28²）。
  若发现颜色/细节幻觉偏多，先调低 max_pixels；细节看不清则调高 min_pixels，用冒烟测试对比。
- `--model`：默认 `Qwen/Qwen3-VL-32B-Instruct-FP8`（96GB 卡推荐）；显存紧张可换
  `Qwen/Qwen3-VL-8B-Instruct`。
- 贪心解码（temperature=0）保证可复现；续跑按 `image_path` 去重跳过。
