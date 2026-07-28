# caption_tools — Caption 数据重建管线

为 DG-ReID 源域训练集批量生成英文 caption（Qwen3-VL-32B-Instruct-FP8 + vLLM），
输出符合 `TransReID/ARCHITECTURE.md` 的 **Caption JSONL 契约**。当前同时保留：

- `v1`：原始六行自然语言 Prompt，作为可复现实验基线；
- `v2`：结构化 ReID 外观属性 JSON，再由确定性规则渲染为训练 caption；
- `v2.1`：保留 V2 schema，要求每项都是完整物品短语，并去除显著特征中的基础衣物复述；
- `v2.2`：增加置信度感知的鞋型规则，并只渲染一条无重叠的综合训练 caption；
- `v2.3`：保留有用的 `graphic` 上位词，但优先具体特征类别并要求图案带有判别性修饰。
- `v2.4`：仅在视觉证据充分时使用 `logo/text/patch`；模糊但确实可见的小标记回退为 `mark`。

```json
{"dataset": "market1501", "split": "train", "image_path": "bounding_box_train/0002_c1s1_000451_03.jpg",
 "pid": 2, "captions": ["a person wearing a dark jacket"], "quality_score": 0.91,
 "generator": "...", "prompt_version": "v2", "attributes": {"upper_clothing": {"visibility": "clear", "attributes": ["dark jacket"]}}}
```

历史 JSONL 没有 `prompt_version` 时自动按 `v1` 读取，不会修改原文件。

> ⚠️ 契约约束：**只生成源域 train split**。目标域 query/gallery 的文本会被
> `CaptionStore` 显式拒绝（防止目标域文本泄漏），不要为它们生成 caption。

## 覆盖数据集（Protocol-1 + Protocol-2 源域）

| 数据集 | 目录约定（`--data-root` 下） | 规模 |
|---|---|---|
| market1501 | `market/bounding_box_train/` | ~1.3 万 |
| msmt17 | `msmt17/MSMT17_V2/mask_train_v2/` | ~3.3 万 |
| cuhk03 | `cuhk03-np/{detected,labeled}/bounding_box_train/` | ~1.5 万 |
| cuhksysu | `cuhksysu/cropped_images/`（全部图片即训练集，train-only） | ~3.5 万 |
| cuhk02 | `cuhk02/{cam1,cam2}/` | ~0.7 万 |

V1 每图生成 6 条主题短句；V2 只保留跨摄像头相对稳定、可见的衣着、鞋、携带物、
配饰、头发和显著图案，排除性别年龄、姿态朝向、场景与光照。

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

# 步骤 0（强烈建议）：先跑 200 张 V2 冒烟测试，人工抽查质量
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v2 --output-dir ./output/smoke/v2 --limit 200

# 复现 V1 基线时使用独立目录
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v1 --output-dir ./output/smoke/v1 --limit 200

# 对比 V2.1（使用同一批前 200 张图片）
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v2.1 --output-dir ./output/smoke/v2.1 --limit 200

# 对比 V2.2（同图；具体鞋型优先，不清楚时退回 color + shoes）
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v2.2 --output-dir ./output/smoke/v2.2 --limit 200

# 对比 V2.3（同图；graphic 必须含颜色、大小或形状）
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v2.3 --output-dir ./output/smoke/v2.3 --limit 200

# 对比 V2.4（同图；避免把模糊小标记过度解释为 logo）
python generate_captions.py --dataset market1501 --data-root /root/autodl-tmp/data \
    --prompt-version v2.4 --output-dir ./output/smoke/v2.4 --limit 200

# 步骤 1：批量生成（冻结 Prompt V2.4；支持中断续跑，重跑同一命令即可）
python generate_captions.py --dataset all --data-root /root/autodl-tmp/data \
    --config configs/default.yaml

# 步骤 2：清洗（训练只使用 clean；raw 仅用于审计生成质量）
for d in market1501 msmt17 cuhk03 cuhksysu cuhk02; do
  python postprocess.py --input ./output/raw/v2.4/$d.jsonl --output ./output/clean/v2.4/$d.jsonl
done

# 步骤 3：合并导出最终 captions.jsonl（可选同时导出旧格式 train_caption_dict.json）
python export_jsonl.py --input-dir ./output/clean/v2.4 --output ./output/captions-v2.4.jsonl
```

生成脚本会检查断点文件中的版本；同一 JSONL 已含 V1 时，V2 会拒绝续写并提示换目录。

## 文件说明

| 文件 | 职责 |
|---|---|
| `prompts.py` | 冻结的 V1、结构化 V2、版本选择、响应解析与 V2 确定性 caption 渲染 |
| `datasets.py` | 5 个源域数据集 train split 遍历，输出 `ImageRecord` |
| `generate_captions.py` | vLLM 批量推理主脚本（分块、增量落盘、断点续跑、进度/ETA） |
| `postprocess.py` | raw → clean：按版本解析、去重、版本化质量分 |
| `export_jsonl.py` | clean → 契约 JSONL 合并导出（校验版本、V2 attributes 和 split） |
| `check_env.py` | 服务器环境自检 |
| `configs/default.yaml` | 默认参数（模型、min/max_pixels、batch、显存利用率） |

## 关键参数

- `--min-pixels / --max-pixels`：控制低分辨率行人图的上采样幅度（默认 128×28² ~ 1024×28²）。
  若发现颜色/细节幻觉偏多，先调低 max_pixels；细节看不清则调高 min_pixels，用冒烟测试对比。
- `--model`：默认 `Qwen/Qwen3-VL-32B-Instruct-FP8`（96GB 卡推荐）；显存紧张可换
  `Qwen/Qwen3-VL-8B-Instruct`。
- `--prompt-version`：默认冻结为 `v2.4`；仍支持 `v1`、`v2`、`v2.1`、`v2.2`、`v2.3`，各版本必须使用独立输出目录。
- `--output-dir`：不传时自动使用 `output/raw/<prompt-version>`。
- 贪心解码（temperature=0）保证可复现；续跑按 `image_path` 去重跳过。

训练侧应加载 clean 后由 `export_jsonl.py` 合并的 `captions.jsonl`，不得直接
加载含 `raw_response` 的生成文件。V2.2–V2.4 的 clean 记录包含
`postprocess_version=p2` 和 `renderer_version=r2-canonical`，且
`captions` 只含一条综合描述；结构化 `attributes` 继续保留用于审计和消融。
