# 01 数据管线：磁盘 jpg → batch 张量

> 历史资料：下文保留旧 CLIP/MaPLe 实现、早期协议规划或调研结论，不代表当前 NaFlex。请先阅读 [当前主线指南](07_naflex_current.md)。论文结论及数值需重新核对原文，旧路径与行号可能失效。

## 流程链

```
bounding_box_train/*.jpg
 └─ Market1501.process_dir()           data/datasets/image/market1501.py:73-97
 └─ process_caption()（可选）           market1501.py:99-143
 └─ ImageDataset.__getitem__           data/datasets/dataset.py:353-371
 └─ RandomIdentitySampler              data/sampler.py:19-87
 └─ cap_collate_fn                     data/datamanager.py:127-132
     img: (B,3,256,128) float32 [-1,1]
     pid/camid: (B,) int64；cap_text: B 个原始字符串（未 tokenize）
```

## 1. 数据集扫描与样本构建

- 入口：`data/build.py:31-34` `build_datamanager(cfg)` → `ImageDataManager`（`data/datamanager.py:212`）。
- 注册表：`data/datasets/__init__.py:10-24`，按名字实例化数据集类。
- `Market1501.process_dir`（market1501.py:73-97）：
  - `glob` 扫 `*.jpg`，正则 `([-\d]+)_c(\d)` 从文件名解析 pid/camid；
  - `pid == -1`（junk）跳过；`camid -= 1`（从 0 开始）；train 集 pid relabel 为 0..N-1；
  - 产出 `(img_path, pid, camid)` 三元组列表。
- 基类 `dataset.py:57-62` 统一扩为 4 元组（+dsetid）；多源数据集 `__add__`（dataset.py:122-164）按累加偏移 relabel pid/camid/dsetid——DG 多源训练的关键。

## 2. caption 配对（训练实际路径）

- `process_caption`（market1501.py:99-143）：读 `train_caption_dict.json`（`{img_path: [cap1, cap2, ...]}`），每条 caption 只保留第一个 `.` 之前的句子并 `strip().lower()`；按 `cap_num` 选条，多条用 `', '` 拼接。
- 结果：4 元组第 4 位被 **caption 字符串顶替 dsetid**（dataset.py:367 注释）。
- ⚠️ 新契约代码 `data/caption_store.py` / `data/records.py`（JSONL、拒绝非 train split）**尚未接入训练 DataLoader**，目前仅 caption_tools 离线工具链和 tests 使用。

## 3. Transforms（data/transforms.py:237-367）

训练管线（L281-330）顺序：

| # | 操作 | 说明 |
|---|---|---|
| 1 | `Resize((256,128), BICUBIC)` | **定尺寸的一步** |
| 2 | `RandomHorizontalFlip` | 概率取的是 `randomerase_prob`=0.5（疑似笔误，原应用 INPUT.PROB） |
| 3 | `Pad(10)` + `RandomCrop(256,128)` | 填充后裁回原尺寸，不改最终尺寸 |
| 4 | `ColorJitter(0.2, 0.15)` | 亮度/对比度 |
| 5 | `ToTensor()` | float [0,1]，(3,H,W) |
| 6 | `Normalize(mean,std)` | **CLIP 风格 [0.5,0.5,0.5] → [-1,1]**，非 ImageNet 值 |
| 7 | `RandomErasing(mode='pixel')` | 在 Normalize 之后 |

测试管线（L337-341）：仅 Resize → ToTensor → Normalize。

尺寸来源：`configs/*/vit_caption_maple.yml` 的 `INPUT.SIZE_TRAIN/SIZE_TEST: [256, 128]`（defaults.py 默认 [384,128]，被 yml 覆盖），经 `data/build.py:12-13` → `ImageDataManager(height, width)` → `build_transforms`。

## 4. Sampler 与 collate

- `RandomIdentitySampler`（sampler.py:19-87）：PK 采样——每 batch 随机选 `batch_size // num_instances` 个 pid，每个 pid 取 `num_instances`（配置 4）张，不足时 `np.random.choice(replace=True)` 有放回补齐。为 triplet loss 提供类内/类间多样性。
- `cap_collate_fn`（datamanager.py:127-132）：`torch.stack(imgs)` + pids/camids int64 + impaths + cap_text 字符串元组。
- 无 caption 分支 `collate_fn`（L120-125）第 5 字段是 dsetids；验证用 `val_cap_collate_fn`（L141-146）返回 6 元组。
- DataLoader 组装：datamanager.py:399-417（caption 分支 `shuffle=False`，顺序全由 sampler 决定）。

## 5. tokenize 在模型侧

batch 的 caption 字符串在模型内才 tokenize：`model/prompt/promptlearner.py:390-397`（`replace("_"," ")` → lower → 拼 `"a photo of"` 前缀 → `clip.tokenize` → (B,77)）。`engine/batch.py:26-32` 还会按 caption 是否为空生成 `caption_mask` (B,) bool。

## 坑

1. caption 模式下 4 元组第 4 位是字符串而非 dsetid：任何按 `items[3]` 取 dsetid 的逻辑（`RandomDatasetSampler`、`dataset.__add__`）在 caption 模式下被绕过/会出错（dataset.py:80-81 有 caption 时不算 num_datasets）。
2. HDRNet 分支（transforms.py:342-362）`transform_low/full` 硬编码 [256,128]，与 cfg 解耦。
