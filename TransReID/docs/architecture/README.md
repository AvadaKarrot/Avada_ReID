# 网络架构梳理索引

> 目的：随时查阅 IPCA-ReID / MaPLe-CLIP 主线模型的代码细节。
> 主线模型：`model/make_model_caption.py`（被 `train_caption.py` 使用）。
> 注意：`model/maple/build_transformer_capmaple.py` 是重构中的旧版本，调用参数不匹配，跑不起来，**不要参考**。

## 文档列表

| 文档 | 内容 |
|---|---|
| [01_data_pipeline.md](01_data_pipeline.md) | 磁盘 jpg → dataset 解析 → Transforms → PK Sampler → collate → batch 张量 |
| [02_visual_encoder.md](02_visual_encoder.md) | (B,3,256,128) → patch embed → pos_embed 插值 → MaPLe 视觉 prompt → CLS 特征 |
| [03_text_encoder_maple.md](03_text_encoder_maple.md) | caption 字符串 → BPE (B,77) → ctx 替换构造 → Text Encoder → 图文耦合机制 |
| [04_model_output.md](04_model_output.md) | encoder 输出 → CLS → BNNeck/classifier → ID+triplet 损失；cross-attn 融合细节；CLIP-ReID baseline 对比；隐患清单；DINOv3 迁移要点 |
| [05_experiment_protocols.md](05_experiment_protocols.md) | DG-ReID Protocol-1/2/3 标准定义（Survey arXiv:2506.12413 §5.3）；Source/Target 轮换规则；数据集准备执行清单 |
| [06_siglip2_finetune_strategies.md](06_siglip2_finetune_strategies.md) | SigLIP2 下游微调策略调研（2025–2026）：官方立场、PEFT 策略光谱、ReID 现有实践、本项目选型建议与空白点 |

## 一图流总览

```
【数据侧】
磁盘 jpg → dataset 解析文件名 (img_path, pid, camid[, caption])
  → PIL 读 RGB → Transforms（Resize 256×128 → flip → pad/crop
    → jitter → ToTensor → Normalize[0.5] → RandomErasing）
  → RandomIdentitySampler（每 batch 选 B/4 个 pid × 每 pid 4 张）
  → cap_collate_fn → img (B,3,256,128) + pid + caption 字符串

【视觉侧】CLIP ViT-B/16，主干冻结
img (B,3,256,128)
  → conv1 patch embed (k=s=16, 3→768) → 16×8=128 patch
  → + class_token → (B,129,768)
  → + pos_embed（197→129 双线性插值适配非方形）
  → 拼 3 个视觉 prompt token → (B,132,768)
  → 12 层 transformer（第 2~12 层逐层替换 deep prompt）
  → 取 CLS：img_feature (B,768) / img_feature_proj (B,512)

【文本侧】CLIP text transformer，width 512，整体冻结
caption 字符串 → "a photo of " + caption + "."
  → BPE tokenize → (B,77)（SOT=49406，EOT=49407，上限 77 硬约束）
  → embedding (B,77,512)，切掉 "a photo of" 的 3 个位置
  → 可学习 ctx (B,3,512) 替换填入 → [SOT|ctx|caption|EOT|pad] (B,77,512)
  → 12 层 transformer（第 2~12 层注入 deep text prompt）
  → 取 EOT 位置 @ text_projection → text_features (B,512)

【MaPLe 耦合】视觉 prompt = 文本 prompt 经 Linear(512→768) 投影，
  浅层 proj(ctx) 1 组 + 深层 proj_i(compound_prompts_text[i]) 11 组
  （PROMPT_DEPTH=12, N_CTX=3，见 configs/*/vit_caption_maple.yml）

【训练参数】冻结：CLIP 图像主干 + 文本编码器全部冻结；
  可训练：prompt_learner（ctx/proj/compound prompts）、cross_attn、
  text_projectoin、classifier、bottleneck
```

## 待补充（后续梳理计划）

- [x] 损失函数（输出侧）：ID + triplet 组合、i2t 死代码 → 见 04_model_output.md §2/§5；`objectives/` 新体系的完整梳理待补
- [ ] 训练循环：`processor/processor_caption.py` / `engine/`
- [ ] 评测：`evaluation/`（mAP / Rank-k，跨域评测协议）
- [ ] 推理/测试路径特征提取
- [ ] DINOv3 / SigLIP2 backbone 迁移方案（耦合点：patch embed、pos_embed 插值、width 768/512、无文本塔问题）
