# 06 — SigLIP2 下游迁移微调策略调研（2025–2026）

> 调研日期：2026-08-09
> 目的：评估将本项目 CLIP 框架（TransReID / CLIP-ReID 路线）替换为 SigLIP2 backbone 的可行性与微调策略选型
> 方法：三路并行文献调研（官方资料 / PEFT 策略全景 / ReID 领域实践），所有关键结论附来源

---

## 1. 总体结论（TL;DR）

1. **SigLIP2 官方没有下游微调 recipe**：除开放词汇检测（OWL-ViT 路线）外，官方全部下游评估都冻结 backbone。无官方 LR / LLRD / 解冻层数建议。
2. **社区共识：LoRA 等 PEFT ≈ 全量微调，且更保泛化**。多个独立消融（IQA、VLA、医学影像）证实：冻结掉点明显，但 LoRA 追平 full FT 且省算力；DG 场景下全量微调有害的证据在累积。
3. **"SigLIP2-ReID" 是空白点**：截至 2026-08，没有任何一篇正式工作系统做过 SigLIP2 的 ReID 微调（全量 vs PEFT 消融、prompt learning 迁移、DG 协议对比）。这是本项目的机会点。
4. **工程注意**：SigLIP2 无 CLS token（用 MAP attention pooling），与 TransReID 取 CLS 的代码路径不兼容；文本侧 sigmoid loss 无 logit scale，CLIP-ReID 式两阶段 prompt learning 需重新设计。

---

## 2. 官方立场（SigLIP2 论文 / big_vision / HF）

来源：Tschannen et al., "SigLIP 2", TMLR 2025, arXiv:2502.14786

| 下游任务 | 官方做法 | 备注 |
|---|---|---|
| 分类 | zero-shot + 10-shot linear probe | 无论文内全量微调实验 |
| 检索 | zero-shot recall@1 | 无微调 |
| VLM 视觉编码器 | **冻结**（引 PaliGemma 2 Sec 5.4：冻结基本不影响质量） | 只训 projector + LLM |
| 稠密预测（分割/深度/法线） | 冻结 + linear probe / DPT decoder（协议转引 TIPS arXiv:2410.16512） | 用 MAP head 输出代替 CLS 拼到 patch 特征 |
| 开放词汇分割 | 冻结 + Cat-Seg 框架（CVPR 2024） | |
| 开放词汇检测 | **全量微调**（唯一），配置转引 OWL-ViT (ECCV'22) / OWLv2 (NeurIPS'23) | 旁证：lr 5e-5→2e-5，droplayer 0.1–0.2，大骨干微调易过拟合需降 LR |
| 指代表达理解 | 冻结 + 从头训 6 层 cross-attn decoder（LocCa, CVPR 2024） | |

官方明确反对的做法：对最终 checkpoint 用小 LR + 去 weight decay 微调换分辨率（各尺寸/分辨率上效果不好，已弃用）；分辨率适配应续训 + pos-emb resize。

**未查到**：官方下游全量微调 LR 范围、LLRD 系数、渐进解冻建议、度量学习/ID 级检索微调 recipe。

---

## 3. 社区微调策略光谱（2025–2026，从轻到重）

| 策略 | 代表工作 | 核心结论 |
|---|---|---|
| Zero-shot / 冻结 | ReID 横评 arXiv:2601.20598（9 数据集） | SigLIP2 零样本 ReID mAP 5–14%，**比 CLIP 零样本强 5–10 倍**；在野数据集反超微调 CLIP-ReID |
| Linear / Attentive probe | Attentive Probing（OpenReview） | ViT-L attentive probe 87.0 IN acc，接近全微调 |
| 特征调制（FLA/FTM） | VLA 泛化性研究（CVPR 2026） | 4.7M 参数 FLA ≈ 467M LoRA；prompt 条件化最弱 |
| Prompt tuning / VPT | ELIP（arXiv:2502.15682，检索）；VICP（arXiv:2508.21222，DINOv2 ReID） | 轻量；对 SigLIP2 效果偏弱 |
| **LoRA（最主流）** | NR-IQA（arXiv:2509.17374）：rank=4 注入 q/k，lr 1e-4；AutoReID（ACL-F 2026）：r=16，lr 1e-5 | **LoRA ≈ full FT 且省算力**；冻结掉 3–20 SRCC |
| 选择性解冻末 2–4 层 | MedGemma MSK（arXiv:2511.05600） | 明显优于全冻 head-only；分层 LR + cosine |
| Adapter | SIGMA（arXiv:2605.27893，10 种 PEFT 横评） | 稠密任务：空间结构 adapter > LoRA/AdaptFormer |
| Full fine-tuning | OWL-ViT 检测路线；Vaani（2603.28714，多语检索） | 同域峰值最高，跨域泛化受损 |
| 领域自适应继续预训练 | MedSigLIP（2% 领域数据混入）；RS-SigLIP2（arXiv:2512.00887，对比+EMA 自蒸馏） | 保通用能力 + 获领域敏感，可作 DG 前置步骤 |

### 直接对比过微调策略的关键消融

| 工作 | 对比 | 结论 |
|---|---|---|
| NR-IQA（2509.17374） | frozen / LoRA / full FT（6 个 backbone 含 SigLIP2-SO400M） | frozen 掉 3–20 SRCC；LoRA ≈ full FT |
| SIGMA（2605.27893） | 10 种 PEFT + full + frozen（SigLIP2-B 等） | 稠密任务 frozen 不可行；LoRA 弱于空间 adapter |
| VLA（CVPR 2026） | LoRA / prompt / FTM / FLA | 特征调制 4.7M ≈ LoRA 467M |
| MedGemma MSK（2511.05600） | head-only / 解冻末 K∈{2,3,4} 层 | 选择性解冻 > head-only |
| ReID 横评（2601.20598） | zero-shot SigLIP2 vs 微调 CLIP-ReID | 微调赢域内（MSMT 66% mAP），zero-shot SigLIP2 赢跨域/在野 |

---

## 4. ReID / 细粒度检索领域的现有实践

**没有"SigLIP2-ReID"正式工作**。现有行人相关工作全部是冻结或 LoRA：

| 工作 | 主干 | 微调策略 | 输入 | 备注 |
|---|---|---|---|---|
| AutoReID（ACL Findings 2026） | SigLIP2-B/16-224 | 全冻 + LoRA r=16（attn），lr 1e-5 | 256×128 | 文本驱动 ReID |
| VLM-PAR（arXiv:2512.22217） | SigLIP2 双塔 | 全冻，只训 cross-attn 融合头 | 默认 | 行人属性识别，跨域强 |
| MoiiAi 横评（arXiv:2601.20598） | SigLIP2-256/384 | 零样本 | 256/384 | 384 vs 256 仅 +1% mAP 但算力翻倍 |
| CLIP-ReID（AAAI'23，参照系） | CLIP ViT-B/16 | 两阶段：冻主干训 prompt → 全量微调图像编码器 | 256×128 | ID CE + Triplet；文本塔推理时不用 |
| CLIP-FGDI/DFGS（DG-ReID SOTA 系） | CLIP | 三阶段（warm-up → prompt → 全量微调） | 256×128 | →M 79.4/91.3 |
| MoDA（TOMM 2025） | CLIP | adapter 插入，主干不更新 | — | PEFT 版 MoE，DG-ReID |
| VICP（arXiv:2508.21222） | DINOv2 | 主干基本冻 + VPT（N=32 tokens） | — | 跨域 object ReID |

---

## 5. 对本项目（DG-ReID，P1+P2 协议）的启示与选型建议

### 5.1 微调策略选型

1. **Baseline 必做**：冻结 SigLIP2 提特征 + BNNeck/轻量 head（官方默认姿态，零成本锚点）
2. **主推路线**：**LoRA（rank 4–16，注入 q/k/v）+ 分层学习率**，而非照搬 CLIP-ReID Stage-2 全量微调
   - 依据：DG 关心预训练语义保留；CLIP-ReID 全量微调被指出灾难性遗忘语义先验（2601.20598 Sec 7）；伪造定位实验（2511.20722）显示 e2e 微调跨域泛化反而变差
3. **低成本对照组**：特征调制（FLA/FTM，千级~百万级参数）或选择性解冻末 2–4 层
4. **泛化保护叠加**：WiSE-FT 权重插值（α≈0.5）、L2-SP 正则
5. **进阶选项**：领域自适应继续预训练（MedSigLIP 式 2% 数据混入 / RS-SigLIP2 式对比+EMA 自蒸馏），可用我们的 caption 数据做行人领域继续训练

### 5.2 工程改造点（相对现有 CLIP 代码）

1. **池化结构**：SigLIP2 无 CLS token，是 MAP attention pooling——`model/maple/clip/model.py` 中取 CLS 的逻辑需改为 MAP head 输出或 patch mean-pool（见 02_visual_encoder.md / 04_output_heads.md）
2. **文本侧**：sigmoid loss 无 logit scale 语义，caption prompt learning 两阶段（CLIP-ReID 式）需重新设计 Stage-1 对比目标
3. **分辨率**：256×128 长条输入 → 优先考虑 **NaFlex 变体**（NaViT+FlexiViT，保宽高比、变长序列），或 pos-emb PI-resize（FlexiViT 伪逆法）；不要强行拉伸方形 checkpoint
4. **架构兼容性**：SigLIP2 与 SigLIP 同构（官方 backward compatible），timm 已有注册（check_env 已验证 12 个 dinov3/siglip 相关条目）

### 5.3 论文卖点（空白点）

- 没有人系统做过 SigLIP2 的 ReID 全量 vs PEFT 消融
- 没有人把 CLIP-ReID 两阶段 prompt learning 迁移到 SigLIP2（sigmoid loss 下的重新设计）
- 没有人在 ReID 上验证 NaFlex 变分辨率输入
- 没有 SigLIP2 在 DG-ReID P1/P2 标准协议上的任何数字

---

## 6. DINOv3 备注（对比项）

- 零样本 ReID 弱（DINOv2 仅 0.3–4.7% mAP）；DINOv3 官方立场同为 "frozen first"（arXiv:2508.10104），冻结 + 线性探针/浅 adapter 即可超专门微调管线
- 有独立证据（arXiv:2605.26383）：预训练语料更大的编码器不一定迁移更好，DINOv3 相对 DINOv2 在实例级任务上未必更优
- 更适合作为**部件级/局部特征提取器**与语义主干互补，而非单独替换 CLIP
- HF 权重需门控申请（被拒后同账号不可重申，见 git 命令笔记旁的实操记录）

---

## 7. 未查到 / 存疑项（诚实声明）

- 未查到 SigLIP2 在 person/vehicle ReID 上全量微调的公开 LR、层冻结粒度经验值
- 未查到 SigLIP2 NaFlex 在任何检索类任务上的实测
- 未查到分类任务（VTAB 类）上 SigLIP2 全 PEFT 谱系统对比
- SIGMA（2605.27893）、ZooClaw（2606.27708）、Vaani（2603.28714）、ReText（2602.05785）、MUSE（2606.16161）为 2026 arXiv 预印本，未经同行评审，引用需谨慎
- 任务初期怀疑的 ADCA / DART / CDNav / ChatAnything 经核实**不是 DG-ReID 方法**（前两个是 VI-ReID），不要误引

## 8. 主要来源

- SigLIP2：arXiv:2502.14786（TMLR 2025）；big_vision README_siglip2；HF blog huggingface.co/blog/siglip2
- 策略对比：arXiv:2509.17374（NR-IQA）、arXiv:2605.27893（SIGMA）、CVPR 2026 VLA-FTM/FLA、arXiv:2511.05600（MedGemma MSK）
- ReID：arXiv:2601.20598（MoiiAi 横评）、ACL-F 2026 AutoReID、arXiv:2512.22217（VLM-PAR）、arXiv:2508.21222（VICP）、TOMM 2025 MoDA、AAAI'23 CLIP-ReID
- 背景：arXiv:2506.12413（DG-ReID 综述）、arXiv:2508.10104（DINOv3）、arXiv:2410.16512（TIPS 稠密探针协议）
