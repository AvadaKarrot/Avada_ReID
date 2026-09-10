# 04 输出侧：encoder 输出 → bottleneck / classifier / 损失

> 历史资料：下文保留旧 CLIP/MaPLe 实现、早期协议规划或调研结论，不代表当前 NaFlex。请先阅读 [当前主线指南](07_naflex_current.md)。论文结论及数值需重新核对原文，旧路径与行号可能失效。

> 主线：`model/make_model_caption.py`（class `build_transformer_caption`，forward L605-713）。
> baseline 对照：`model/make_model_clipreid_base.py`（同一套输出头，无文本侧）。

## 0. Encoder 输出（回顾）

```
x11   (B,132,768)   前 11 层输出，无 ln_post
x12   (B,132,768)   12 层 + ln_post（CLIP 自带 LayerNorm，冻结）
xproj (B,132,512)   x12 @ proj（CLIP 自带 768→512 投影，冻结）
text_features (B,512)  文本侧 ln_final 后取 EOT 位置 @ text_projection（CLIP 自带，冻结）
```

> ⚠️ 文本侧 77→1 取的是 **EOT token**（`textencoder.py:66-67`，`tokenized_prompts.argmax(-1)` 找最大 id=49407），不是 CLS——CLIP 文本是因果 mask，EOT 聚合全句语义。

## 1. 骨架路径（无 cross-attn，= 测试路径 = baseline 完整路径）

```
① 取 CLS：img_feature_last = x11[:,0]  (B,768)
           img_feature      = x12[:,0]  (B,768)
           img_feature_proj = xproj[:,0](B,512)

② BNNeck（仅后两路；BatchNorm1d，bias 冻结，w=1/b=0 初始化）
   feat      = bottleneck(img_feature)          (B,768)
   feat_proj = bottleneck_proj(img_feature_proj)(B,512)

③ Classifier（Linear → num_classes，bias=False，N(0,0.001²) 初始化）
   cls_score      = classifier(feat)            (B,N)
   cls_score_proj = classifier_proj(feat_proj)  (B,N)
```

**训练返回**（L701）：`[cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]`

**测试返回**（L708-713）：
- `TEST.NECK_FEAT='after'`（默认）：`cat([feat, feat_proj])` → (B,1280)，BN 后
- `'before'`：`cat([img_feature, img_feature_proj])` → (B,1280)，BN 前

## 2. 各特征去向总账

| 特征 | 形状 | 经过的头 | 训练损失 | 测试用 |
|---|---|---|---|---|
| `img_feature_last` | (B,768) | 无 | Triplet | ❌ |
| `img_feature` | (B,768) | （BN 前） | Triplet | NECK_FEAT='before' |
| `img_feature_proj` | (B,512) | （BN 前） | Triplet | NECK_FEAT='before' |
| `feat` (→cls_score) | (B,768) | bottleneck → classifier | ID（label-smooth CE） | NECK_FEAT='after' |
| `feat_proj` (→cls_score_proj) | (B,512) | bottleneck_proj → classifier_proj | ID（CE） | NECK_FEAT='after' |
| `text_features` | (B,512) | text_projectoin（可选，默认关） | **无直接损失** | ❌ |

损失组合（`loss/make_loss_clipreid.py:38-59`，processor_caption.py:85-86 消费）：

```
loss = ID_LOSS_WEIGHT  × Σ CE_labelsmooth(score_i)      # 2 个 logits 头求和
     + TRIPLET_LOSS_WEIGHT × Σ Triplet(feat_i, margin=0.3)  # 3 个特征求和
# i2tscore 接口存在但 processor 从不传 → i2t 分支死代码
# CLIP 对比损失 logit_scale*exp() * img @ text.T 被整段注释（make_model_caption.py:681-684）
```

三条规律：
1. **两空间**：768 = CLIP 视觉原生空间；512 = CLIP 图文共享投影空间，各配一套 BN+classifier；
2. **两特征**：ID loss 永远吃 BN 后，triplet 永远吃 BN 前（BNNeck 解耦）；
3. **训练/测试不对称**：推理只留 768‖512 拼接的 1280 维，文本/融合/img_feature_last 全部消失。

## 3. cross-attn 融合（仅训练时，L676-679）

```python
img_feature = image_features + cross_attn_text(image_features, text_features)  # L676
img_feature = img_feature[:,0]                                                 # L677
img_feature_proj = img_feature_proj + cross_attn_text_proj(img_feature_proj, text_features)  # L679
```

机制（`make_model_caption.py:58-136`）：**图像做 Q，文本做 K/V**；
`Cross_Attention_text` 内含 `test_proj` Linear(512→768) 给文本升维。

⚠️ 三个已确认的设计细节：

1. **注意力退化**：text_features 融合前已压成单个 EOT 向量，K/V 序列长度=1 → softmax 恒为 1 → 输出与 Q 无关，**等价于每个视觉 token += MLP(text)**（FiLM 式全局条件注入）。`qq` 投影梯度为零，白算。
2. **两支路不对称**：768 支路融合用全部 132 token 做 Q（L650 先取的 CLS 被 L676 覆盖丢弃），融合后再取 CLS；512 支路是先取 CLS 再融合。
3. **推理时不执行**：融合包在 `if self.training:` 内——caption 是训练期正则/监督信号，部署时模型是纯视觉的。

## 4. Projection 归属盘点

| 投影 | 维度 | 来源 | 状态 |
|---|---|---|---|
| text_projection | 512→512 | CLIP 自带 | 冻结 |
| visual proj（xproj） | 768→512 | CLIP 自带 | 冻结 |
| prompt 投影 proj + 11 组 compound proj | 512→768 | MaPLe 新增（输入侧） | 可学习 |
| test_proj（cross_attn_text 内） | 512→768 | 新增 | 可学习 |
| text_projectoin（ProjectionHead 残差 MLP+LN） | 512→512 | 新增 | 可学习，`MODEL.TEXT_PROJ` 默认 **False** |
| bottleneck ×2 / classifier ×2 | — / →N | BNNeck 套路（继承 CLIP-ReID baseline） | 可学习 |

## 5. 与 CLIP-ReID 原始设计的关系

- 原始 CLIP-ReID 两阶段：Stage-1 冻图像编码器 + i2t/t2i 对比损失训 prompt learner；Stage-2 冻 prompt 训图像编码器。
- `make_model_clipreid_base.py` 只复现了 **Stage-2 图像端骨架**（prompt 全关、无文本侧、无冻结切换，`logit_scale` 是死参数）。
- 主线 = baseline 双头骨架 + caption + MaPLe prompt + cross-attn 融合的单阶段方案；输出头结构（双 bottleneck/双 classifier/三特征返回）完全继承自 baseline，损失代码也是同一个 `make_loss_clipreid.py`。
- 并行新体系 `objectives/`（ReIDObjective + CaptionAlignmentObjective，带 caption_mask、temperature 0.07 双向对比）被 `tools/train.py` 使用，**train_caption.py 主线尚未接入**。

## 6. 隐患清单（迁移 DINOv3 时必须处理第 1、2 条）

1. **训练/测试特征不对称**：训练 img_feature 融合文本，测试是纯视觉 CLS；
2. `ID_LOSS_TYPE` 设 arcface/cosface/amsoftmax/circle 时返回值缺 `cls_score_proj` 会 NameError（L694-701），实际只支持 softmax；
3. cross-attn 名义实现，实际退化为广播加法（见 3.1）；768/512 支路融合粒度不对称（见 3.2）；
4. `Cross_Attention_text_Block.norm1` 定义未使用；`gap`（AdaptiveAvgPool2d）定义未使用；`loss_func` 的 `target_cam` 参数传入未使用；
5. `TEXT_PROJ` 默认 False，text_projectoin 默认不建，实验配置需显式确认。

## 7. 优化方向（fusion 改造候选）

1. **简化**：cross-attn 换成 FiLM/AdaLN 或直接 `img += MLP(text)`，语义等价、省 Q 侧计算；
2. **真 cross-attention**：文本不先取 EOT，用 ln_final 后的完整 (B,77,512) 序列做 K/V（带 padding mask），132↔~10 token 细粒度交互；
3. **部分级对齐**：配合 6 主题 caption 做 patch ↔ 属性词的 token 级对齐（FILIP 风格 late interaction），作为论文第二个贡献点；
4. 输出侧配套：保留 `img_feature_last`（纯视觉中层）作 ablation 对照。

## 8. DINOv3/SigLIP2 迁移要点（输出侧）

- 512 支路整套（proj/bottleneck_proj/classifier_proj/cross_attn_text_proj）是 CLIP 双塔特有产物：DINOv3 无投影空间 → **砍掉变单支路**，或新加可学习投影自建第二空间；
- SigLIP2 有自带文本塔和 sigmoid 对比损失，文本侧可不再借 CLIP——i2t 死代码可真正激活；
- BNNeck（BN 后 ID / BN 前 triplet）与 backbone 无关，直接保留。
