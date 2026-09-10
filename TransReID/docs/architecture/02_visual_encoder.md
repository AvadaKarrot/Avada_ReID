# 02 视觉编码器：(B,3,256,128) → token → CLS 特征

> 历史资料：下文保留旧 CLIP/MaPLe 实现、早期协议规划或调研结论，不代表当前 NaFlex。请先阅读 [当前主线指南](07_naflex_current.md)。论文结论及数值需重新核对原文，旧路径与行号可能失效。

主线：`model/make_model_caption.py` + `model/maple/clip/model.py`（`VisionTransformer_MaPLe`）。

## 0. 模型选择

- 配置 `MODEL.NAME: 'ViT-B-16'`（configs/*/vit_caption_maple.yml）→ CLIP ViT-B/16：embed width 768，12 层 12 头，proj 输出 512。
- 加载：`make_model_caption.py:192-214 load_clip_to_cpu_maple()` → `clip.build_model(state_dict, design_details, h_res, w_res, stride)`（model.py:734）；`trainer='MaPLe'` 时实例化 `VisionTransformer_MaPLe`（model.py:565-576）。

## 1. patch 化与序列构造（model.py:484-506）

```python
x = self.conv1(x)                          # Conv2d(3→768, k=16, s=16)： (B,3,256,128)→(B,768,16,8)
x = x.reshape(B,768,-1).permute(0,2,1)     # (B,128,768)   16×8=128 个 patch
x = cat([class_embedding + zeros(B,1,768), x], 1)   # (B,129,768)
x = x + self.positional_embedding          # (129,768) 广播
visual_ctx = shared_ctx.expand(B,-1,-1)    # (B,3,768)，来自 prompt_learner.proj(ctx)
x = cat([x, visual_ctx], 1)                # (B,132,768)  ← 浅层 MaPLe prompt
x = self.ln_pre(x).permute(1,0,2)          # (132,B,768)
```

- grid：(256−16)/16+1 = 16，(128−16)/16+1 = 8 → 128 patch + 1 cls = 129；+3 prompt = **132** 进 transformer。
- `STRIDE_SIZE: [16,16]`（无重叠 patch）。

## 2. 位置编码：非方形输入的插值适配

原始 ViT-B/16 的 pos_embed 是 224×224 的 `(197,768)`。`build_model` 里 `resize_pos_embed`（model.py:716-730）：去 cls 位 → reshape 14×14 → `F.interpolate(size=(16,8), bilinear)` → 拼回 cls → `(129,768)`，写回 state_dict 加载（model.py:790）。

> 迁移要点：任何新 backbone / 新输入尺寸都要处理这一步。

## 3. MaPLe compound prompts 注入方式

**逐层替换 prompt token（concat 到序列尾部），不是加偏置。**

- 浅层（第 1 层前）：`shared_ctx = proj(ctx)`，(3,768)，拼到序列尾部 → 132。
- 深层（第 2~12 层）：`deep_compound_prompts_vision = [proj_i(compound_prompts_text[i])]`，**11 组** (3,768)（`PROMPT_DEPTH=12`，make_model_caption.py:308,346-351）。
- 每个 `ResidualAttentionBlock_MaPLe`（model.py:305-349）：第 1 层（i=0）不注入；其余层砍掉上一层 prompt 的输出（`prefix = x[:L-3]`），取 `compound_prompts_deeper[counter]` expand 成 (3,B,768) 拼回尾部，counter+1。
- 前向：`x11 = resblocks[:11]([x, prompts, 0]); x12 = resblocks[11](x11)`（model.py:513-514，CLIP-ReID 式两段输出）。

## 4. 输出与特征提取

`VisionTransformer_MaPLe.forward` 返回三元组（model.py:518-526）：

| 输出 | 含义 | 形状 |
|---|---|---|
| `x11` | 前 11 层输出（未过 ln_post） | (B,132,768) |
| `x12` | 12 层 + ln_post | (B,132,768) |
| `xproj` | `x12 @ proj`（768→512） | (B,132,512) |

上层取 CLS token（make_model_caption.py:645-651）：
- `img_feature_last = x11[:,0]` → (B,768)
- `img_feature = x12[:,0]` → (B,768)（测试 neck_feat='before' 输出它）
- `img_feature_proj = xproj[:,0]` → (B,512)（与文本特征跨模态对齐/融合用）

## 5. 冻结/可训练

- `solver/make_optimizer.py:12-18`：`text_encoder.*` 与 **`image_encoder.*` 全部冻结**（含 pos_embed）。
- 可训练：prompt_learner（ctx、proj、compound_prompts_text×11、compound_prompt_projections×11）、cross_attn_text(_proj)、text_projectoin、classifier、bottleneck。

## 附：另一条路线 build_transformer_maple.py

视觉骨干用 timm 版 `TransReid_Prompts`（`modeling/backbones/vit_pytorch.py:656`）：`PatchEmbed_overlap`（支持重叠 patch）+ pos_embed (1,129,768)，MaPLe 注入在 `forward_features`（vit_pytorch.py:744-773）与 `Block_MAPLE`（vit_pytorch.py:243-285），机制相同。区别：**ViT 主干未冻结**；`load_clip_to_cpu` 硬编码 ViT-L-14，仅为 prompt learner 提供文本侧。
