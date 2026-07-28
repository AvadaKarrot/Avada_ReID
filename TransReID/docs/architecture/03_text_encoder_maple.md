# 03 文本编码器与 MaPLe 耦合：caption → (B,77) → text_features

## 1. BPE tokenize（model/maple/clip/simple_tokenizer.py + clip.py:185-221）

- Vocab：字节级 512 基础符号 + 48,894 条 BPE merge 符号 + SOT/EOT = **49,408**；**SOT=49406，EOT=49407**（最大 id，后面 argmax 取 EOT 的依据）。
- encode（simple_tokenizer.py:121-127）：ftfy 修复 → 压缩空白 → 全小写 → 正则切词 → byte-level BPE 贪心合并。
- tokenize 组装（clip.py:204-219）：
  ```python
  all_tokens = [[SOT] + encode(text) + [EOT] for text in texts]
  result = torch.zeros(len(texts), 77, dtype=torch.long)
  # 超长（>75 个内容 token）：truncate=False 默认直接 RuntimeError
  ```
- 输出：**(B,77) int64**，结构 `[SOT, w1..wk, EOT, 0, 0, ...]`。

> 为什么是 77：CLIP 预训练写死的 context_length，`positional_embedding` 形状 (77,512) 是预训练权重，不能改。任何输入必须恰好 77。

## 2. Prompt 构造：ctx 是"替换"不是"插入"

`model/prompt/promptlearner.py`（caption 版 `CapMultiModalPromptLearner` 同理）：

```python
prompts = [prompt_prefix + " " + caption + "."]     # "a photo of <caption>."
tokenized = clip.tokenize(prompts)                  # (B,77)
embedding = clip_model.token_embedding(tokenized)   # (B,77,512)，width=512

token_prefix = embedding[:, :1, :]           # (B,1,512)   ← SOT
token_suffix = embedding[:, 1+n_ctx:, :]     # (B,73,512)  ← caption+EOT+pad
                                             # "a photo of" 的 3 个位置被切掉
prompts = cat([prefix, ctx, suffix], dim=1)  # 1 + 3 + 73 = 77
```

- 可学习 `ctx` (3,512)（`MAPLE.N_CTX=3`）顶替 "a photo of" 的坑位；`CTX_INIT="a photo of"` 时用这三个词的 embedding 初始化 ctx（make_model_caption.py:319-325），训练后漂离原义。
- 长度约束：caption 最多 **72 个 BPE token**（77 − SOT1 − ctx3 − EOT1）。caption 重建时必须控制句长。

## 3. Text Encoder 前向

- 12 层 `ResidualAttentionBlock_MaPLe`（model.py:367-370，trainer='MaPLe'），文本侧 `text_layer=True` 分支（model.py:331-346）：
  - 第 1 层不注入（ctx 已在输入构造时拼好）；
  - 第 2~12 层：拆掉上一层 ctx 输出，替换为本层 `compound_prompts_text[counter]`： 
    `x = cat([x[:1], textual_context(B,3,512), x[1+3:]], dim=0)`（SOT 之后、caption 之前）。
- `TextEncoder_MaPLE`（make_model_caption.py:417- / model/prompt/textencoder.py:58-67）：
  ```python
  x = ln_final(x)                                     # (B,77,512)
  x = x[arange(B), tokenized_prompts.argmax(-1)] @ text_projection
  # argmax 找 EOT(49407) 下标 → 取 EOT 位置向量 → text_features (B,512)
  ```

## 4. MaPLe 图文耦合（核心设计）

视觉 prompt 不是独立学习的，由文本 prompt 线性投影得到：

- 浅层：`proj = Linear(512,768)`，`proj(ctx)` → (3,768) = `shared_ctx` → 视觉第 1 层输入（拼在 patch token 尾部）。
- 深层：`compound_prompt_projections = _get_clones(Linear(512,768), 11)`，`visual_deep_prompts[i] = proj_i(compound_prompts_text[i])` → 视觉第 2~12 层逐层替换。
- 文本侧 prompt 插在 SOT 之后；视觉侧拼在序列**尾部**。

梯度路径：文本 encoder 整体冻结（requires_grad=False），梯度经 `proj`/`compound_prompt_projections` 从图像分支回传到 ctx / compound_prompts_text。

## 5. 特征使用（主线 make_model_caption.py）

- `text_features` 经可学习 `text_projectoin` 后，通过 **cross_attn_text 融合进图像特征**（make_model_caption.py:676,679：`img_feature = image_features + cross_attn_text(image_features, text_features)`）。
- 标准 CLIP 对比损失（`logit_scale.exp() * img @ text.T`）**被整段注释**（L681-684），实际损失 = ID 分类 + metric loss（processor/processor_caption.py:85-86）。
- `engine/batch.py` 生成 `caption_mask`，空 caption 样本可跳过文本分支。

## 形状变化链

```
caption str → "a photo of " + caption + "."
 → BPE → (B,77) int64  [SOT,ids,EOT,0-pad]
 → token_embedding → (B,77,512)
 → [SOT | ctx×3 | caption | EOT | pad] (B,77,512)
 → + pos_embed(77,512) → 12 层 MaPLe block（2~12 层注入 text deep prompt）
 → ln_final → 取 EOT 位置 @ text_projection → text_features (B,512)

耦合支路: ctx(3,512) --Linear(512,768)--> shared_ctx(3,768) → 视觉第1层
          compound_prompts_text[i](3,512) --proj_i--> (3,768) → 视觉第2~12层
```
