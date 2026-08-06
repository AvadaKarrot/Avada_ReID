# CLIP / SigLIP2 Protocol-2 Architecture and Result Analysis

Date: 2026-08-05

## 1. Scope and comparison rule

This report analyzes only the two implemented model families in detail:

- CLIP ViT-B/16 + `CLIPReIDParityHead`
- SigLIP2 Base Patch16 + `SigLIP2NativePoolerHead`

Other domain-generalizable ReID methods are included only as published
Protocol-2 result references. Their internal architectures are not reproduced
or analyzed here.

The primary comparison for our methods uses the fixed 30th epoch. The
target-domain `model_best` values are retained only as diagnostics because
selecting a checkpoint by target labels is not a strict domain-generalization
protocol.

## 2. Shared experiment contract

| Item | Setting |
|---|---|
| Protocol | Protocol-2 |
| Directions | M+MS+CS->C3; M+CS+C3->MS; MS+CS+C3->M |
| Source data | Official source training splits, `COMBINEALL=False` |
| Target data | Query/gallery images only; no target Caption input |
| Image size | 256 x 128 |
| Train/test batch | 64 / 128 |
| Sampler | RandomIdentitySampler, 4 instances per identity |
| Optimizer | Adam, base LR 5e-6, weight decay 1e-4 |
| ReID loss | label-smoothed ID loss + batch-hard triplet loss |
| ID smoothing / triplet margin | 0.1 / 0.3 |
| Caption loss | source-only PID-aware symmetric multi-positive NCE |
| Caption weight / temperature | 0.1 / 0.07 |
| Caption projection | disabled; align in each pretrained native space |
| Epochs / evaluation | 30 epochs; evaluate and checkpoint every 5 epochs |
| Seed | 1234 |

The CLIP runs use LR milestones `[15, 25]`; the final SigLIP2 Static-native
runs use `[5, 20]`. Therefore their result difference is an end-to-end system
comparison, not a perfectly controlled backbone-only ablation.

## 3. Detailed architecture

### 3.1 Layer and feature flow

| Stage | CLIP ViT-B/16 | SigLIP2 Base Patch16 |
|---|---|---|
| Pretrained visual tower | CLIP ViT-B/16, 12 Transformer blocks | `google/siglip2-base-patch16-224`, 12 Transformer blocks, loaded through its compatible `SiglipVisionModel` contract |
| Patch tokens | 16x16 patches; ReID image gives a 16x8 spatial grid | 16x16 patches; ReID image gives 128 tokens on a 16x8 grid |
| Position embedding | pretrained square grid resized for the ReID grid by the legacy CLIP loader | pretrained 14x14 table resized once to 16x8 by bicubic interpolation; static table used during every forward |
| First metric branch | legacy visual API `last_tokens[:,0]`, historically the pre-final/x11 global token, 768-D | mean of final-block tokens captured before post-layernorm, 768-D |
| Second metric branch | final visual token `tokens[:,0]`, 768-D | mean of final post-layernorm tokens, 768-D |
| Third metric branch | final CLIP visual projection `projected_tokens[:,0]`, 512-D | native pretrained attention `pooler_output`, 768-D |
| Native image-text space | projected CLS, 512-D | attention-pooler output, 768-D |
| Patch-token loss | none in the current global baseline | none in the current global baseline |

The crucial structural difference is that CLIP's three ReID branches cover a
pre-final representation, a final representation, and the pretrained
cross-modal projection. SigLIP2's first two branches are two normalizations of
the same final-stage token tensor; only the attention pooler adds a materially
different aggregation. SigLIP2 therefore has a wider descriptor but not
necessarily more complementary supervision.

### 3.2 Head and loss placement

| Component | CLIP | SigLIP2 |
|---|---|---|
| ID branch 1 | BNNeck(768-D final feature) -> classifier -> CE | BNNeck(768-D final-token mean) -> classifier -> CE |
| ID branch 2 | BNNeck(512-D projected feature) -> classifier -> CE | BNNeck(768-D native pooler) -> classifier -> CE |
| Triplet branch 1 | pre-final/x11 feature, 768-D | final pre-norm mean, 768-D |
| Triplet branch 2 | final/x12 feature, 768-D | final post-norm mean, 768-D |
| Triplet branch 3 | native projected feature, 512-D | native attention pooler, 768-D |
| Inference descriptor | final 768 + projected 512 = 1280-D | final mean 768 + pooler 768 = 1536-D |
| Caption alignment | projected image CLS 512 <-> projected text EOT 512 | image pooler 768 <-> text pooler 768 |
| Total loss | sum(2 CE) + sum(3 triplet) + 0.1 caption loss | same loss topology |

The Caption loss is not an instance-only diagonal loss. For each source batch,
all image/text pairs sharing the same PID are positives. It is symmetric:
image-to-text and text-to-image multi-positive NCE are averaged.

### 3.3 Trainable and frozen modules

| Module | Image-only | Caption alignment |
|---|---|---|
| Visual Transformer | trainable end-to-end | trainable end-to-end |
| BNNecks and ID classifiers | trainable | trainable |
| Text Transformer | not constructed/used | frozen (`requires_grad=False`, forced eval mode) |
| Caption projector | absent | absent in the formal experiments |
| Target-domain inference | visual tower + ReID head only | visual tower + ReID head only |

Thus neither CLIP nor SigLIP2 is a frozen-image-encoder experiment. Only the
text tower is frozen. The visual tower receives gradients from all enabled ID,
triplet, and Caption losses.

CLIP tokenization is truncated to its 77-token context. SigLIP2 uses a
64-token fixed context with truncation. The full 204,531-caption audit found
220 SigLIP2 captions above 64 tokens (0.1076%); Market contained none.

## 4. Protocol-2 published reference and our fixed-epoch results

All values are percentages. Published reference values are transcribed from
the ReNorm Protocol-2 comparison table. Our rows use fixed epoch 30.

| Method | Venue / variant | M+MS+CS->C3 mAP / R1 | M+CS+C3->MS mAP / R1 | MS+CS+C3->M mAP / R1 | Average mAP / R1 |
|---|---|---:|---:|---:|---:|
| SNR | CVPR 2020 | 8.9 / 8.9 | 6.8 / 19.9 | 34.6 / 62.7 | 16.8 / 30.5 |
| ISR | ICCV 2023 | 27.4 / 26.1 | 21.2 / 45.7 | 65.1 / 85.1 | 37.9 / 52.3 |
| M^3L | CVPR 2021 | 34.2 / 34.4 | 16.7 / 37.5 | 61.5 / 82.3 | 37.5 / 51.4 |
| MetaBIN | CVPR 2021 | 28.8 / 28.1 | 17.8 / 40.2 | 57.9 / 80.1 | 34.8 / 49.5 |
| META | ECCV 2022 | 36.3 / 35.1 | 22.5 / 49.9 | 67.5 / 86.1 | 42.1 / 57.0 |
| ACL | ECCV 2022 | 41.2 / 41.8 | 20.4 / 45.9 | 74.3 / 89.3 | 45.3 / 59.0 |
| BAU | NeurIPS 2024 | 42.8 / 43.9 | 24.3 / 50.9 | 77.1 / 90.4 | 48.1 / 61.7 |
| ReNorm | ECCV 2024 | 43.6 / 44.7 | 25.6 / 55.6 | 72.7 / 89.1 | 47.3 / 63.1 |
| **CLIP** | image-only, epoch 30 | 28.19 / 31.21 | 21.93 / 42.62 | 62.49 / 79.22 | 37.54 / 51.02 |
| **CLIP + Caption** | epoch 30 | **32.59 / 35.21** | **26.06 / 49.82** | **64.34 / 80.37** | **41.00 / 55.13** |
| **SigLIP2 Static-native** | image-only, epoch 30 | 35.46 / 35.64 | 17.89 / 39.14 | 54.42 / 74.02 | 35.92 / 49.60 |
| **SigLIP2 Static-native + Caption** | epoch 30 | **36.60 / 36.43** | **19.93 / 42.89** | **56.49 / 75.74** | **37.67 / 51.69** |

## 5. Caption contribution

| Backbone | Target C3 gain | Target MS gain | Target M gain | Average gain |
|---|---:|---:|---:|---:|
| CLIP | +4.40 mAP / +4.00 R1 | +4.13 / +7.20 | +1.85 / +1.15 | +3.46 / +4.12 |
| SigLIP2 | +1.14 mAP / +0.79 R1 | +2.04 / +3.75 | +2.07 / +1.72 | +1.75 / +2.09 |

The sign is positive in all six backbone/target combinations. This is strong
evidence that source-caption alignment is useful. It is not evidence that the
current visual representation is already optimal: CLIP benefits more from the
same alignment objective, while SigLIP2 remains weaker overall.

## 6. Interpretation

1. **CLIP + Caption is the strongest current implementation.** Its average
   41.00 mAP / 55.13 R1 exceeds SNR, M^3L, MetaBIN and approximately matches
   META in average scale. It remains below ACL, BAU and ReNorm.
2. **The MS target is the most encouraging result.** CLIP + Caption reaches
   26.06 mAP, slightly above ReNorm's reported 25.6, although its R1 remains
   lower (49.82 versus 55.6).
3. **SigLIP2's lower result is not caused by a frozen image tower or a small
   descriptor.** Its image tower is trained and its descriptor is 1536-D.
   The more plausible current bottlenecks are branch redundancy, pooling
   adaptation, and a schedule tuned separately from CLIP.
4. **No token-level cross-attention is present.** Both Caption variants align
   one global image vector to one global text vector. Patch tokens and word
   tokens do not interact directly.
5. **The next clean ablation should equalize optimization.** Run CLIP and
   SigLIP2 with identical LR milestones, augmentation, source sampling, and
   checkpoint rule, then replace SigLIP2's pre/post-norm duplicate branch with
   an intermediate-block feature or token-aware pooling. Only after that is a
   cross-attention fusion layer interpretable.

## 7. Primary references

- ReNorm Protocol-2 table: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08753.pdf
- BAU: https://papers.neurips.cc/paper_files/paper/2024/file/53fba4404ebecf9730dc8919b71d4d22-Paper-Conference.pdf
- ACL: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740209.pdf
- SNR: https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.html
- MetaBIN: https://openaccess.thecvf.com/content/CVPR2021/html/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.html
- M^3L: https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf
- ISR: https://openaccess.thecvf.com/content/ICCV2023/papers/Dou_Identity-Seeking_Self-Supervised_Representation_Learning_for_Generalizable_Person_Re-Identification_ICCV_2023_paper.pdf
- META: https://arxiv.org/abs/2112.08684
