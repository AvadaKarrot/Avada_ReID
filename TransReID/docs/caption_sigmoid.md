# Caption loss variants (codex/sigmoidloss)

This branch starts from codex/naflex-reid at 66f0e61. Default behavior is unchanged:
LOSS_TYPE=nce, POSITIVE_MODE=pid, fixed TEMPERATURE=0.07.
Do not edit the GPU7 running worktree to try these variants.

## Independent configuration switches

| LOSS_TYPE | POSITIVE_MODE | Meaning |
|---|---|---|
| nce | instance | CLIP-style symmetric single-pair NCE |
| nce | pid | Existing Caption-C, log of total same-PID probability |
| sigmoid | instance | Independent pair BCE with diagonal positives |
| sigmoid | pid | Independent pair BCE with all same-PID positives |

With a fixed temperature, the NCE option reproduces the loss form, not the full
CLIP pretraining recipe. Sigmoid is the image-text component, not all SigLIP2
pretraining objectives.

Sigmoid logits are exp(logit_scale) * cosine + logit_bias, computed in FP32.
Initial exp(logit_scale)=1/TEMPERATURE, capped at 100. Default bias=-10.
SIGMOID_LEARNABLE=True makes scale/bias optimizer parameters; False stores buffers.
They are included in checkpoints. Optimizer LR/decay rules remain unchanged,
including BIAS_LR_FACTOR for logit_bias. These choices must be recorded.

## Explicit reductions

For B valid global paired samples and pair BCE ell:
- anchor: sum(ell)/B (SigLIP-style normalization).
- pair_mean: sum(ell)/(B*B).
- balanced: 0.5*mean(positive ell) + 0.5*mean(negative ell).
  If a sign group is absent globally, use the remaining group with weight 1.
  An all-invalid batch returns differentiable zero.

Balanced reweights pair classes; it does NOT guarantee equal gradient magnitude
or equal numerical contribution. It is not the original SigLIP loss.
anchor and pair_mean differ by B; do not compare their raw losses or reuse
Caption weight=0.1 as if their scales were identical. Balanced also changes
the objective. Bias=-10 is configurable, not a verified optimum for ReID.

The shared forward averages image->text and text->image. For sigmoid over a
complete square matrix this does NOT double the loss; each orientation is
weighted 0.5. DDP uses local anchors/global candidates with differentiable gather,
global pair counts for balanced reduction, and DDP-aware anchor normalization.
GATHER_ACROSS_RANKS=False explicitly means rank-local losses, not global parity.
PID labels must be remapped across source datasets; equal raw IDs from different
datasets must not be treated as the same identity.

## Selecting a variant

Use the existing base YAML and append CLI overrides (no duplicate full recipes):

    OBJECTIVE.CAPTION.LOSS_TYPE sigmoid
    OBJECTIVE.CAPTION.POSITIVE_MODE pid
    OBJECTIVE.CAPTION.SIGMOID_REDUCTION balanced
    OBJECTIVE.CAPTION.SIGMOID_BIAS -10.0
    OBJECTIVE.CAPTION.SIGMOID_LEARNABLE True

For instance sigmoid change POSITIVE_MODE to instance.
For SigLIP-style normalization change SIGMOID_REDUCTION to anchor.
For original Caption-C use LOSS_TYPE nce and POSITIVE_MODE pid.
NCE ignores sigmoid settings. TEXT_ENCODER selects features, not the loss family.

build_caption_objective in objectives/build.py selects the class;
reid_objective.py still only invokes the Caption objective and weights it.
ID, Triplet, Codebook, Relation and image-only evaluation are unchanged.
Use new output directories per loss/reduction. Do not resume a NCE optimizer
checkpoint into sigmoid: sigmoid adds scale/bias state. Start each ablation
from the same image pretrained weights and keep source, seed, batch and schedule
matched. Loss weights and scalar settings require explicit control.

## CPU validation

    CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p 'test_caption_sigmoid*.py' -v
    CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p 'test_distributed_objectives.py' -v

Tests include BCE formula, NCE equivalence, masks, extreme logits, missing sign
groups, checkpoint state, and two-rank Gloo value/gradient equivalence with
unequal local batches and an empty-caption rank. No GPU training is launched.
