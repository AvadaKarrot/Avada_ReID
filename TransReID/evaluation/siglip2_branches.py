"""Feature-branch extraction for zero-training-cost SigLIP2 diagnosis."""

from collections import OrderedDict

import torch

from modeling.heads import MultiBranchParityHead


BRANCH_NAMES = (
    "pre_norm_mean",
    "post_norm_mean",
    "native_pooler",
    "projected_pooler",
    "current_concat",
    "mean_native_concat",
)


@torch.no_grad()
def extract_siglip2_branch_features(model, images):
    """Run the backbone once and expose each candidate retrieval feature.

    The returned tensors are intentionally left unnormalized. The standard
    ReID metric applies the same final L2 normalization used by normal target
    evaluation, making the branch results directly comparable.
    """

    if not isinstance(model.head, MultiBranchParityHead):
        raise TypeError(
            "SigLIP2 branch evaluation requires MultiBranchParityHead, got "
            f"{type(model.head).__name__}"
        )

    backbone_output = model.backbone.forward_features(images)
    if backbone_output.pre_norm_global is None:
        raise RuntimeError("Backbone did not provide pre_norm_global")
    if backbone_output.secondary_global is None:
        raise RuntimeError("Backbone did not provide secondary_global")

    head_output = model.head(backbone_output)
    post_norm_mean = backbone_output.global_feature
    native_pooler = backbone_output.secondary_global
    projected_pooler = model.head.secondary_projection(native_pooler)

    features = OrderedDict(
        (
            ("pre_norm_mean", backbone_output.pre_norm_global),
            ("post_norm_mean", post_norm_mean),
            ("native_pooler", native_pooler),
            ("projected_pooler", projected_pooler),
            ("current_concat", head_output.embedding),
            (
                "mean_native_concat",
                torch.cat((post_norm_mean, native_pooler), dim=1),
            ),
        )
    )
    if tuple(features) != BRANCH_NAMES:
        raise RuntimeError("Unexpected SigLIP2 diagnostic branch order")
    return features
