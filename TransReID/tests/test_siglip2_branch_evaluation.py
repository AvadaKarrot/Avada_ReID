import unittest

import torch
from torch import nn

from evaluation.siglip2_branches import (
    BRANCH_NAMES,
    extract_siglip2_branch_features,
)
from modeling.heads import MultiBranchParityHead
from modeling.outputs import BackboneOutput
from modeling.reid_model import ReIDModel


class CountingBackbone(nn.Module):
    output_dim = 4
    secondary_dim = 4

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_features(self, images):
        self.calls += 1
        batch = images.shape[0]
        base = torch.arange(
            batch * 4, dtype=images.dtype, device=images.device
        ).reshape(batch, 4)
        return BackboneOutput(
            global_feature=base + 1.0,
            pre_norm_global=base,
            secondary_global=base + 2.0,
        )


class SigLIP2BranchEvaluationTest(unittest.TestCase):
    def test_extracts_all_branches_with_one_backbone_forward(self):
        backbone = CountingBackbone()
        head = MultiBranchParityHead(
            input_dim=4,
            secondary_dim=4,
            projected_dim=2,
            num_classes=3,
            neck_feature="before",
        )
        model = ReIDModel(backbone, head).eval()
        images = torch.randn(3, 3, 16, 16)

        features = extract_siglip2_branch_features(model, images)

        self.assertEqual(backbone.calls, 1)
        self.assertEqual(tuple(features), BRANCH_NAMES)
        self.assertEqual(
            {name: tuple(value.shape) for name, value in features.items()},
            {
                "pre_norm_mean": (3, 4),
                "post_norm_mean": (3, 4),
                "native_pooler": (3, 4),
                "projected_pooler": (3, 2),
                "current_concat": (3, 6),
                "mean_native_concat": (3, 8),
            },
        )
        torch.testing.assert_close(
            features["current_concat"],
            torch.cat(
                (
                    features["post_norm_mean"],
                    features["projected_pooler"],
                ),
                dim=1,
            ),
        )
        torch.testing.assert_close(
            features["mean_native_concat"],
            torch.cat(
                (
                    features["post_norm_mean"],
                    features["native_pooler"],
                ),
                dim=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
