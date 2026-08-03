import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modeling.backbones.dinov3 import DINOv3Adapter
from modeling.backbones.siglip2 import SigLIP2Adapter
from modeling.heads import CLIPReIDParityHead, ReIDHead
from modeling.outputs import BackboneOutput
from modeling.reid_model import ReIDModel
from objectives import ReIDObjective


class FakeEncoder(nn.Module):
    def __init__(self, token_count, hidden_size=32, pool=False):
        super().__init__()
        self.token_count = token_count
        self.pool = pool
        self.config = SimpleNamespace(patch_size=16, hidden_size=hidden_size)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values, **_):
        batch = pixel_values.shape[0]
        tokens = torch.randn(
            batch,
            self.token_count,
            self.config.hidden_size,
            device=pixel_values.device,
        ) + self.anchor
        return SimpleNamespace(
            last_hidden_state=tokens,
            pooler_output=tokens.mean(dim=1) if self.pool else None,
        )


class ModelingContractTest(unittest.TestCase):
    def test_dinov3_register_tokens_are_excluded(self):
        images = torch.randn(2, 3, 32, 16)
        # CLS + four register tokens + two spatial patches.
        adapter = DINOv3Adapter(
            encoder=FakeEncoder(token_count=7),
            register_tokens=4,
        )
        output = adapter.forward_features(images)
        self.assertEqual(output.global_feature.shape, (2, 32))
        self.assertEqual(output.patch_features.shape, (2, 2, 32))
        self.assertEqual(output.spatial_shape, (2, 1))

    def test_siglip2_adapter_is_image_only(self):
        images = torch.randn(2, 3, 32, 16)
        adapter = SigLIP2Adapter(
            encoder=FakeEncoder(token_count=2, pool=True)
        )
        output = adapter.forward_features(images)
        self.assertEqual(output.global_feature.shape, (2, 32))
        self.assertEqual(output.spatial_shape, (2, 1))

    def test_reid_model_and_objective_contract(self):
        images = torch.randn(4, 3, 32, 16)
        adapter = SigLIP2Adapter(
            encoder=FakeEncoder(token_count=2, hidden_size=32, pool=True)
        )
        model = ReIDModel(adapter, ReIDHead(32, 16, num_classes=2))
        model.train()
        outputs = model(images)
        self.assertEqual(outputs.embedding.shape, (4, 16))
        self.assertEqual(outputs.logits.shape, (4, 2))

        objective = ReIDObjective()
        losses = objective(
            outputs,
            {"pids": torch.tensor([0, 0, 1, 1])},
        )
        self.assertTrue(torch.isfinite(losses["total"]))

        model.eval()
        with torch.no_grad():
            inference_outputs = model(images)
        self.assertIsNone(inference_outputs.logits)
        self.assertEqual(inference_outputs.embedding.shape, (4, 16))

    def test_clipreid_parity_head_preserves_legacy_branches(self):
        class FakeCLIPBackbone(nn.Module):
            def forward_features(self, images):
                batch = images.shape[0]
                primary = torch.randn(batch, 768)
                return BackboneOutput(
                    global_feature=primary,
                    auxiliary_features={
                        "last_global": torch.randn(batch, 768),
                        "projected_global": torch.randn(batch, 512),
                    },
                )

        model = ReIDModel(
            FakeCLIPBackbone(),
            CLIPReIDParityHead(768, 512, num_classes=2),
        )
        images = torch.randn(4, 3, 32, 16)
        model.train()
        outputs = model(images)
        self.assertEqual(outputs.embedding.shape, (4, 1280))
        self.assertEqual(len(outputs.id_logits), 2)
        self.assertEqual(len(outputs.metric_features), 3)

        objective = ReIDObjective(label_smoothing=0.1)
        losses = objective(
            outputs,
            {"pids": torch.tensor([0, 0, 1, 1])},
        )
        self.assertTrue(torch.isfinite(losses["total"]))

        model.eval()
        with torch.no_grad():
            inference_outputs = model(images)
        self.assertIsNone(inference_outputs.logits)
        self.assertIsNone(inference_outputs.id_logits)
        self.assertEqual(inference_outputs.embedding.shape, (4, 1280))


if __name__ == "__main__":
    unittest.main()
