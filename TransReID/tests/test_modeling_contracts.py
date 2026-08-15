import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modeling.backbones.dinov3 import DINOv3Adapter
from modeling.backbones.siglip2 import (
    MAPHead,
    SigLIP2Adapter,
    SigLIP2NaFlexAdapter,
    resize_siglip2_position_embedding,
)
from modeling.heads import (
    CLIPReIDParityHead,
    MultiBranchParityHead,
    ReIDHead,
    SigLIP2NativePoolerHead,
)
from modeling.outputs import BackboneOutput
from modeling.reid_model import ReIDModel
from objectives import ReIDObjective


class FakeEncoder(nn.Module):
    def __init__(self, token_count, hidden_size=32, pool=False):
        super().__init__()
        self.token_count = token_count
        self.pool = pool
        self.config = SimpleNamespace(
            patch_size=16,
            hidden_size=hidden_size,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            layer_norm_eps=1e-6,
        )
        self.anchor = nn.Parameter(torch.zeros(1))
        self.last_pixel_values_shape = None

    def forward(self, pixel_values, **_):
        batch = pixel_values.shape[0]
        self.last_pixel_values_shape = tuple(pixel_values.shape)
        tokens = torch.randn(
            batch,
            self.token_count,
            self.config.hidden_size,
            device=pixel_values.device,
        ) + self.anchor
        return SimpleNamespace(
            last_hidden_state=tokens,
            pooler_output=tokens.mean(dim=1) if self.pool else None,
            hidden_states=(tokens - 2.0, tokens - 1.0, tokens),
        )


class FakeStaticEncoder(nn.Module):
    def __init__(self, hidden_size=1, grid_size=2):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=16,
            hidden_size=hidden_size,
            num_attention_heads=1,
            intermediate_size=hidden_size * 4,
            layer_norm_eps=1e-6,
        )
        embeddings = nn.Module()
        embeddings.position_embedding = nn.Embedding(
            grid_size * grid_size, hidden_size
        )
        embeddings.register_buffer(
            "position_ids",
            torch.arange(grid_size * grid_size).expand((1, -1)),
        )
        self.vision_model = nn.Module()
        self.vision_model.embeddings = embeddings


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
        self.assertEqual(output.pre_norm_global.shape, (2, 32))
        self.assertEqual(output.secondary_global.shape, (2, 32))

    def test_siglip2_adapter_is_image_only(self):
        images = torch.randn(2, 3, 32, 16)
        encoder = FakeEncoder(token_count=2, pool=True)
        adapter = SigLIP2Adapter(encoder=encoder)
        output = adapter.forward_features(images)
        self.assertEqual(output.global_feature.shape, (2, 32))
        self.assertEqual(output.spatial_shape, (2, 1))
        self.assertEqual(encoder.last_pixel_values_shape, (2, 3, 32, 16))
        self.assertEqual(output.pre_norm_global.shape, (2, 32))
        self.assertEqual(output.secondary_global.shape, (2, 32))
        self.assertEqual(
            output.auxiliary_features["alignment_global"].shape,
            (2, 32),
        )
        self.assertNotIn(
            "penultimate_map_global", output.auxiliary_features
        )

    def test_siglip2_penultimate_map_is_explicitly_opt_in(self):
        images = torch.randn(2, 3, 32, 16)
        adapter = SigLIP2Adapter(
            encoder=FakeEncoder(token_count=2, hidden_size=32, pool=True),
            penultimate_map_pooler=True,
        )

        output = adapter.forward_features(images)

        self.assertEqual(
            output.auxiliary_features["penultimate_map_global"].shape,
            (2, 32),
        )

    def test_big_vision_map_head_contract(self):
        pooler = MAPHead(hidden_size=32, num_heads=4)
        tokens = torch.randn(2, 6, 32, requires_grad=True)

        pooled = pooler(tokens)

        self.assertEqual(pooled.shape, (2, 32))
        self.assertEqual(pooler.probe.shape, (1, 1, 32))
        pooled.sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(pooler.probe.grad)

    def test_map_head_excludes_padding_tokens(self):
        torch.manual_seed(7)
        pooler = MAPHead(hidden_size=8, num_heads=2).eval()
        valid = torch.randn(1, 2, 8)
        first = torch.cat((valid, torch.full((1, 2, 8), 1000.0)), dim=1)
        second = torch.cat((valid, torch.full((1, 2, 8), -1000.0)), dim=1)
        mask = torch.tensor([[True, True, False, False]])

        torch.testing.assert_close(
            pooler(first, mask),
            pooler(second, mask),
        )

    def test_siglip2_naflex_adapter_preserves_mask_contract(self):
        try:
            from transformers import Siglip2VisionConfig, Siglip2VisionModel
        except ImportError:
            self.skipTest("installed transformers has no NaFlex SigLIP2")
        config = Siglip2VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            num_patches=4,
        )
        adapter = SigLIP2NaFlexAdapter(
            encoder=Siglip2VisionModel(config),
            penultimate_map_pooler=True,
        )
        vision_tower = getattr(
            adapter.encoder, "vision_model", adapter.encoder
        )
        self.assertIsNot(adapter.penultimate_map_head, vision_tower.head)
        native_parameters = dict(vision_tower.head.named_parameters())
        penultimate_parameters = dict(
            adapter.penultimate_map_head.named_parameters()
        )
        self.assertEqual(
            native_parameters.keys(), penultimate_parameters.keys()
        )
        for name in native_parameters:
            self.assertIsNot(
                native_parameters[name], penultimate_parameters[name]
            )
            torch.testing.assert_close(
                native_parameters[name], penultimate_parameters[name]
            )
        image_inputs = {
            "pixel_values": torch.randn(2, 4, 12),
            "pixel_attention_mask": torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 1]]
            ),
            "spatial_shapes": torch.tensor([[2, 1], [2, 2]]),
        }

        output = adapter(image_inputs)

        self.assertEqual(output.global_feature.shape, (2, 8))
        self.assertEqual(output.patch_features.shape, (2, 4, 8))
        self.assertEqual(output.secondary_global.shape, (2, 8))
        self.assertIsNone(output.spatial_shape)
        self.assertTrue(
            torch.equal(
                output.auxiliary_features["patch_attention_mask"],
                image_inputs["pixel_attention_mask"].bool(),
            )
        )
        self.assertTrue(
            torch.equal(
                output.auxiliary_features["spatial_shapes"],
                image_inputs["spatial_shapes"],
            )
        )
        self.assertEqual(
            output.auxiliary_features["penultimate_map_global"].shape,
            (2, 8),
        )

    def test_siglip2_adapter_requires_pretrained_final_pooler(self):
        images = torch.randn(2, 3, 32, 16)
        adapter = SigLIP2Adapter(
            encoder=FakeEncoder(token_count=2, hidden_size=32, pool=False)
        )

        with self.assertRaisesRegex(RuntimeError, "pretrained final-layer"):
            adapter.forward_features(images)

    def test_siglip2_adapter_matches_released_pixel_contract(self):
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("installed transformers has no SigLIP2")
        config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            image_size=32,
        )
        adapter = SigLIP2Adapter(
            encoder=SiglipVisionModel(config),
            penultimate_map_pooler=True,
        )
        vision_tower = getattr(
            adapter.encoder, "vision_model", adapter.encoder
        )
        self.assertIsNot(
            adapter.penultimate_map_head,
            vision_tower.head,
        )
        native_state = vision_tower.head.state_dict()
        penultimate_state = adapter.penultimate_map_head.state_dict()
        self.assertEqual(native_state.keys(), penultimate_state.keys())
        for name in native_state:
            torch.testing.assert_close(
                native_state[name], penultimate_state[name]
            )
        output = adapter(torch.randn(2, 3, 32, 16))
        self.assertEqual(output.patch_features.shape, (2, 2, 32))
        self.assertEqual(output.pre_norm_global.shape, (2, 32))
        self.assertEqual(output.secondary_global.shape, (2, 32))
        self.assertEqual(
            output.auxiliary_features["penultimate_map_global"].shape,
            (2, 32),
        )

    def test_siglip2_static_position_embedding_uses_target_grid(self):
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("installed transformers has no SigLIP2")
        config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            image_size=32,
        )
        encoder = SiglipVisionModel(config).eval()
        adapter = SigLIP2Adapter(
            encoder=copy.deepcopy(encoder),
            static_position_embedding=True,
            target_image_size=(32, 16),
        )
        vision_tower = getattr(
            adapter.encoder, "vision_model", adapter.encoder
        )
        embeddings = vision_tower.embeddings
        self.assertEqual(embeddings.position_embedding.weight.shape, (2, 32))
        self.assertTrue(embeddings.position_embedding.weight.requires_grad)
        images = torch.randn(2, 3, 32, 16)
        output = adapter(images)
        self.assertEqual(output.patch_features.shape, (2, 2, 32))
        output.global_feature.sum().backward()
        self.assertIsNotNone(embeddings.position_embedding.weight.grad)

    def test_siglip2_static_position_embedding_uses_linear_resampling(self):
        encoder = FakeStaticEncoder(hidden_size=1, grid_size=2)
        with torch.no_grad():
            encoder.vision_model.embeddings.position_embedding.weight.copy_(
                torch.tensor([[0.0], [1.0], [2.0], [3.0]])
            )
        adapter = SigLIP2Adapter(
            encoder=encoder,
            static_position_embedding=True,
            target_image_size=(48, 16),
        )
        expected = torch.nn.functional.interpolate(
            torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
            size=(3, 1),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(-1, 1)
        actual = (
            adapter.encoder.vision_model.embeddings.position_embedding.weight
        )
        torch.testing.assert_close(actual, expected)

    def test_siglip2_resize_supports_rectangular_source_and_3d_table(self):
        table = torch.arange(6.0).reshape(1, 6, 1)
        resized = resize_siglip2_position_embedding(
            table,
            target_grid=(4, 1),
            source_grid=(2, 3),
        )
        expected = torch.nn.functional.interpolate(
            table[0].reshape(1, 2, 3, 1).permute(0, 3, 1, 2),
            size=(4, 1),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, 4, 1)
        self.assertEqual(resized.shape, (1, 4, 1))
        torch.testing.assert_close(resized, expected)

    def test_siglip2_native_pooler_head_uses_only_final_map_feature(self):
        backbone_output = BackboneOutput(
            global_feature=torch.randn(4, 32),
            patch_features=torch.randn(4, 2, 32),
            pre_norm_global=torch.randn(4, 32),
            secondary_global=torch.randn(4, 32),
            auxiliary_features={
                "alignment_global": torch.randn(4, 32),
            },
        )
        head = SigLIP2NativePoolerHead(
            input_dim=32,
            pooler_dim=32,
            num_classes=2,
        )
        head.train()
        outputs = head(backbone_output)
        self.assertEqual(outputs.embedding.shape, (4, 32))
        self.assertEqual(len(outputs.id_logits), 1)
        self.assertEqual(
            [feature.shape[1] for feature in outputs.metric_features],
            [32],
        )
        self.assertEqual(outputs.alignment_feature.shape, (4, 32))
        self.assertIs(
            outputs.metric_features[0],
            backbone_output.secondary_global,
        )
        self.assertIs(outputs.raw_feature, backbone_output.secondary_global)
        self.assertTrue(
            torch.equal(
                outputs.id_logits[0],
                head.classifier_pooler(
                    head.bnneck_pooler(backbone_output.secondary_global)
                ),
            )
        )
        self.assertFalse(hasattr(head, "classifier"))
        self.assertFalse(hasattr(head, "bnneck"))
        self.assertFalse(hasattr(head, "secondary_projection"))
        torch.testing.assert_close(
            outputs.embedding,
            backbone_output.secondary_global,
        )

    def test_siglip2_native_pooler_head_adds_only_penultimate_triplet(self):
        final_map = torch.randn(4, 32)
        penultimate_map = torch.randn(4, 32)
        alignment = torch.randn(4, 32)
        backbone_output = BackboneOutput(
            global_feature=torch.randn(4, 32),
            patch_features=torch.randn(4, 2, 32),
            pre_norm_global=torch.randn(4, 32),
            secondary_global=final_map,
            auxiliary_features={
                "alignment_global": alignment,
                "penultimate_map_global": penultimate_map,
            },
        )
        head = SigLIP2NativePoolerHead(
            input_dim=32,
            pooler_dim=32,
            num_classes=2,
            use_penultimate_metric=True,
        )
        head.train()

        outputs = head(backbone_output)

        self.assertEqual(len(outputs.id_logits), 1)
        self.assertEqual(head.metric_dims, (32, 32))
        self.assertEqual(len(outputs.metric_features), 2)
        self.assertIs(outputs.metric_features[0], penultimate_map)
        self.assertIs(outputs.metric_features[1], final_map)
        self.assertIs(outputs.raw_feature, final_map)
        self.assertIs(outputs.alignment_feature, alignment)
        torch.testing.assert_close(outputs.embedding, final_map)

    def test_siglip2_native_pooler_head_requires_penultimate_feature(self):
        backbone_output = BackboneOutput(
            global_feature=torch.randn(4, 32),
            secondary_global=torch.randn(4, 32),
        )
        head = SigLIP2NativePoolerHead(
            input_dim=32,
            pooler_dim=32,
            num_classes=2,
            use_penultimate_metric=True,
        )

        with self.assertRaisesRegex(RuntimeError, "penultimate_map_global"):
            head(backbone_output)

    def test_siglip2_penultimate_contract_has_one_id_and_two_triplets(self):
        class CountingTriplet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = []

            def forward(self, features, _):
                self.features.append(features)
                return features.sum() * 0.0

        class RecordingCaptionObjective(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_features = None

            def forward(self, image_features, **_):
                self.image_features = image_features
                return image_features.sum() * 0.0

        final_map = torch.randn(4, 32)
        penultimate_map = torch.randn(4, 32)
        backbone_output = BackboneOutput(
            global_feature=torch.randn(4, 32),
            secondary_global=final_map,
            auxiliary_features={
                "alignment_global": final_map,
                "penultimate_map_global": penultimate_map,
            },
        )
        head = SigLIP2NativePoolerHead(
            input_dim=32,
            pooler_dim=32,
            num_classes=2,
            use_penultimate_metric=True,
        ).train()
        outputs = head(backbone_output)
        caption_objective = RecordingCaptionObjective()
        objective = ReIDObjective(
            caption_objective=caption_objective,
            caption_weight=0.1,
        )
        triplet = CountingTriplet()
        objective.triplet = triplet

        losses = objective(
            outputs,
            {
                "pids": torch.tensor([0, 0, 1, 1]),
                "captions": ("a", "b", "c", "d"),
                "caption_mask": torch.ones(4, dtype=torch.bool),
            },
        )

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(len(outputs.id_logits), 1)
        self.assertEqual(len(triplet.features), 2)
        self.assertIs(triplet.features[0], penultimate_map)
        self.assertIs(triplet.features[1], final_map)
        self.assertIs(caption_objective.image_features, final_map)

    def test_multibranch_parity_head_matches_clip_loss_contract(self):
        penultimate_map = torch.randn(4, 32)
        backbone_output = BackboneOutput(
            global_feature=torch.randn(4, 32),
            patch_features=torch.randn(4, 2, 32),
            pre_norm_global=torch.randn(4, 32),
            secondary_global=torch.randn(4, 32),
            auxiliary_features={
                "alignment_global": torch.randn(4, 32),
                "penultimate_map_global": penultimate_map,
            },
        )
        head = MultiBranchParityHead(
            input_dim=32,
            secondary_dim=32,
            projected_dim=16,
            num_classes=2,
        )
        head.train()
        outputs = head(backbone_output)
        self.assertEqual(outputs.embedding.shape, (4, 48))
        self.assertEqual(len(outputs.id_logits), 2)
        self.assertEqual(
            [feature.shape[1] for feature in outputs.metric_features],
            [32, 32, 16],
        )
        self.assertEqual(outputs.alignment_feature.shape, (4, 32))
        self.assertEqual(head.alignment_dim, 32)
        self.assertIs(outputs.metric_features[0], penultimate_map)
        self.assertIs(outputs.metric_features[1], backbone_output.global_feature)
        self.assertEqual(outputs.metric_features[2].shape, (4, 16))
        self.assertTrue(
            torch.equal(
                outputs.id_logits[0],
                head.classifier(head.bnneck(backbone_output.global_feature)),
            )
        )
        self.assertTrue(
            torch.equal(
                outputs.id_logits[1],
                head.classifier_proj(
                    head.bnneck_proj(outputs.metric_features[2])
                ),
            )
        )

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
        self.assertEqual(outputs.alignment_feature.shape, (4, 512))
        self.assertEqual(len(outputs.id_logits), 2)
        self.assertEqual(len(outputs.metric_features), 3)

        class CountingTriplet(nn.Module):
            def __init__(self):
                super().__init__()
                self.dimensions = []

            def forward(self, features, _):
                self.dimensions.append(features.shape[1])
                return features.sum() * 0.0

        class RecordingCaptionObjective(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_dimension = None

            def forward(self, image_features, **_):
                self.image_dimension = image_features.shape[1]
                return image_features.sum() * 0.0

        caption_objective = RecordingCaptionObjective()
        objective = ReIDObjective(
            label_smoothing=0.1,
            caption_objective=caption_objective,
            caption_weight=0.1,
        )
        triplet = CountingTriplet()
        objective.triplet = triplet
        losses = objective(
            outputs,
            {
                "pids": torch.tensor([0, 0, 1, 1]),
                "captions": ("a", "b", "c", "d"),
                "caption_mask": torch.ones(4, dtype=torch.bool),
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(triplet.dimensions, [768, 768, 512])
        self.assertEqual(caption_objective.image_dimension, 512)

        model.eval()
        with torch.no_grad():
            inference_outputs = model(images)
        self.assertIsNone(inference_outputs.logits)
        self.assertIsNone(inference_outputs.id_logits)
        self.assertEqual(inference_outputs.embedding.shape, (4, 1280))


if __name__ == "__main__":
    unittest.main()
