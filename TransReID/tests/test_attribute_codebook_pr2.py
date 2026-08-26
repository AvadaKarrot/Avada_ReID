import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.attribute_store import ATTRIBUTE_BUCKETS
from data.collate import NaFlexCollator
from data.semantic_targets import AttributeSemanticTargetStore
from objectives.losses import AttributeCodebookObjective
from semantic.artifact import save_torch_artifact


def _artifacts(directory):
    phrase_buckets = {}
    code_buckets = {}
    for bucket in ATTRIBUTE_BUCKETS:
        phrase_buckets[bucket] = {
            "phrases": (f"{bucket} value",),
            "embeddings": torch.tensor([[1.0, 0.0]]),
        }
        code_buckets[bucket] = {
            "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "anchor_confidence": torch.tensor([1.0, 0.5]),
        }
    phrase_path = Path(directory) / "phrase_bank.pt"
    code_path = Path(directory) / "codebook.pt"
    save_torch_artifact({"buckets": phrase_buckets}, phrase_path)
    save_torch_artifact(
        {
            "expected_domains": ("market1501:camera:0",),
            "buckets": code_buckets,
        },
        code_path,
    )
    return phrase_path, code_path, code_buckets


def _caption_record():
    attributes = {
        "upper_clothing": {
            "visibility": "clear",
            "attributes": ["upper_clothing value"],
        },
        "lower_clothing": {
            "visibility": "not_visible",
            "attributes": ["lower_clothing value"],
        },
        "footwear": {
            "visibility": "clear",
            "attributes": ["footwear value"],
        },
        "carried_items": ["carried_items value"],
        "accessories": ["accessories value"],
        "hair": ["hair value"],
        "distinctive_features": ["distinctive_features value"],
    }
    return {
        "dataset": "market1501",
        "split": "train",
        "image_path": "bounding_box_train/a.jpg",
        "pid": 1,
        "quality_score": 0.8,
        "attributes": attributes,
    }


class AttributeSemanticTargetTest(unittest.TestCase):
    def test_visibility_and_soft_qt_are_bound_source_only(self):
        with tempfile.TemporaryDirectory() as directory:
            phrase_path, code_path, _ = _artifacts(directory)
            caption_path = Path(directory) / "captions.jsonl"
            caption_path.write_text(
                json.dumps(_caption_record()) + "\n", encoding="utf-8"
            )
            store = AttributeSemanticTargetStore.from_artifacts(
                caption_path,
                phrase_path,
                code_path,
                source_datasets=("market1501",),
                text_temperature=0.1,
            )
            bound = store.bind(
                [("/data/market/bounding_box_train/a.jpg", 1, 2, 0)],
                "market1501",
            )
        target = bound[0][-1]
        self.assertTrue(target.valid["upper_clothing"])
        self.assertFalse(target.valid["lower_clothing"])
        self.assertAlmostEqual(
            float(target.distributions["upper_clothing"].sum()), 1.0
        )
        self.assertGreater(
            target.distributions["upper_clothing"][0],
            target.distributions["upper_clothing"][1],
        )

    def test_naflex_collate_keeps_semantic_target_without_caption(self):
        class Processor:
            def __call__(self, **kwargs):
                return {
                    "pixel_values": torch.randn(1, 1, 2),
                    "pixel_attention_mask": torch.ones(1, 1),
                    "spatial_shapes": torch.ones(1, 2, dtype=torch.long),
                }

        with tempfile.TemporaryDirectory() as directory:
            phrase_path, code_path, _ = _artifacts(directory)
            caption_path = Path(directory) / "captions.jsonl"
            caption_path.write_text(
                json.dumps(_caption_record()) + "\n", encoding="utf-8"
            )
            store = AttributeSemanticTargetStore.from_artifacts(
                caption_path,
                phrase_path,
                code_path,
                source_datasets=("market1501",),
            )
            target = store.get(
                "bounding_box_train/a.jpg", "market1501"
            )
        batch = [(object(), 1, 2, "a.jpg", 0, None, target)]
        result = NaFlexCollator(Processor())(batch)
        self.assertIsNone(result["captions"])
        self.assertEqual(
            result["attribute_targets"]["distributions"][
                "upper_clothing"
            ].shape,
            (1, 2),
        )


class AttributeCodebookObjectiveTest(unittest.TestCase):
    def test_kl_has_image_gradient_and_frozen_artifact_state(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, buckets = _artifacts(directory)
        objective = AttributeCodebookObjective(
            buckets, image_dim=2, image_temperature=0.1
        )
        image = torch.tensor([[0.0, 1.0]], requires_grad=True)
        targets = {
            "distributions": {
                bucket: torch.tensor([[1.0, 0.0]])
                for bucket in ATTRIBUTE_BUCKETS
            },
            "mask": {
                bucket: torch.tensor([bucket == "upper_clothing"])
                for bucket in ATTRIBUTE_BUCKETS
            },
            "quality": torch.tensor([1.0]),
        }
        loss = objective(image, targets)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(image.grad)
        self.assertEqual(objective.state_dict(), {})
        self.assertEqual(list(objective.parameters()), [])


if __name__ == "__main__":
    unittest.main()
