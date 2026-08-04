import unittest
from types import SimpleNamespace

import torch
from torch import nn

from objectives.losses.caption_alignment import CaptionAlignmentObjective
from objectives.reid_objective import ReIDObjective
from objectives.text_encoders import LegacyCLIPTextEncoder, SigLIP2TextEncoder


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, captions):
        rows = []
        for caption in captions:
            value = 1.0 if "one" in caption else -1.0
            rows.append(
                torch.tensor([value, 1.0], device=self.anchor.device)
            )
        return torch.stack(rows) * self.anchor


class CaptionObjectiveTest(unittest.TestCase):
    @staticmethod
    def _objective_with_text_encoder(*, frozen):
        text_encoder = FakeTextEncoder()
        text_encoder.requires_grad_(not frozen)
        caption_objective = CaptionAlignmentObjective(
            image_dim=2,
            text_dim=2,
            text_encoder=text_encoder,
            projection_dim=2,
        )
        return ReIDObjective(
            caption_objective=caption_objective,
            caption_weight=1.0,
        )

    def test_checkpoint_omits_only_frozen_text_tower(self):
        objective = self._objective_with_text_encoder(frozen=True)
        state = objective.state_dict()
        self.assertFalse(
            any(
                key.startswith("caption_objective.text_encoder.")
                for key in state
            )
        )
        self.assertIn("caption_objective.image_projection.weight", state)
        self.assertIn("caption_objective.text_projection.weight", state)

        restored = self._objective_with_text_encoder(frozen=True)
        restored.load_state_dict(state, strict=True)
        self.assertTrue(
            torch.equal(
                restored.caption_objective.image_projection.weight,
                state["caption_objective.image_projection.weight"],
            )
        )

    def test_checkpoint_keeps_trainable_text_tower(self):
        objective = self._objective_with_text_encoder(frozen=False)
        self.assertIn(
            "caption_objective.text_encoder.anchor",
            objective.state_dict(),
        )

    def test_compact_checkpoint_still_rejects_other_missing_keys(self):
        objective = self._objective_with_text_encoder(frozen=True)
        state = objective.state_dict()
        del state["caption_objective.image_projection.weight"]
        with self.assertRaisesRegex(RuntimeError, "image_projection"):
            objective.load_state_dict(state, strict=True)

    def test_same_pid_pairs_are_all_positives(self):
        logits = torch.tensor(
            [
                [10.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        pids = torch.tensor([1, 1, 2])
        positives = pids[:, None].eq(pids[None, :])
        loss = CaptionAlignmentObjective._multi_positive_nce(
            logits, positives
        )
        self.assertLess(loss.item(), 0.001)

    def test_instance_mode_keeps_only_diagonal_positives(self):
        pids = torch.tensor([1, 1, 2])
        positives = CaptionAlignmentObjective._positive_mask(
            pids,
            "instance",
        )
        self.assertTrue(
            torch.equal(
                positives,
                torch.eye(3, dtype=torch.bool),
            )
        )

    def test_pid_mode_keeps_same_identity_positives(self):
        pids = torch.tensor([1, 1, 2])
        positives = CaptionAlignmentObjective._positive_mask(pids, "pid")
        self.assertTrue(positives[0, 1])
        self.assertFalse(positives[0, 2])

    def test_rejects_unknown_positive_mode(self):
        with self.assertRaisesRegex(ValueError, "positive_mode"):
            CaptionAlignmentObjective(
                image_dim=2,
                text_dim=2,
                text_encoder=FakeTextEncoder(),
                positive_mode="unknown",
            )

    def test_caption_alignment_supports_masks_and_backpropagation(self):
        objective = CaptionAlignmentObjective(
            image_dim=2,
            text_dim=2,
            text_encoder=FakeTextEncoder(),
            projection_dim=2,
        )
        images = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            requires_grad=True,
        )
        loss = objective(
            image_features=images,
            captions=("one front", "one rear", "two"),
            valid_mask=torch.tensor([True, True, False]),
            pids=torch.tensor([1, 1, 2]),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(images.grad)

    def test_projection_free_alignment_preserves_native_clip_space(self):
        objective = CaptionAlignmentObjective(
            image_dim=2,
            text_dim=2,
            text_encoder=FakeTextEncoder(),
            use_projection=False,
        )
        self.assertIsInstance(objective.image_projection, nn.Identity)
        self.assertIsInstance(objective.text_projection, nn.Identity)
        images = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            requires_grad=True,
        )
        loss = objective(
            image_features=images,
            captions=("one front", "one rear", "two"),
            valid_mask=torch.ones(3, dtype=torch.bool),
            pids=torch.tensor([1, 1, 2]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(images.grad)

    def test_projection_free_alignment_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal image and text"):
            CaptionAlignmentObjective(
                image_dim=3,
                text_dim=2,
                text_encoder=FakeTextEncoder(),
                use_projection=False,
            )

    def test_legacy_clip_wrapper_retains_only_text_contract(self):
        clip_model = SimpleNamespace(
            token_embedding=nn.Embedding(16, 4),
            positional_embedding=nn.Parameter(torch.zeros(3, 4)),
            transformer=nn.Identity(),
            ln_final=nn.Identity(),
            text_projection=nn.Parameter(torch.randn(4, 2)),
        )

        tokenizer_calls = []

        def tokenizer(captions, *, truncate=False):
            tokenizer_calls.append(
                {"captions": list(captions), "truncate": truncate}
            )
            return torch.tensor([[1, 2, 9] for _ in captions])

        encoder = LegacyCLIPTextEncoder(
            clip_model=clip_model,
            tokenizer=tokenizer,
            trainable=False,
        )
        output = encoder(["a person", "another person"])
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(
            tokenizer_calls,
            [
                {
                    "captions": ["a person", "another person"],
                    "truncate": True,
                }
            ],
        )
        self.assertEqual(encoder.output_dim, 2)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in encoder.parameters())
        )
        encoder.train()
        self.assertFalse(encoder.training)

    def test_siglip2_wrapper_uses_native_pooled_text_space(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def __call__(self, captions, **kwargs):
                self.calls.append((list(captions), kwargs))
                return {
                    "input_ids": torch.ones(len(captions), 4, dtype=torch.long),
                    "attention_mask": torch.ones(
                        len(captions), 4, dtype=torch.long
                    ),
                }

        class FakeSigLIP2TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(
                    hidden_size=6,
                    max_position_embeddings=4,
                )

            def forward(self, input_ids, attention_mask, return_dict=True):
                self.last_attention_mask = attention_mask
                return SimpleNamespace(
                    pooler_output=torch.ones(input_ids.shape[0], 6)
                    * self.anchor
                )

        tokenizer = FakeTokenizer()
        text_model = FakeSigLIP2TextModel()
        encoder = SigLIP2TextEncoder(
            text_model=text_model,
            tokenizer=tokenizer,
            trainable=False,
        )
        output = encoder(["first person", "second person"])
        self.assertEqual(output.shape, (2, 6))
        self.assertEqual(encoder.output_dim, 6)
        self.assertEqual(tokenizer.calls[0][1]["padding"], "max_length")
        self.assertTrue(tokenizer.calls[0][1]["truncation"])
        self.assertEqual(tokenizer.calls[0][1]["max_length"], 4)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in encoder.parameters())
        )
        encoder.train()
        self.assertFalse(encoder.training)

if __name__ == "__main__":
    unittest.main()
