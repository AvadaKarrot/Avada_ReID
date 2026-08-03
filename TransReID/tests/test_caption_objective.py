import unittest
from types import SimpleNamespace

import torch
from torch import nn

from objectives.losses.caption_alignment import CaptionAlignmentObjective
from objectives.text_encoders import LegacyCLIPTextEncoder


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

if __name__ == "__main__":
    unittest.main()
