import unittest
from unittest.mock import patch

import torch
from torch import nn

from objectives.distributed import GatherLayout
from objectives.losses.batch_hard_triplet import BatchHardTripletLoss
from objectives.losses.caption_alignment import CaptionAlignmentObjective


class FakeTextEncoder(nn.Module):
    def forward(self, captions):
        values = {
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
        }
        return torch.tensor([values[caption] for caption in captions])


class DistributedObjectiveTest(unittest.TestCase):
    def test_caption_uses_local_anchors_and_global_candidates(self):
        objective = CaptionAlignmentObjective(
            image_dim=2,
            text_dim=2,
            text_encoder=FakeTextEncoder(),
            use_projection=False,
            positive_mode="pid",
            gather_across_ranks=True,
        )
        images = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        remote_images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        remote_texts = remote_images.clone()
        layout = GatherLayout((2, 2), local_offset=0, global_size=4)

        gathered_features = [
            (torch.cat((images, remote_images), dim=0), layout),
            (
                torch.cat(
                    (
                        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                        remote_texts,
                    ),
                    dim=0,
                ),
                layout,
            ),
        ]
        with (
            patch(
                "objectives.losses.caption_alignment.distributed_ready",
                return_value=True,
            ),
            patch(
                "objectives.losses.caption_alignment."
                "gather_variable_with_grad",
                side_effect=gathered_features,
            ),
            patch(
                "objectives.losses.caption_alignment.gather_variable",
                return_value=(torch.tensor([0, 1, 0, 1]), layout),
            ),
            patch(
                "objectives.losses.caption_alignment."
                "globally_normalized_local_sum",
                side_effect=lambda values: values.mean(),
            ),
        ):
            loss = objective(
                image_features=images,
                captions=("left", "right"),
                valid_mask=torch.ones(2, dtype=torch.bool),
                pids=torch.tensor([0, 1]),
            )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(images.grad)

    def test_triplet_mines_remote_candidates_without_self_pair(self):
        objective = BatchHardTripletLoss(
            margin=0.3,
            gather_across_ranks=True,
        )
        local = torch.tensor(
            [[0.0, 0.0], [0.2, 0.0]], requires_grad=True
        )
        global_features = torch.tensor(
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.5, 0.0]],
            requires_grad=True,
        )
        layout = GatherLayout((2, 2), local_offset=0, global_size=4)
        with (
            patch(
                "objectives.losses.batch_hard_triplet.distributed_ready",
                return_value=True,
            ),
            patch(
                "objectives.losses.batch_hard_triplet."
                "gather_variable_with_grad",
                return_value=(global_features, layout),
            ),
            patch(
                "objectives.losses.batch_hard_triplet.gather_variable",
                return_value=(torch.tensor([0, 0, 0, 1]), layout),
            ),
            patch(
                "objectives.losses.batch_hard_triplet."
                "globally_normalized_local_sum",
                side_effect=lambda values: values.mean(),
            ),
        ):
            loss = objective(local, torch.tensor([0, 0]))

        # Remote same-PID sample at distance 0.4 must become the hardest
        # positive; the self-pair at distance 0 must not affect mining.
        expected = torch.tensor((0.4 - 0.5 + 0.3 + 0.2 - 0.3 + 0.3) / 2)
        torch.testing.assert_close(loss.detach(), expected)
        loss.backward()
        self.assertIsNotNone(local.grad)


if __name__ == "__main__":
    unittest.main()
