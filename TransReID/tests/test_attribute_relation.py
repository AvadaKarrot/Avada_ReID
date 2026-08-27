import unittest

import torch

from objectives.losses import AttributeRelationObjective


def _buckets():
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.2, 0.4, 0.8944272],
        ]
    )
    return {
        "upper_clothing": {
            "prototypes": prototypes,
            "anchor_confidence": torch.tensor([1.0, 0.9, 0.8]),
        }
    }


def _targets(indices):
    distributions = torch.zeros(len(indices), 3)
    distributions[torch.arange(len(indices)), torch.tensor(indices)] = 1.0
    return {
        "distributions": {"upper_clothing": distributions},
        "mask": {"upper_clothing": torch.ones(len(indices), dtype=torch.bool)},
        "quality": torch.ones(len(indices)),
    }


def _batch(size):
    return {
        "pids": torch.arange(size),
        "dataset_ids": tuple(0 for _ in range(size)),
        "camids": torch.zeros(size, dtype=torch.long),
    }


class AttributeRelationObjectiveTest(unittest.TestCase):
    def test_matching_topology_is_small_and_has_image_gradient(self):
        objective = AttributeRelationObjective(
            _buckets(),
            image_dim=3,
            domain_key="global",
            queue_size=16,
            min_code_mass=1.0,
            min_effective_samples=2.0,
            min_active_codes=3,
        )
        prototypes = _buckets()["upper_clothing"]["prototypes"]
        image = prototypes.repeat_interleave(2, dim=0).clone()
        image.requires_grad_(True)
        loss = objective(
            image,
            _targets([0, 0, 1, 1, 2, 2]),
            _batch(6),
        )
        loss.backward()
        self.assertLess(float(loss.detach()), 1e-5)
        self.assertIsNotNone(image.grad)
        self.assertEqual(float(objective.last_metrics["relation_graphs"]), 1.0)
        self.assertEqual(objective.state_dict(), {})
        self.assertEqual(list(objective.parameters()), [])

    def test_queue_supplies_missing_codes_but_current_batch_keeps_gradient(self):
        objective = AttributeRelationObjective(
            _buckets(),
            image_dim=3,
            domain_key="dataset",
            queue_size=16,
            min_code_mass=1.0,
            min_effective_samples=2.0,
            min_active_codes=3,
        )
        prototypes = _buckets()["upper_clothing"]["prototypes"]
        warmup = prototypes[:2].repeat_interleave(2, dim=0)
        first = objective(
            warmup,
            _targets([0, 0, 1, 1]),
            _batch(4),
        )
        self.assertEqual(float(first), 0.0)
        self.assertEqual(float(objective.last_metrics["relation_graphs"]), 0.0)

        current = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 0.995]]
        )
        current.requires_grad_(True)
        second = objective(
            current,
            _targets([2, 2]),
            _batch(2),
        )
        second.backward()
        self.assertEqual(float(objective.last_metrics["relation_graphs"]), 1.0)
        self.assertIsNotNone(current.grad)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
