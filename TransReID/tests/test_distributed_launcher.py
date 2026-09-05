import unittest

from utils.distributed_launcher import (
    distributed_overrides,
    validate_gpu_ids,
    wrap_torchrun,
)


class DistributedLauncherTest(unittest.TestCase):
    def test_two_gpu_global_batch_256(self):
        command = wrap_torchrun(
            ["python", "tools/train.py", "--config_file", "recipe.yml"],
            nproc_per_node=2,
        )
        self.assertEqual(command[1:5], [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
        ])
        self.assertEqual(
            distributed_overrides(
                nproc_per_node=2, global_batch_size=256
            ),
            [
                "MODEL.DIST_TRAIN",
                "True",
                "SOLVER.IMS_PER_BATCH",
                "256",
            ],
        )

    def test_rejects_non_divisible_global_batch(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            distributed_overrides(
                nproc_per_node=3, global_batch_size=256
            )

    def test_gpu_list_must_match_world_size(self):
        validate_gpu_ids("0,2", nproc_per_node=2)
        with self.assertRaisesRegex(ValueError, "exposes 1 GPUs"):
            validate_gpu_ids("0", nproc_per_node=2)


if __name__ == "__main__":
    unittest.main()
