import unittest
from unittest.mock import patch

from data.sampler_ddp import RandomIdentitySampler_DDP


class DistributedSamplerContractTest(unittest.TestCase):
    @patch("data.sampler_ddp.dist.get_rank", return_value=0)
    @patch("data.sampler_ddp.dist.get_world_size", return_value=2)
    def test_accepts_caption_attribute_extended_records(self, *_):
        records = [
            ("a.jpg", 7, 0, 0, "caption", True, {"upper": "dark"}),
            ("b.jpg", 7, 1, 0, "caption", True, {"upper": "dark"}),
            ("c.jpg", 8, 0, 0, "caption", True, {"upper": "light"}),
            ("d.jpg", 8, 1, 0, "caption", True, {"upper": "light"}),
        ]
        sampler = RandomIdentitySampler_DDP(
            records, batch_size=4, num_instances=1
        )
        self.assertEqual(set(sampler.index_dic), {7, 8})


if __name__ == "__main__":
    unittest.main()
