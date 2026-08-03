import os
import tempfile
import unittest
from pathlib import Path

from data.datasets.image.cuhksysu import CUHKSYSU
from data.datasets.image.market1501 import Market1501


class CUHKSYSUDeterminismTest(unittest.TestCase):
    def test_paths_and_pid_labels_are_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "p2_s2_1.jpg",
                "p1_s3_1.jpg",
                "p2_s1_1.jpg",
            ):
                Path(directory, name).touch()

            dataset = CUHKSYSU.__new__(CUHKSYSU)
            records = dataset.process_dir(directory)

        self.assertEqual(
            [os.path.basename(record[0]) for record in records],
            ["p1_s3_1.jpg", "p2_s1_1.jpg", "p2_s2_1.jpg"],
        )
        self.assertEqual([record[1] for record in records], [0, 1, 1])


class Market1501DeterminismTest(unittest.TestCase):
    def test_paths_and_pid_labels_are_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "0002_c2s1_000001_00.jpg",
                "0001_c1s1_000002_00.jpg",
                "0002_c1s1_000003_00.jpg",
            ):
                Path(directory, name).touch()

            dataset = Market1501.__new__(Market1501)
            records = dataset.process_dir(directory, relabel=True)

        self.assertEqual(
            [os.path.basename(record[0]) for record in records],
            [
                "0001_c1s1_000002_00.jpg",
                "0002_c1s1_000003_00.jpg",
                "0002_c2s1_000001_00.jpg",
            ],
        )
        self.assertEqual([record[1] for record in records], [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
