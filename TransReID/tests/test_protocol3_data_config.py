import unittest
from types import SimpleNamespace

from data.build import imagedata_kwargs


class Protocol3DataConfigTests(unittest.TestCase):
    def test_build_kwargs_forwards_combineall(self):
        cfg = SimpleNamespace(
            DATASETS=SimpleNamespace(
                ROOT_DIR="/datasets",
                SOURCES="market1501,msmt17,cuhksysu",
                TARGETS="cuhk03",
                TRANSFORMS=[],
                COMBINEALL=True,
            ),
            INPUT=SimpleNamespace(
                SIZE_TRAIN=[256, 128],
                PIXEL_MEAN=[0.5, 0.5, 0.5],
                PIXEL_STD=[0.5, 0.5, 0.5],
                RE_PROB=0.5,
                PADDING=10,
                SOBEL_PROB=0.7,
            ),
            SOLVER=SimpleNamespace(IMS_PER_BATCH=64),
            TEST=SimpleNamespace(IMS_PER_BATCH=128),
            DATALOADER=SimpleNamespace(
                NUM_WORKERS=8,
                NUM_INSTANCE=4,
                SAMPLER="RandomIdentitySampler",
            ),
            MODEL=SimpleNamespace(DIST_TRAIN=False),
            OBJECTIVE=SimpleNamespace(
                CAPTION=SimpleNamespace(
                    ENABLED=False,
                    FILE="",
                    SELECTION="random",
                    MISSING_POLICY="error",
                )
            ),
        )

        kwargs = imagedata_kwargs(cfg)

        self.assertTrue(kwargs["combineall"])

if __name__ == "__main__":
    unittest.main()
