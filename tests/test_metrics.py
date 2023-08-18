import numpy as np
from pathlib import Path
import shutil
import tempfile

import unittest

from stylegan_metrics import dnnlib
from calc_metrics import subprocess_fn


def create_dummy_dataset(f1: str, f2: str, resolution: int) -> None:
    shape = (resolution, resolution, 3)

    # create two folders
    f1 = Path(f1)
    f1.mkdir(parents=True, exist_ok=True)

    f2 = Path(f2)
    f2.mkdir(parents=True, exist_ok=True)

    for i in range(100):
        np.save(f1 / f"{i}.npy", np.random.randint(0, 255, shape).astype(np.uint8))
        np.save(f2 / f"{i}.npy", np.random.randint(0, 255, shape).astype(np.uint8))


class TestMetrics(unittest.TestCase):
    f1 = None
    f2 = None
    opts = None

    @classmethod
    def setUpClass(cls):
        """Set up the test environment. Create dummy dataset for testing and setup options"""

        f1 = "./test1"
        f2 = "./test2"
        run_dir = "./test_result"
        resolution = 256
        create_dummy_dataset(f1, f2, resolution)

        opts = dnnlib.EasyDict(num_gpus=1, verbose=False)
        opts.path_real = f1
        opts.path_fake = f2
        opts.resolution = resolution
        opts.run_dir = run_dir

        cls.opts = opts
        cls.f1 = f1
        cls.f2 = f2

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment. Remove dummy dataset"""

        shutil.rmtree(cls.f1)
        shutil.rmtree(cls.f2)

        if Path("./dnnlib").exists():
            shutil.rmtree("./dnnlib")

    def test_fid(self):
        self.opts.metrics = ["fid"]

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, args=self.opts, temp_dir=temp_dir)

    def test_kid(self):
        self.opts.metrics = ["kid"]

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, args=self.opts, temp_dir=temp_dir)

    def test_is(self):
        self.opts.metrics = ["inception_score"]

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, args=self.opts, temp_dir=temp_dir)

    def test_pr(self):
        self.opts.metrics = ["pr"]

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, args=self.opts, temp_dir=temp_dir)


if __name__ == "__main__":
    unittest.main()
