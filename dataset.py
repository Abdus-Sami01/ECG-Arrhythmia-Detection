import numpy as np

from config import PROCESSED_DIR


def load_dataset(path=None):
    path = path or (PROCESSED_DIR / "mitbih_ds1ds2.npz")
    data = np.load(path)
    return {split: (data[f"x_{split}"], data[f"y_{split}"]) for split in ("train", "val", "test")}
