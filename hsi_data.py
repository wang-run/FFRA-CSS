"""ARAD-1K / CAVE data loading.

Each dataset root must contain a ``Train_Spec`` folder with one ``.mat`` (HDF5, MATLAB v7.3)
file per image. Every file stores the hyperspectral cube under the key ``cube``
(31 bands, 400-700 nm, values in [0, 1]).

Splits (Section 4.1):
  * ARAD-1K: the 950 public files are sorted by name, shuffled with ``random.Random(42)``
    and divided into 850 / 50 / 50 (train / validation / test).
  * CAVE: the 32 files are sorted by name, shuffled with ``random.Random(1024)`` and
    divided into 20 / 4 / 8 (train / validation / test).
"""
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

ARAD_SPLIT = (850, 50, 50)
CAVE_SPLIT = (20, 4, 8)
SPLIT_SEEDS = {"arad": 42, "cave": 1024}
SPLITS = {"arad": ARAD_SPLIT, "cave": CAVE_SPLIT}


def split_files(root, dataset, part):
    """Return the sorted-and-shuffled file paths of one split part ('train' / 'val' / 'test')."""
    spec_dir = os.path.join(root, "Train_Spec")
    files = sorted(f for f in os.listdir(spec_dir) if f.endswith(".mat"))
    random.Random(SPLIT_SEEDS[dataset]).shuffle(files)
    n_train, n_val, n_test = SPLITS[dataset]
    ranges = {"train": (0, n_train), "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, n_train + n_val + n_test)}
    start, end = ranges[part]
    return [os.path.join(spec_dir, f) for f in files[start:end]]


def load_cube(path):
    with h5py.File(path, "r") as mat:
        cube = np.float32(np.array(mat["cube"]))
    cube = np.transpose(cube, [0, 2, 1])  # -> [31, H, W]
    return torch.from_numpy(cube).float()


class TrainDataset(Dataset):
    """Random 256x256 crops with random rotation / flips; 25 crops per image per epoch."""

    def __init__(self, root, dataset="arad", crop_size=256, augment=True, patches_per_img=25):
        self.paths = split_files(root, dataset, "train")
        self.crop_size = crop_size
        self.augment = augment
        print(f"Loading {dataset} training set: {len(self.paths)} images")
        self.hypers = [load_cube(p) for p in self.paths]
        self.img_num = len(self.hypers)
        self.length = self.img_num * patches_per_img

    def __getitem__(self, idx):
        hyper = self.hypers[idx % self.img_num]
        _, h, w = hyper.shape
        cs = self.crop_size
        # torch RNG keeps crops different across dataloader workers
        y = torch.randint(0, max(1, h - cs + 1), (1,)).item()
        x = torch.randint(0, max(1, w - cs + 1), (1,)).item()
        patch = hyper[:, y:y + cs, x:x + cs]
        if self.augment:
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0:
                patch = torch.rot90(patch, k=rot, dims=[1, 2])
            if torch.randint(0, 2, (1,)).item() == 1:
                patch = torch.flip(patch, dims=[1])
            if torch.randint(0, 2, (1,)).item() == 1:
                patch = torch.flip(patch, dims=[2])
        return patch.contiguous()

    def __len__(self):
        return self.length


class FullImageDataset(Dataset):
    """Full-size images of the validation or test split."""

    def __init__(self, root, dataset="arad", part="test"):
        self.paths = split_files(root, dataset, part)
        self.names = [os.path.basename(p) for p in self.paths]
        print(f"Loading {dataset} {part} set: {len(self.paths)} images")
        self.hypers = [load_cube(p) for p in self.paths]

    def __getitem__(self, idx):
        return self.hypers[idx]

    def __len__(self):
        return len(self.hypers)
