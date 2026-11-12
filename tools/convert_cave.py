"""Convert the CAVE multispectral database (31 PNG bands per scene) to the .mat format
read by hsi_data.py: HDF5 file with the cube under key 'cube', values in [0, 1].

Download the per-scene zip files from https://cave.cs.columbia.edu/repository/Multispectral
and extract them so that each scene folder (e.g. balloons_ms/) contains balloons_ms_01.png ...
balloons_ms_31.png (400-700 nm in 10 nm steps).

    python tools/convert_cave.py --src /path/to/cave_png --dst /path/to/CAVE/Train_Spec

Values are divided by 65535 (16-bit PNG). The band images of watercolors_ms are 8-bit;
they are also divided by 65535 here, which is how the data were prepared for the paper
(this scene belongs to the training split).
"""
import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image


def read_band(path):
    band = np.array(Image.open(path))
    if band.ndim == 3:          # a few CAVE bands are stored as RGB(A); use the first channel
        band = band[..., 0]
    return band.astype(np.float32) / 65535.0


def main():
    p = argparse.ArgumentParser(description="Convert CAVE PNG bands to .mat cubes")
    p.add_argument("--src", required=True, help="folder that contains the extracted scene folders")
    p.add_argument("--dst", required=True, help="output folder (use <CAVE root>/Train_Spec)")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    scenes = sorted({os.path.basename(os.path.dirname(f))
                     for f in glob.glob(os.path.join(args.src, "**", "*_ms_01.png"), recursive=True)})
    for scene in scenes:
        files = sorted(glob.glob(os.path.join(args.src, "**", scene, f"{scene}_[0-9][0-9].png"), recursive=True))
        if len(files) != 31:
            print(f"skip {scene}: found {len(files)} bands")
            continue
        cube = np.stack([read_band(f) for f in files], axis=0)            # [31, H, W]
        with h5py.File(os.path.join(args.dst, f"{scene}.mat"), "w") as mat:
            mat.create_dataset("cube", data=np.transpose(cube, [0, 2, 1]))  # MATLAB-style [31, W, H]
        print(f"{scene}: {cube.shape}, max {cube.max():.4f}")


if __name__ == "__main__":
    main()
