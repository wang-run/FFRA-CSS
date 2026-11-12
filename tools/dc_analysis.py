"""Distribution compactness (DC) analysis of band redundancy (Sections 3.2.1 and 4.2, Fig. 5).

Steps:
  1. Sample 50 patches from the ARAD-1K training split: a random 128x128 crop of a randomly
     chosen training image, then its central 32x32 pixels (D = 1024 pixels).
  2. Least-squares self-representation with L2 regularization (Eq. 1), closed form with diag(Z) = 0.
  3. Squared angular similarity between coefficient columns (Eq. 2); average over the patches.
  4. Eigen-decomposition and DC components d_i (Eq. 3); log + mean filter of length 3 (Eq. 4).
  5. Global baseline B(k), deviation R(k), cumulative mean deviation G(k) (Eq. 5).

    python tools/dc_analysis.py --data_root /path/to/ARAD_1K --out_dir results/dc
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsi_data import load_cube, split_files  # noqa: E402


def similarity_matrix(hsi, beta=0.1, eps=1e-10):
    """hsi: [N_bands, H, W] tensor -> squared angular similarity W, shape [N, N] (Eqs. 1-2)."""
    n, h, w = hsi.shape
    y = hsi.reshape(n, h * w).t()                           # Y in R^{D x N}
    p = torch.inverse(y.t() @ y + beta * torch.eye(n, dtype=y.dtype))
    z = -p / torch.diag(p).unsqueeze(0)                      # closed-form LSR solution
    z.fill_diagonal_(0)                                      # diag(Z) = 0
    z = z / (torch.norm(z, p=2, dim=0, keepdim=True) + eps)
    return ((z.t() @ z) ** 2).numpy()


def dc_curves(w_avg, window=3, eps=1e-10):
    """Eqs. (3)-(5): returns k, d_k, f(k), B(k), R(k), G(k)."""
    n = w_avg.shape[0]
    eigval, eigvec = np.linalg.eigh(w_avg)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    d = eigval * (np.ones(n) @ eigvec) ** 2                  # d_i = lambda_i (1^T u_i)^2
    f = uniform_filter1d(np.log(d + eps), size=window)       # log DC curve f(k)
    k = np.arange(1, n + 1)
    baseline = f[0] + (k - 1) / (n - 1) * (f[-1] - f[0])     # B(k)
    r = np.maximum(baseline - f, 0.0)                        # R(k)
    g = np.cumsum(r) / k                                     # G(k)
    return k, d, f, baseline, r, g


def get_args():
    p = argparse.ArgumentParser(description="DC analysis of band redundancy")
    p.add_argument("--data_root", required=True, help="ARAD-1K root containing Train_Spec/")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--center", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/dc")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = split_files(args.data_root, "arad", "train")
    # same sampling as the training loader: 25 virtual patches per image
    indices = random.sample(range(len(files) * 25), args.num_samples)
    w_sum = None
    for i, idx in enumerate(indices, 1):
        cube = load_cube(files[idx % len(files)])
        _, h, w = cube.shape
        y = torch.randint(0, max(1, h - args.crop + 1), (1,)).item()
        x = torch.randint(0, max(1, w - args.crop + 1), (1,)).item()
        patch = cube[:, y:y + args.crop, x:x + args.crop]
        c0 = args.crop // 2 - args.center // 2
        patch = patch[:, c0:c0 + args.center, c0:c0 + args.center].contiguous()
        sim = similarity_matrix(patch, beta=args.beta)
        w_sum = sim if w_sum is None else w_sum + sim
        print(f"\r{i}/{args.num_samples}", end="", flush=True)
    print()

    k, d, f, baseline, r, g = dc_curves(w_sum / args.num_samples, window=args.window)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "dc_curve.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["k", "dc_component", "log_dc_smoothed", "baseline", "deviation_R", "G"])
        for row in zip(k, d, f, baseline, r, g):
            wr.writerow([int(row[0])] + [f"{v:.10g}" for v in row[1:]])
    print("G(k), k = 1..6:", np.round(g[:6], 4))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
