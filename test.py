"""Evaluate FFRA-CSS (online CSS) or a fixed-CSS reconstruction network on a test split.

Online FFRA-CSS model (ASIModules + CASF-SST):
    python test.py --mode online --checkpoint best_ema.pth --data_root /path/to/ARAD_1K --dataset arad

Fixed optimized CSS + another reconstruction network (Section 4.4.2):
    python test.py --mode fixed_css --model mprnet --css css/optimized_css_4ch.npy \
        --checkpoint mprnet_best_ema.pth --data_root /path/to/ARAD_1K --dataset arad

Use --tta none for single-forward inference (Table 2) and --tta four_flip
(original, horizontal flip, vertical flip, 180-degree rotation) otherwise.
"""
import argparse
import csv
import os

import numpy as np
import torch

from hsi_data import FullImageDataset
from metrics import AverageMeter, evaluate
from models import FIXED_CSS_MODELS, build_fixed_css_model, build_online_model, load_weights

FLIPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[2, 3]),
]  # every transform is its own inverse


def get_args():
    p = argparse.ArgumentParser(description="FFRA-CSS evaluation")
    p.add_argument("--mode", default="online", choices=["online", "fixed_css"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="arad", choices=["arad", "cave"])
    p.add_argument("--num_channels", type=int, default=4)
    p.add_argument("--poly_degree", type=int, default=4)
    p.add_argument("--model", default="casfsst_fixed", choices=FIXED_CSS_MODELS,
                   help="reconstruction network for --mode fixed_css")
    p.add_argument("--css", default="css/optimized_css_4ch.npy", help="[K, 31] CSS for --mode fixed_css")
    p.add_argument("--tta", default="four_flip", choices=["four_flip", "none"])
    p.add_argument("--mrae_threshold", type=float, default=0.01)
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--save_css", action="store_true", help="save the per-image CSS of the online model")
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "online":
        model = build_online_model(args.num_channels, args.poly_degree, pretrained_backbone=False)
        css = None
    else:
        model = build_fixed_css_model(args.model, args.num_channels)
        css = torch.from_numpy(np.load(args.css).astype(np.float32)).to(device)   # [K, 31]
        if css.shape != (args.num_channels, 31):
            raise ValueError(f"CSS must have shape [{args.num_channels}, 31], got {tuple(css.shape)}")
    load_weights(model, args.checkpoint, device)
    model.to(device).eval()

    def forward(hsi):
        if css is None:
            return model(hsi)
        msi = torch.einsum("bchw,kc->bkhw", hsi, css)   # Eq. (25) with the fixed CSS
        return model(msi)

    test_set = FullImageDataset(args.data_root, args.dataset, part="test")
    rows, meters, curves = [], {}, []
    for name, target in zip(test_set.names, test_set):
        target = target.unsqueeze(0).to(device)
        if args.tta == "none":
            output = forward(target)
            if css is None:
                curves.append(model.css[0].cpu().numpy())
        else:
            outs = []
            for i, t in enumerate(FLIPS):
                outs.append(t(forward(t(target))))
                if i == 0 and css is None:
                    curves.append(model.css[0].cpu().numpy())
            output = torch.stack(outs).mean(dim=0)
        m = evaluate(output, target, args.mrae_threshold)
        rows.append({"image": name, **m})
        for k, v in m.items():
            meters.setdefault(k, AverageMeter()).update(v)
        print(name, " ".join(f"{k}={v:.6f}" for k, v in m.items()), flush=True)

    mean = {k: v.avg for k, v in meters.items()}
    rows.append({"image": "mean", **mean})
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Mean:", " ".join(f"{k}={v:.6f}" for k, v in mean.items()))
    if args.save_css and curves:
        np.save(os.path.join(args.out_dir, "test_css_per_image.npy"), np.stack(curves))


if __name__ == "__main__":
    main()
