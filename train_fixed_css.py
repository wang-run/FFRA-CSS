"""Train a reconstruction network with a fixed CSS (Section 4.4.2).

The CSS is frozen; the K-channel observation is synthesized from each HSI patch
(Eq. 25) and the network is trained with the MRAE loss.

    python train_fixed_css.py --model mprnet --css css/optimized_css_4ch.npy \
        --data_root /path/to/ARAD_1K --dataset arad
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from hsi_data import FullImageDataset, TrainDataset
from metrics import AverageMeter, MRAELoss, validation_metrics
from models import FIXED_CSS_MODELS, build_fixed_css_model
from train import ModelEMA, get_logger, unwrap


def get_args():
    p = argparse.ArgumentParser(description="Train a reconstruction network with a fixed CSS")
    p.add_argument("--model", required=True, choices=FIXED_CSS_MODELS)
    p.add_argument("--css", default="css/optimized_css_4ch.npy")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="arad", choices=["arad", "cave"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--end_epoch", type=int, default=200)
    p.add_argument("--init_lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./exp")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, css, device):
    model.eval()
    meters = {}
    for target in loader:
        target = target.to(device)
        msi = torch.einsum("bchw,kc->bkhw", target, css)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(msi)
        for k, v in validation_metrics(output.float(), target.float()).items():
            meters.setdefault(k, AverageMeter()).update(v)
    model.train()
    return {k: m.avg for k, m in meters.items()}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{args.model}_{args.dataset}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info(f"Config: {json.dumps(vars(args))}")

    css = torch.from_numpy(np.load(args.css).astype(np.float32)).to(device)   # [K, 31]
    train_set = TrainDataset(args.data_root, args.dataset, crop_size=args.patch_size)
    val_set = FullImageDataset(args.data_root, args.dataset, part="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = build_fixed_css_model(args.model, in_channels=css.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))
    total_iters = len(train_loader) * args.end_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=args.min_lr)
    ema = ModelEMA(model, args.ema_decay)
    criterion = MRAELoss()
    best_mrae, iteration = float("inf"), 0

    for epoch in range(1, args.end_epoch + 1):
        model.train()
        losses = AverageMeter()
        for target in train_loader:
            target = target.to(device, non_blocking=True)
            msi = torch.einsum("bchw,kc->bkhw", target, css)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(msi)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            iteration += 1
            losses.update(loss.item(), target.size(0))

        ema_m = validate(ema.ema, val_loader, css, device)
        logger.info(f"e{epoch} train_mrae={losses.avg:.6f} [EMA] "
                    + " ".join(f"{k}:{v:.6f}" for k, v in ema_m.items()))
        torch.save({"epoch": epoch, "iteration": iteration, "state_dict": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict()}, os.path.join(out_dir, "latest_model.pth"))
        if ema_m["MRAE"] < best_mrae:
            best_mrae = ema_m["MRAE"]
            torch.save({"epoch": epoch, "iteration": iteration, "state_dict": ema.ema.state_dict()},
                       os.path.join(out_dir, "best_ema.pth"))
            logger.info(f"  saved best_ema.pth (val MRAE {best_mrae:.6f})")


if __name__ == "__main__":
    main()
