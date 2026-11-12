"""Joint optimization of the CSS (ASIModules) and the CASF-SST reconstruction network.

Example (ARAD-1K, 4 channels, settings of the paper):
    python train.py --data_root /path/to/ARAD_1K --dataset arad --num_channels 4

Channel-number study (Section 4.2): run with --num_channels 1 ... 6.
Polynomial-order ablation (Section 4.3.1): run with --poly_degree 3 / 4 / 5.
"""
import argparse
import copy
import datetime
import json
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from hsi_data import FullImageDataset, TrainDataset
from metrics import AverageMeter, MRAELoss, SAMLoss, sam_weight, validation_metrics
from models import build_online_model


def get_args():
    p = argparse.ArgumentParser(description="FFRA-CSS joint CSS / reconstruction training")
    p.add_argument("--data_root", required=True, help="dataset root containing Train_Spec/")
    p.add_argument("--dataset", default="arad", choices=["arad", "cave"])
    p.add_argument("--num_channels", type=int, default=4, help="number of CSS channels K")
    p.add_argument("--poly_degree", type=int, default=4, help="highest polynomial order")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--end_epoch", type=int, default=200)
    p.add_argument("--init_lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lambda_sam", type=float, default=20.0, help="maximum SAM loss weight")
    p.add_argument("--warmup_epochs", type=int, default=100, help="cosine warm-up length of the SAM weight")
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="use torch.compile (as in the paper runs)")
    p.add_argument("--resume", default=None, help="latest_model.pth to resume from")
    p.add_argument("--out_dir", default="./exp")
    p.add_argument("--css_snapshot_every", type=int, default=5, help="save CSS curves every N epochs (0: off)")
    return p.parse_args()


def unwrap(model):
    core = model
    if hasattr(core, "_orig_mod"):
        core = core._orig_mod
    if hasattr(core, "module"):
        core = core.module
    return core


class ModelEMA:
    """Exponential moving average of the model weights."""

    def __init__(self, model, decay):
        self.ema = copy.deepcopy(unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        ema_sd, cur_sd = self.ema.state_dict(), unwrap(model).state_dict()
        keys = [k for k, v in ema_sd.items() if v.dtype.is_floating_point]
        torch._foreach_mul_([ema_sd[k] for k in keys], self.decay)
        torch._foreach_add_([ema_sd[k] for k in keys], [cur_sd[k] for k in keys], alpha=1 - self.decay)
        for k, v in ema_sd.items():
            if not v.dtype.is_floating_point:
                v.copy_(cur_sd[k])


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meters = {}
    for target in loader:
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(target)
        for k, v in validation_metrics(output.float(), target).items():
            meters.setdefault(k, AverageMeter()).update(v)
    model.train()
    return {k: m.avg for k, m in meters.items()}


def get_logger(path):
    logger = logging.getLogger("ffra_css")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True

    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = os.path.join(args.out_dir, f"ffra_css_{args.dataset}_k{args.num_channels}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info(f"Config: {json.dumps(vars(args))}")

    train_set = TrainDataset(args.data_root, args.dataset, crop_size=args.patch_size)
    val_set = FullImageDataset(args.data_root, args.dataset, part="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = build_online_model(args.num_channels, args.poly_degree).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))

    iters_per_epoch = len(train_set) // args.batch_size
    total_iters = iters_per_epoch * args.end_epoch
    warmup_iters = iters_per_epoch * args.warmup_epochs

    iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v
                               for k, v in ckpt["state_dict"].items()})
        optimizer.load_state_dict(ckpt["optimizer"])
        iteration = ckpt.get("iter", 0)
        logger.info(f"Resumed from {args.resume} at iteration {iteration}")
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", args.init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=args.min_lr, last_epoch=iteration if iteration > 0 else -1)

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    ema = ModelEMA(model, args.ema_decay)
    criterion_mrae, criterion_sam = MRAELoss(), SAMLoss()
    best_mrae = float("inf")

    model.train()
    while iteration < total_iters:
        losses = AverageMeter()
        for target in train_loader:
            if iteration >= total_iters:
                break
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(target)
            lam = sam_weight(iteration, warmup_iters, args.lambda_sam)
            loss = criterion_mrae(output, target) + lam * criterion_sam(output, target)   # Eq. (18)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            iteration += 1
            losses.update(loss.item())

            if iteration % max(1, iters_per_epoch // 5) == 0:
                logger.info(f"[iter {iteration}/{total_iters}] lr={optimizer.param_groups[0]['lr']:.3e} "
                            f"loss={losses.avg:.6f} lambda_sam={lam:.3f}")

            if iteration % iters_per_epoch == 0:
                epoch = iteration // iters_per_epoch
                raw = validate(model, val_loader, device)
                ema_m = validate(ema.ema, val_loader, device)
                logger.info(f"[RAW] e{epoch} " + " ".join(f"{k}:{v:.6f}" for k, v in raw.items()))
                logger.info(f"[EMA] e{epoch} " + " ".join(f"{k}:{v:.6f}" for k, v in ema_m.items()))

                state = {"epoch": epoch, "iter": iteration, "state_dict": unwrap(model).state_dict(),
                         "optimizer": optimizer.state_dict()}
                if ema_m["MRAE"] < best_mrae:              # select by validation MRAE of the EMA weights
                    best_mrae = ema_m["MRAE"]
                    torch.save(state, os.path.join(out_dir, "best_model.pth"))
                    torch.save({"epoch": epoch, "iter": iteration, "state_dict": ema.ema.state_dict()},
                               os.path.join(out_dir, "best_ema.pth"))
                    logger.info(f"  saved best_ema.pth (val MRAE {best_mrae:.6f})")
                if epoch % 5 == 0 or iteration >= total_iters:
                    torch.save(state, os.path.join(out_dir, "latest_model.pth"))
                if args.css_snapshot_every and epoch % args.css_snapshot_every == 0:
                    css = unwrap(model).css
                    if css is not None:
                        snap_dir = os.path.join(out_dir, "css_snapshots")
                        os.makedirs(snap_dir, exist_ok=True)
                        np.save(os.path.join(snap_dir, f"css_epoch_{epoch:03d}.npy"),
                                css.detach().float().mean(dim=0).cpu().numpy())
                losses = AverageMeter()


if __name__ == "__main__":
    main()
