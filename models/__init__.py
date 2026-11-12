import torch

from .ffra_css import ASIModule, FFRACSSNet
from .fixed_css import FIXED_CSS_MODELS, build_fixed_css_model


def build_online_model(num_channels=4, poly_degree=4, pretrained_backbone=True):
    """FFRA-CSS online model: ASIModules (CSS generation) + CASF-SST reconstruction."""
    return FFRACSSNet(in_channels=num_channels, poly_degree=poly_degree,
                      pretrained_backbone=pretrained_backbone)


def load_weights(model, path, device="cpu"):
    """Load a checkpoint saved by train.py / train_fixed_css.py (or the released weights)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


__all__ = ["ASIModule", "FFRACSSNet", "build_online_model", "build_fixed_css_model", "FIXED_CSS_MODELS",
           "load_weights"]
