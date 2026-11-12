from .casf_sst_fixed import CASF_SST_FixedCSS
from .mprnet import MPRNet
from .mst_plus_plus import MST_Plus_Plus
from .restormer import Restormer

# Reconstruction networks used with the fixed optimized CSS (Section 4.4.2).
# Every network takes a K-channel observation [B, K, H, W] and outputs [B, 31, H, W].
FIXED_CSS_MODELS = ("casfsst_fixed", "mprnet", "mstplusplus", "restormer")


def build_fixed_css_model(name, in_channels=4):
    name = name.lower()
    if name in ("casfsst_fixed", "casf_sst_fixed"):
        return CASF_SST_FixedCSS(in_channels=in_channels, out_channels=31, n_feat=31, stage=3)
    if name in ("mprnet", "mpr"):
        return MPRNet(input_channels=in_channels, in_c=31, out_c=31, n_feat=31, num_cab=4)
    if name in ("mstplusplus", "mst++", "mst_plus_plus"):
        return MST_Plus_Plus(in_channels=in_channels, out_channels=31, n_feat=31, stage=3)
    if name == "restormer":
        return Restormer(inp_channels=in_channels, out_channels=31, dual_pixel_task=False)
    raise ValueError(f"Unknown fixed-CSS model '{name}'. Choose from {FIXED_CSS_MODELS}.")


__all__ = ["build_fixed_css_model", "FIXED_CSS_MODELS"]
