"""FFRA-CSS online model: K ASIModules generate the CSS curves, and a CASF-SST
reconstruction network recovers the 31-band hyperspectral image from the K-channel
multispectral observation. Both parts are trained jointly end-to-end.

Parameter names are kept identical to the original training code so that released
checkpoints load with ``strict=True``.
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange


# ----------------------------------------------------------------------------
# Initialization helpers
# ----------------------------------------------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.", stacklevel=2)
    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ----------------------------------------------------------------------------
# ASIModule: polynomial CSS generation (Section 3.2.2)
# ----------------------------------------------------------------------------
class ASIModule(nn.Module):
    """Adaptive Spectral-sensitivity Inference Module.

    ResNet-18 regresses (poly_degree + 1) polynomial coefficients from the input HSI.
    The coefficients are bounded by ``a_max * tanh`` (Eq. 7), combined with a Vandermonde
    basis over the normalized wavelengths (Eqs. 8-9), mapped to non-negative values by
    Softplus (Eq. 10) and L1-normalized (Eq. 11). The resulting curve is applied to the
    HSI as a band-wise weighted sum, which gives one multispectral channel (Eq. 25).
    """

    def __init__(self, input_channels=31, output_dim=31, poly_degree=4, poly_basis="vandermonde",
                 coeff_bound=4.0, pretrained_backbone=True):
        super().__init__()
        if poly_degree < 0:
            raise ValueError("poly_degree must be non-negative")
        if poly_basis not in ("vandermonde", "qr"):
            raise ValueError("poly_basis must be 'vandermonde' or 'qr'")
        self.poly_degree = poly_degree
        self.poly_basis = poly_basis
        self.coeff_bound = coeff_bound

        if pretrained_backbone:
            try:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:  # torchvision < 0.13
                self.resnet = models.resnet18(pretrained=True)
        else:
            self.resnet = models.resnet18()
        # 31-band input and (poly_degree + 1) coefficient output
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, poly_degree + 1)

        # Wavelengths 400-700 nm (31 bands) linearly mapped to [-1, 1]
        x = torch.linspace(-1, 1, steps=output_dim, dtype=torch.float32)
        if poly_basis == "vandermonde":
            # rows: x^d, ..., x, 1  -> shape [d + 1, 31]
            basis = torch.stack([x ** i for i in range(poly_degree, -1, -1)], dim=0)
        else:
            vander = torch.stack([x ** i for i in range(0, poly_degree + 1)], dim=1)
            q, _ = torch.linalg.qr(vander, mode="reduced")
            basis = q.transpose(0, 1).contiguous()
        self.register_buffer("X_matrix", basis)

    def forward(self, x):
        raw_coeffs = self.resnet(x)                                   # [B, d + 1]
        poly_coeffs = torch.tanh(raw_coeffs) * self.coeff_bound      # Eq. (7)
        self.poly_coeffs = poly_coeffs
        curve = torch.matmul(poly_coeffs, self.X_matrix)             # Eq. (9), [B, 31]
        curve = F.softplus(curve)                                    # Eq. (10)
        css = curve / (curve.sum(dim=1, keepdim=True) + 1e-8)        # Eq. (11)
        band = (x * css[:, :, None, None]).sum(dim=1, keepdim=True)  # Eq. (25)
        return band, css


# ----------------------------------------------------------------------------
# Spectral Transformer backbone (MST blocks, adapted from MST++, MIT License)
# ----------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """x_in: [b, h, w, c] -> [b, h, w, c]"""
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp, k_inp, v_inp = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                           PreNorm(dim, FeedForward(dim=dim))])
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=(2, 4, 4)):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.bottleneck = MSAB(dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                     heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        fea = self.embedding(x)
        fea_encoder = []
        for msab, down in self.encoder_layers:
            fea = msab(fea)
            fea_encoder.append(fea)
            fea = down(fea)
        fea = self.bottleneck(fea)
        for i, (up, fusion, msab) in enumerate(self.decoder_layers):
            fea = up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = msab(fea)
        return self.mapping(fea) + x


# ----------------------------------------------------------------------------
# Full online model (Fig. 2)
# ----------------------------------------------------------------------------
class FFRACSSNet(nn.Module):
    """K ASIModules (one per channel) + CASF-SST reconstruction network.

    Input:  HSI [B, 31, H, W]
    Output: reconstructed HSI [B, 31, H, W]
    After each forward pass, ``self.css`` holds the generated CSS, shape [B, K, 31].
    """

    def __init__(self, in_channels=4, out_channels=31, n_feat=31, stage=3, poly_degree=4,
                 poly_basis="vandermonde", pretrained_backbone=True):
        super().__init__()
        self.in_channels = in_channels
        self.poly_degree = poly_degree
        self.poly_basis = poly_basis

        # one independent ASIModule per observation channel
        self.msi_nets = nn.ModuleList([
            ASIModule(poly_degree=poly_degree, poly_basis=poly_basis, pretrained_backbone=pretrained_backbone)
            for _ in range(in_channels)
        ])
        self.css = None
        self.stage = stage

        # channel-wise projection and fusion (Eqs. 13-14)
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1, bias=False)          # phi_all
        self.conv_single = nn.ModuleList([
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1, bias=False) for _ in range(in_channels)   # phi_c
        ])
        self.conv_k = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)                # phi_mix
        self.conv_input = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)

        # multi-stage spectral Transformer backbone of CASF-SST
        self.body = nn.Sequential(*[MST(dim=31, stage=2, num_blocks=[1, 1, 1]) for _ in range(stage)])
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hsi):
        bands, curves, coeffs = [], [], []
        for asi in self.msi_nets:
            band, css = asi(hsi)
            bands.append(band)
            curves.append(css)
            coeffs.append(asi.poly_coeffs)
        self.css = torch.stack(curves, dim=1)          # [B, K, 31]
        self.poly_coeffs = torch.stack(coeffs, dim=1)  # [B, K, d + 1]

        x = torch.cat(bands, dim=1)                    # K-channel multispectral observation
        _, _, h_inp, w_inp = x.shape
        pad_h = (8 - h_inp % 8) % 8
        pad_w = (8 - w_inp % 8) % 8
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        fused = 0
        for i in range(self.in_channels):
            fused = fused + self.relu(self.conv_single[i](x[:, i:i + 1]))
        fused = self.conv_k(fused)

        x = self.conv_in(x) + fused
        x = self.conv_input(x)
        h = self.conv_out(self.body(x)) + x
        return h[:, :, :h_inp, :w_inp]
