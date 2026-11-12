# FFRA-CSS

Code for the paper **"Fabrication feasibility and reconstruction accuracy balanced camera spectral sensitivity optimization method"** (submitted to *Expert Systems with Applications*).

FFRA-CSS jointly optimizes the camera spectral sensitivity (CSS) and a hyperspectral reconstruction network. Three design choices make the learned CSS easier to fabricate:

1. **Choosing the number of channels.** Distribution compactness (DC) analysis of band redundancy is combined with reconstruction experiments for 1 to 6 channels, and four channels are selected.
2. **Fabrication-oriented CSS parameterization.** Each channel is a fourth-order polynomial with bounded coefficients, followed by Softplus and L1 normalization. The curves are therefore smooth and non-negative by construction.
3. **Dynamic loss weighting.** The weight of the spectral angle mapper (SAM) loss increases with a cosine warm-up during training, so amplitude accuracy and spectral shape are both taken into account.

## Repository structure

```
FFRA-CSS/
├── train.py               # joint training of the CSS (ASIModules) and CASF-SST
├── test.py                # evaluation: online FFRA-CSS model or fixed CSS + other networks
├── export_css.py          # average the generated CSS over the validation set
├── train_fixed_css.py     # train a reconstruction network with a fixed CSS
├── hsi_data.py            # ARAD-1K / CAVE loading and data splits
├── metrics.py             # losses and the six evaluation metrics
├── models/
│   ├── ffra_css.py        # ASIModule + CASF-SST (online model)
│   └── fixed_css/         # CASF-SST, MPRNet, MST++, Restormer with a fixed-CSS input
├── tools/
│   ├── dc_analysis.py     # DC curve and cumulative mean deviation G(k)
│   └── convert_cave.py    # CAVE PNG bands -> .mat cubes
├── css/                   # the optimized four-channel CSS
└── third_party_licenses/
```

## Environment

The experiments in the paper used Python 3.12, PyTorch 2.8.0 and CUDA 12.8 on a single NVIDIA RTX 5090. The released code has also been run with Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # choose your CUDA version
pip install -r requirements.txt
```

## Data

Download the public datasets:

- **ARAD-1K** (NTIRE 2022 Spectral Recovery Challenge): https://github.com/caiyuanhao1998/MST-plus-plus#2-data-preparation
- **CAVE**: https://cave.cs.columbia.edu/repository/Multispectral

Each dataset root must contain a `Train_Spec/` folder with one `.mat` file (HDF5, MATLAB v7.3) per image. Each file stores the 31-band cube (400 to 700 nm in 10 nm steps, values in [0, 1]) under the key `cube`. The ARAD-1K `.mat` files from the challenge already use this format. CAVE is distributed as PNG bands; convert it with

```bash
python tools/convert_cave.py --src /path/to/cave_png --dst /path/to/CAVE/Train_Spec
```

```
ARAD_1K/Train_Spec/ARAD_1K_0001.mat ...
CAVE/Train_Spec/balloons_ms.mat ...
```

The splits are created in `hsi_data.py`:

| Dataset | Files | Shuffle | Train / Val / Test |
|---|---|---|---|
| ARAD-1K | 950 public files, sorted by name | `random.Random(42)` | 850 / 50 / 50 |
| CAVE | 32 files, sorted by name | `random.Random(1024)` | 20 / 4 / 8 |

## Usage

### 1. Joint optimization of the CSS and the reconstruction network

```bash
python train.py --data_root /path/to/ARAD_1K --dataset arad --num_channels 4
```

Default settings follow the paper:
- Adam optimizer; learning rate decays from 1e-4 to 1e-6 with cosine annealing.
- Batch size 4, 256x256 random crops, 200 epochs.
- Maximum SAM weight 20, reached through a cosine warm-up over the first 100 epochs.
- EMA decay 0.9998.
- bfloat16 mixed precision on GPU.

The EMA weights with the lowest validation MRAE are saved as `best_ema.pth`. Add `--compile` to use `torch.compile`, as in the paper runs.

- Channel-number study (Section 4.2): `--num_channels 1` to `6`.
- Polynomial-order ablation (Section 4.3.1): `--poly_degree 3`, `4` or `5`.

### 2. Evaluation

```bash
# FFRA-CSS online model, four-branch flip TTA
python test.py --mode online --checkpoint best_ema.pth --data_root /path/to/ARAD_1K --dataset arad

# single forward pass, without TTA (Table 2 setting)
python test.py --mode online --checkpoint best_ema.pth --data_root /path/to/ARAD_1K --dataset arad --tta none
```

Six metrics are reported: MRAE, RMSE, PSNR, SSIM, SAM and CC. Outputs are clipped to [0, 1], and the test MRAE only counts elements whose ground truth is greater than 0.01. Per-image results and the mean are written to `results/metrics.csv`.

### 3. Exporting the optimized CSS

```bash
python export_css.py --checkpoint best_ema.pth --data_root /path/to/ARAD_1K --out css/my_css.npy
```

The CSS generated for each validation image is averaged into a `[K, 31]` array. The curves used in the paper are in `css/optimized_css_4ch.npy` and `css/optimized_css_4ch.csv`. They are the average over the 50 ARAD-1K validation images, with response peaks at 640, 460, 580 and 400 nm.

### 4. Fixed CSS with other reconstruction networks (Section 4.4.2)

```bash
python train_fixed_css.py --model mprnet --css css/optimized_css_4ch.npy --data_root /path/to/ARAD_1K
python test.py --mode fixed_css --model mprnet --css css/optimized_css_4ch.npy \
    --checkpoint exp/mprnet_arad_xxx/best_ema.pth --data_root /path/to/ARAD_1K
```

Available networks: `casfsst_fixed`, `mprnet`, `mstplusplus`, `restormer`.

### 5. DC analysis of band redundancy (Fig. 5)

```bash
python tools/dc_analysis.py --data_root /path/to/ARAD_1K --out_dir results/dc
```

This samples 50 ARAD-1K training patches, computes the DC components, the smoothed log DC curve, its global baseline and the cumulative mean deviation G(k), and writes them to `dc_curve.csv`. The paper ran it on a GPU in single precision, so the smallest DC components can differ slightly on a CPU.

## Pretrained weights

<!-- TODO: upload best_ema.pth files to GitHub Releases after acceptance and add the links here. -->
To be released.

## Acknowledgements and third-party code

- The CASF-SST reconstruction network and the JOCSR baseline follow **JOCSR** (Tong et al., 2026, *Optics & Laser Technology*, 203, 115472): https://github.com/tgg-77/JOCSR
- The spectral Transformer blocks are adapted from **MST++** (MIT License): https://github.com/caiyuanhao1998/MST-plus-plus
- `models/fixed_css/restormer.py` is adapted from **Restormer** (MIT License): https://github.com/swz30/Restormer
- `models/fixed_css/mprnet.py` is adapted from **MPRNet**: https://github.com/swz30/MPRNet. It is distributed under MPRNet's ACADEMIC PUBLIC LICENSE (non-commercial use only); see `third_party_licenses/MPRNet_LICENSE.md`.

## Citation

<!-- TODO: add the BibTeX entry after publication. -->

## License

The code in this repository is released under the MIT License (see `LICENSE`), except for third-party files that keep their original licenses (see above).
