"""Plot the per-device CGD/CGS ANNs (src/train_cap_per_device.py) against
measured C-V data -- the visual counterpart to plot_cap_ann_fit.py, but for
the per-device architecture that replaced the combined (VG,W,L) model after
it visibly misplaced each device's turn-on knee.

Usage:
    python scripts/plot_cap_ann_fit_per_device.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.model import TFTNet

CAP_DIR = os.path.join(REPO, "outputs_cap", "per_device")
CV_CSV = os.path.join(REPO, "data_cv_cleaned", "merged_cv_dataset.csv")
OUT_DIR = os.path.join(REPO, "outputs_cap", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

DEVICE_ORDER = [(20, 20), (40, 20), (160, 20), (160, 15)]
DEVICE_COLOR = {
    (20, 20): "#0072B2", (40, 20): "#E69F00",
    (160, 20): "#009E73", (160, 15): "#D55E00",
}


def load(target, W, L):
    key = f"{target.lower()}_W{W}_L{L}"
    with open(os.path.join(CAP_DIR, f"weights_{key}.json")) as f:
        w = json.load(f)
    model = TFTNet(n_inputs=1, n_hidden=w["architecture"]["n_hidden"], n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(CAP_DIR, f"model_{key}.pt")))
    model.eval()
    return model, w


def predict(model, w, vg_arr):
    vg_lo, vg_hi = w["input_scaling_minmax"]["VG"]
    c_mean, c_std = w["target_transform"]["mean_F"], w["target_transform"]["std_F"]
    vg_s = np.clip((vg_arr - vg_lo) / (vg_hi - vg_lo), 0, 1)
    with torch.no_grad():
        y = model(torch.as_tensor(vg_s, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
    return np.clip(y * c_std + c_mean, 0, None)


def main():
    df = pd.read_csv(CV_CSV)
    vg_dense = np.linspace(-3, 5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (W, L) in DEVICE_ORDER:
        sub = df[(df.W == W) & (df.L == L)].sort_values("VG")
        color = DEVICE_COLOR[(W, L)]
        cgd_model, cgd_w = load("CGD", W, L)
        cgs_model, cgs_w = load("CGS", W, L)
        axes[0].plot(sub.VG, sub.CGD * 1e12, "o", ms=4, color=color, alpha=0.6, label=f"W={W},L={L}")
        axes[0].plot(vg_dense, predict(cgd_model, cgd_w, vg_dense) * 1e12, "-", lw=1.8, color=color)
        axes[1].plot(sub.VG, sub.CGS * 1e12, "o", ms=4, color=color, alpha=0.6, label=f"W={W},L={L}")
        axes[1].plot(vg_dense, predict(cgs_model, cgs_w, vg_dense) * 1e12, "-", lw=1.8, color=color)
    axes[0].set_title("$C_{GD}$ per-device ANN fit")
    axes[1].set_title("$C_{GS}$ per-device ANN fit")
    for a in axes:
        a.set_xlabel("$V_G$ (V)"); a.set_ylabel("Capacitance (pF)")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Per-device CGD/CGS ANNs vs. measured C-V (dots = data, lines = ANN, VDS=0)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "cgd_cgs_ann_fit_per_device.png")
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
