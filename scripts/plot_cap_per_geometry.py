"""Per-geometry CGD/CGS measured-vs-fitted panels.

The counterpart to trained_ANN/deep/dc_analysis/dc_sweep_all19.png (which
gives the ID model one panel per (W, L) geometry): here each of the 4
measured C-V geometries gets its own panel per target, instead of all four
being overlaid on a single axes as in cgd_cgs_ann_fit_per_device.png.
Uses the already-trained per-device networks in outputs_cap/per_device --
nothing is retrained.

Usage:
    python scripts/plot_cap_per_geometry.py
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
TARGET_COLOR = {"CGD": "#0072B2", "CGS": "#D55E00"}


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
    summary = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, target in enumerate(("CGD", "CGS")):
        for col, (W, L) in enumerate(DEVICE_ORDER):
            ax = axes[row, col]
            sub = df[(df.W == W) & (df.L == L)].sort_values("VG")
            vg = sub["VG"].to_numpy(dtype=np.float64)
            meas = sub[target].to_numpy(dtype=np.float64)

            model, w = load(target, W, L)
            fit_at_meas = predict(model, w, vg)
            vg_fine = np.linspace(vg.min(), vg.max(), 400)
            fit_fine = predict(model, w, vg_fine)

            rmse = float(np.sqrt(np.mean((fit_at_meas - meas) ** 2)))
            mare = float(np.mean(np.abs(meas - fit_at_meas) / np.maximum(np.abs(meas), 1e-15)) * 100)
            ss_res = np.sum((meas - fit_at_meas) ** 2)
            ss_tot = np.sum((meas - meas.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot)
            summary.append({"target": target, "W": W, "L": L, "W_over_L": W / L,
                            "n_points": len(vg), "rmse_F": rmse,
                            "MARE_percent": mare, "r2": r2})

            c = TARGET_COLOR[target]
            ax.scatter(vg, meas * 1e12, s=22, color=c, alpha=0.75, zorder=3, label="measured")
            ax.plot(vg_fine, fit_fine * 1e12, color=c, lw=2, zorder=2, label="ANN fit")
            ax.set_title(f"$C_{{{target[1:]}}}$  W={W}, L={L}  (W/L={W/L:g})\n"
                         f"$R^2$={r2:.4f}  MARE={mare:.1f}%", fontsize=10)
            if row == 1:
                ax.set_xlabel("$V_G$ (V)")
            if col == 0:
                ax.set_ylabel("Capacitance (pF)")
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("Per-geometry C-V measured vs fitted -- all 4 measured (W, L) geometries, "
                 "both targets (dots = measured, lines = ANN, VDS=0)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT_DIR, "cap_per_geometry_fit.png")
    fig.savefig(out_png)
    plt.close(fig)

    # Parity, one panel per target, colored by geometry
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, target in zip(axes, ("CGD", "CGS")):
        for (W, L) in DEVICE_ORDER:
            sub = df[(df.W == W) & (df.L == L)].sort_values("VG")
            vg = sub["VG"].to_numpy(dtype=np.float64)
            meas = sub[target].to_numpy(dtype=np.float64)
            model, w = load(target, W, L)
            fit = predict(model, w, vg)
            ax.scatter(meas * 1e12, fit * 1e12, s=18, alpha=0.7, label=f"W={W},L={L}")
        lim = [0, max(df[target].max() * 1e12 * 1.05, 1)]
        ax.plot(lim, lim, color="#1f1f1f", lw=1, ls="--", alpha=0.6, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"measured $C_{{{target[1:]}}}$ (pF)")
        ax.set_ylabel(f"fitted $C_{{{target[1:]}}}$ (pF)")
        ax.set_title(f"{target} parity -- all geometries")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cap_per_geometry_parity.png"))
    plt.close(fig)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT_DIR, "cap_per_geometry_summary.csv"), index=False)
    print(sdf.to_string(index=False))
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()
