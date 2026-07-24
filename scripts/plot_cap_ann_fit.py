"""Plot the trained CGD/CGS ANNs against the measured C-V data, per device
geometry, plus a predicted-vs-measured parity plot. Sends a visual QC check
that the R^2/MARE numbers in outputs_cap/metrics.json don't convey on their
own (error is concentrated at the sharp turn-on knee, not the overall
curve shape -- see data_cv_cleaned/README.md for why: only 4 (W,L) points).

Usage:
    python scripts/plot_cap_ann_fit.py
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
CAP_DIR = os.path.join(REPO, "outputs_cap")
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


def load_model(target):
    with open(os.path.join(CAP_DIR, f"weights_{target.lower()}.json")) as f:
        w = json.load(f)
    model = TFTNet(n_inputs=3, n_hidden=w["architecture"]["n_hidden"], n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(CAP_DIR, f"model_weights_{target.lower()}.pt")))
    model.eval()
    return model, w


def predict(model, w, vg_arr, W, L):
    vg_lo, vg_hi = w["input_scaling_minmax"]["VG"]
    w_lo, w_hi = w["input_scaling_minmax"]["W"]
    l_lo, l_hi = w["input_scaling_minmax"]["L"]
    c_mean = w["target_transform"]["mean_F"]
    c_std = w["target_transform"]["std_F"]
    vg_s = np.clip((vg_arr - vg_lo) / (vg_hi - vg_lo), 0, 1)
    w_s = np.clip((W - w_lo) / (w_hi - w_lo), 0, 1)
    l_s = np.clip((L - l_lo) / (l_hi - l_lo), 0, 1)
    X = np.stack([vg_s, np.full_like(vg_s, w_s), np.full_like(vg_s, l_s)], axis=1).astype(np.float32)
    with torch.no_grad():
        y = model(torch.as_tensor(X)).squeeze(1).numpy()
    c = y * c_std + c_mean
    return np.clip(c, 0, None)


def main():
    df = pd.read_csv(CV_CSV)
    cgd_model, cgd_w = load_model("CGD")
    cgs_model, cgs_w = load_model("CGS")

    vg_dense = np.linspace(-3, 5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (W, L) in DEVICE_ORDER:
        sub = df[(df.W == W) & (df.L == L)].sort_values("VG")
        color = DEVICE_COLOR[(W, L)]
        axes[0].plot(sub.VG, sub.CGD * 1e12, "o", ms=4, color=color, alpha=0.6,
                     label=f"W={W},L={L} (measured)")
        axes[0].plot(vg_dense, predict(cgd_model, cgd_w, vg_dense, W, L) * 1e12, "-",
                     lw=1.8, color=color)
        axes[1].plot(sub.VG, sub.CGS * 1e12, "o", ms=4, color=color, alpha=0.6,
                     label=f"W={W},L={L} (measured)")
        axes[1].plot(vg_dense, predict(cgs_model, cgs_w, vg_dense, W, L) * 1e12, "-",
                     lw=1.8, color=color)
    axes[0].set_title("$C_{GD}$ ANN fit (10 hidden neurons) vs. measured")
    axes[1].set_title("$C_{GS}$ ANN fit (10 hidden neurons) vs. measured")
    for a in axes:
        a.set_xlabel("$V_G$ (V)")
        a.set_ylabel("Capacitance (pF)")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Trained CGD/CGS ANNs vs. measured C-V (dots = data, lines = ANN, VDS=0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cgd_cgs_ann_fit.png"))
    plt.close(fig)

    # parity plot (predicted vs measured, all rows, both targets)
    fig, ax = plt.subplots(figsize=(6, 6))

    def predict_rows(model, w, sub):
        vg_lo, vg_hi = w["input_scaling_minmax"]["VG"]
        w_lo, w_hi = w["input_scaling_minmax"]["W"]
        l_lo, l_hi = w["input_scaling_minmax"]["L"]
        c_mean = w["target_transform"]["mean_F"]
        c_std = w["target_transform"]["std_F"]
        vg_s = np.clip((sub.VG.to_numpy() - vg_lo) / (vg_hi - vg_lo), 0, 1)
        w_s = np.clip((sub.W.to_numpy() - w_lo) / (w_hi - w_lo), 0, 1)
        l_s = np.clip((sub.L.to_numpy() - l_lo) / (l_hi - l_lo), 0, 1)
        X = np.stack([vg_s, w_s, l_s], axis=1).astype(np.float32)
        with torch.no_grad():
            y = model(torch.as_tensor(X)).squeeze(1).numpy()
        return np.clip(y * c_std + c_mean, 0, None)

    cgd_pred = predict_rows(cgd_model, cgd_w, df)
    cgs_pred = predict_rows(cgs_model, cgs_w, df)
    ax.scatter(df.CGD * 1e12, cgd_pred * 1e12, s=20, color="#D55E00", alpha=0.7, label="$C_{GD}$")
    ax.scatter(df.CGS * 1e12, cgs_pred * 1e12, s=20, color="#0072B2", alpha=0.7, marker="^", label="$C_{GS}$")
    lims = [0, max(df.CGD.max(), df.CGS.max()) * 1e12 * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Measured (pF)")
    ax.set_ylabel("ANN predicted (pF)")
    ax.set_title("Predicted vs. measured (all 164 rows, both targets)")
    ax.legend(frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cgd_cgs_parity.png"))
    plt.close(fig)

    print("Wrote", os.path.join(OUT_DIR, "cgd_cgs_ann_fit.png"))
    print("Wrote", os.path.join(OUT_DIR, "cgd_cgs_parity.png"))


if __name__ == "__main__":
    main()
