"""Plot the area-normalised Cgd/Cgs ANNs (trained_Cg_ANN/) against measured C-V.

One panel per measured (W, L) geometry per component: dots = measured,
line = the trained 4-input (VG, VD, W, L) two-layer network evaluated on a
fine VG sweep. These are the area-normalised models -- target
C_pF/(W*L)*1e3 -- from src/train_cg_final.py, NOT the superseded per-device
absolute-pF networks in outputs_cap/per_device.

Usage:
    python scripts/plot_cg_areanorm_fit.py
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
from src.model import TFTNet2

CG_DIR = os.path.join(REPO, "trained_Cg_ANN")
CV_CSV = os.path.join(REPO, "data_cleaned_cv", "merged_cap_dataset.csv")
OUT_DIR = os.path.join(REPO, "trained_Cg_ANN", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.facecolor": "#f7f7f5", "axes.facecolor": "#f7f7f5",
})

DEVICE_ORDER = [(20, 20), (40, 20), (160, 15), (160, 20)]
MEAS = "#4c72b0"
MODEL = "#c44e52"
FEATURES = ["VG", "VD", "W", "L"]


def load(cap_type):
    with open(os.path.join(CG_DIR, cap_type, "weights.json")) as f:
        w = json.load(f)
    a = w["architecture"]
    model = TFTNet2(n_inputs=4, n_hidden1=a["n_hidden1"], n_hidden2=a["n_hidden2"], n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(CG_DIR, cap_type, "model_weights.pt")))
    model.eval()
    return model, w


def predict_pf(model, w, vg, vd, W, L):
    """Recover C in pF: C_pF = (out*y_std + y_mean) * W * L / 1e3."""
    b = w["input_scaling_minmax"]
    cols = []
    for name, val in zip(FEATURES, (vg, vd, W, L)):
        lo, hi = b[name]
        cols.append((np.asarray(val, dtype=np.float64) - lo) / (hi - lo))
    X = np.stack(np.broadcast_arrays(*cols), axis=1).astype(np.float32)
    with torch.no_grad():
        out = model(torch.as_tensor(X)).squeeze(1).numpy()
    t = w["target_transform"]
    return (out * t["y_std"] + t["y_mean"]) * W * L / 1e3


def panel_row(fig, gs_row, axes, df, cap_type, model, w, summary):
    label = "GD" if cap_type == "cgd" else "GS"
    for ax, (W, L) in zip(axes, DEVICE_ORDER):
        sub = df[(df.W == W) & (df.L == L)].sort_values("VG")
        vg = sub["VG"].to_numpy(dtype=np.float64)
        vd = sub["VD"].to_numpy(dtype=np.float64)
        meas = sub["C_pF"].to_numpy(dtype=np.float64)

        fit = predict_pf(model, w, vg, vd, W, L)
        vg_f = np.linspace(vg.min(), vg.max(), 400)
        fit_f = predict_pf(model, w, vg_f, np.full_like(vg_f, vd[0]), W, L)

        rmse = float(np.sqrt(np.mean((fit - meas) ** 2)))
        ss_res = np.sum((meas - fit) ** 2)
        ss_tot = np.sum((meas - meas.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        nrmse = rmse / (meas.max() - meas.min()) * 100
        summary.append({"component": f"C{label}", "W": W, "L": L, "W_over_L": W / L,
                        "n_points": len(vg), "r2": r2, "rmse_pF": rmse,
                        "nrmse_range_percent": nrmse})

        ax.scatter(vg, meas, s=20, color=MEAS, alpha=0.85, zorder=3, label="measured")
        ax.plot(vg_f, fit_f, color=MODEL, lw=2, zorder=2, label="ANN")
        ax.set_title(f"W={W}, L={L} um (VD={vd[0]:g}V)", fontsize=11)
        ax.set_xlabel("VG (V)")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].set_ylabel(f"$C_{{{label}}}$ (pF)")


def main():
    df = pd.read_csv(CV_CSV)
    summary = []

    for cap_type in ("cgd", "cgs"):
        label = "GD" if cap_type == "cgd" else "GS"
        sub = df[df.cap_type == cap_type]
        model, w = load(cap_type)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
        panel_row(fig, 0, axes, sub, cap_type, model, w, summary)
        fig.suptitle(f"C{label} ANN vs measurement (dots=measured, line=model)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(OUT_DIR, f"c{cap_type[1:]}_vg_fit.png")
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out)

    # Both components stacked, for the report
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.6))
    for row, cap_type in enumerate(("cgd", "cgs")):
        sub = df[df.cap_type == cap_type]
        model, w = load(cap_type)
        panel_row(fig, row, axes[row], sub, cap_type, model, w, [])
    fig.suptitle("Area-normalised Cgd / Cgs ANNs vs measured C-V "
                 "(dots = measured, lines = model)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "cg_areanorm_vg_fit.png"))
    plt.close(fig)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT_DIR, "cg_areanorm_per_geometry.csv"), index=False)
    print()
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
