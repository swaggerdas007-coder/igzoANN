"""Diagnostic plots for the trained parasitic-capacitance ANNs (C_GD, C_GS).

Produces, for each capacitance component:
  outputs/cap_<cap>/plots/scatter_C.png    -- predicted vs measured C (test set)
  outputs/cap_<cap>/plots/C_vg_curves.png  -- C-VG curves, model vs measured,
                                               one panel per measured geometry

Run:
    python -m src.evaluate_cap
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import FEATURE_BOUNDS
from src.cap_dataset import load_and_split
from src.model import TFTNet

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")

BG = "#f7f7f5"
FG = "#1f1f1f"
MEASURED = "#4c72b0"
MODEL = "#c44e52"


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.25)


def scale_point(vg, vd, w, l):
    def s(name, val):
        lo, hi = FEATURE_BOUNDS[name]
        return (val - lo) / (hi - lo)
    return np.array([s("VG", vg), s("VD", vd), s("W", w), s("L", l)], dtype=np.float32)


def load_model(cap):
    meta_path = os.path.join(OUT_ROOT, f"cap_{cap}", "metrics.json")
    with open(meta_path) as f:
        m = json.load(f)
    model = TFTNet(n_inputs=4, n_hidden=m["n_hidden"], n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(OUT_ROOT, f"cap_{cap}", "model_weights.pt")))
    model.eval()
    y_mean = m["target_standardization"]["y_mean_pF"]
    y_std = m["target_standardization"]["y_std_pF"]
    return model, y_mean, y_std, m


def predict_c(model, y_mean, y_std, vg, vd, w, l):
    X = np.stack([scale_point(a, b, c, d) for a, b, c, d in zip(vg, vd, w, l)])
    with torch.no_grad():
        pred = model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    return pred * y_std + y_mean


def scatter_plot(cap, d, m):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.scatter(d["c_true"], d["c_pred"], s=18, alpha=0.6, color=MEASURED, edgecolors="none")
    lo = 0.0
    hi = float(max(d["c_true"].max(), d["c_pred"].max())) * 1.05
    ax.plot([lo, hi], [lo, hi], color=FG, lw=1, ls="--", alpha=0.6, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"measured $C_{%s}$ (pF)" % cap[1:].upper())
    ax.set_ylabel(r"model $C_{%s}$ (pF)" % cap[1:].upper())
    ax.set_title(f"{cap.upper()} test set  (R$^2$={m['test_r2']:.3f}, "
                 f"RMSE={m['test_rmse_pF']:.2f} pF)")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_ROOT, f"cap_{cap}", "plots", "scatter_C.png")
    fig.savefig(out)
    plt.close(fig)


def cv_curves(cap, model, y_mean, y_std):
    """Full measured dataset (all splits) overlaid with the model curve, one
    panel per geometry -- shows how well the single ANN captures every C-VG
    curve simultaneously."""
    df = pd.read_csv(DATA_PATH)
    df = df[df.cap_type == cap]
    geoms = sorted(set(zip(df.W, df.L)))

    ncol = len(geoms)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2), dpi=140)
    fig.patch.set_facecolor(BG)
    if ncol == 1:
        axes = [axes]

    for ax, (w, l) in zip(axes, geoms):
        sub = df[(df.W == w) & (df.L == l)].sort_values("VG")
        vd0 = float(sub.VD.iloc[0])
        ax.scatter(sub.VG, sub.C_pF, s=16, color=MEASURED, alpha=0.7, label="measured")
        vg_fine = np.linspace(sub.VG.min(), sub.VG.max(), 200)
        c_fine = predict_c(model, y_mean, y_std, vg_fine,
                           np.full_like(vg_fine, vd0),
                           np.full_like(vg_fine, w), np.full_like(vg_fine, l))
        ax.plot(vg_fine, c_fine, color=MODEL, lw=2, label="ANN")
        ax.set_xlabel("VG (V)")
        ax.set_ylabel(r"$C_{%s}$ (pF)" % cap[1:].upper())
        ax.set_title(f"W={w:g}, L={l:g} um (VD={vd0:g}V)")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cap.upper()} ANN vs measurement (dots=measured, line=model)", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_ROOT, f"cap_{cap}", "plots", "C_vg_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    for cap in ("cgd", "cgs"):
        plot_dir = os.path.join(OUT_ROOT, f"cap_{cap}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        model, y_mean, y_std, m = load_model(cap)
        _, _, test = load_and_split(DATA_PATH, cap_type=cap, seed=42)
        d = np.load(os.path.join(OUT_ROOT, f"cap_{cap}", "test_predictions.npz"))
        scatter_plot(cap, d, m)
        cv_curves(cap, model, y_mean, y_std)
        print(f"[{cap}] saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
