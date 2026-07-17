"""Generate diagnostic plots for the trained TFT ANN model.

Produces:
  outputs/plots/scatter_log_id.png   -- predicted vs true log10|ID| on test set
  outputs/plots/id_vd_curves.png     -- ID-VD output characteristics, model vs data
  outputs/plots/id_vg_curves.png     -- ID-VG transfer characteristics, model vs data
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import FEATURE_BOUNDS, load_and_split
from src.model import TFTNet

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned", "merged_ann_dataset.csv")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

BG = "#f7f7f5"
FG = "#1f1f1f"
MEASURED = "#4c72b0"
MODEL = "#c44e52"


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.25)


def load_model(n_hidden=22):
    model = TFTNet(n_inputs=4, n_hidden=n_hidden, n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model_weights.pt")))
    model.eval()
    return model


def scale_point(vg, vd, w, l):
    def s(name, val):
        lo, hi = FEATURE_BOUNDS[name]
        return (val - lo) / (hi - lo)
    return np.array([s("VG", vg), s("VD", vd), s("W", w), s("L", l)], dtype=np.float32)


def predict_id(model, y_mean, y_std, vg, vd, w, l):
    X = np.stack([scale_point(a, b, c, d) for a, b, c, d in zip(vg, vd, w, l)])
    with torch.no_grad():
        pred_std = model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    log_id = pred_std * y_std + y_mean
    return 10.0 ** log_id


def scatter_plot(test, d):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.scatter(d["log_id_true"], d["log_id_pred"], s=4, alpha=0.15, color=MEASURED,
               edgecolors="none")
    lo, hi = -14, -1.5
    ax.plot([lo, hi], [lo, hi], color=FG, lw=1, ls="--", alpha=0.6, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"measured $\log_{10}|I_D|$")
    ax.set_ylabel(r"model $\log_{10}|I_D|$")
    ax.set_title("Test set: predicted vs measured drain current (log scale)")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "scatter_log_id.png"))
    plt.close(fig)


def id_vd_curves(test, model, y_mean, y_std):
    df = test.df
    combo_counts = df.groupby(["W", "L"]).size().sort_values(ascending=False)
    combos = combo_counts.index[:2]

    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 5), dpi=140)
    fig.patch.set_facecolor(BG)
    if len(combos) == 1:
        axes = [axes]

    for ax, (w, l) in zip(axes, combos):
        sub = df[(df.W == w) & (df.L == l)]
        vg_levels = sorted(sub.VG.unique())
        vg_pick = [vg_levels[i] for i in np.linspace(0, len(vg_levels) - 1, 5).astype(int)]
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(vg_pick)))
        for vg, color in zip(vg_pick, cmap):
            s = sub[sub.VG == vg].sort_values("VD")
            if len(s) < 2:
                continue
            ax.scatter(s.VD, s.ID * 1e6, s=10, color=color, alpha=0.5)
            vd_fine = np.linspace(FEATURE_BOUNDS["VD"][0], FEATURE_BOUNDS["VD"][1], 100)
            id_fine = predict_id(model, y_mean, y_std, np.full_like(vd_fine, vg), vd_fine,
                                  np.full_like(vd_fine, w), np.full_like(vd_fine, l))
            ax.plot(vd_fine, id_fine * 1e6, color=color, lw=2, label=f"VG={vg:g}V")
        ax.set_xlabel("VD (V)")
        ax.set_ylabel(r"$I_D$ ($\mu$A)")
        ax.set_title(f"W={w:g}um, L={l:g}um  (dots=measured, lines=model)")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "id_vd_curves.png"))
    plt.close(fig)


def id_vg_curves(test, model, y_mean, y_std):
    df = test.df
    combo_counts = df.groupby(["W", "L"]).size().sort_values(ascending=False)
    combos = combo_counts.index[:2]

    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 5), dpi=140)
    fig.patch.set_facecolor(BG)
    if len(combos) == 1:
        axes = [axes]

    for ax, (w, l) in zip(axes, combos):
        sub = df[(df.W == w) & (df.L == l)]
        vd_levels = sorted(sub.VD.unique())
        vd_pick = [vd_levels[i] for i in np.linspace(0, len(vd_levels) - 1, 4).astype(int)]
        cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(vd_pick)))
        for vd, color in zip(vd_pick, cmap):
            s = sub[sub.VD == vd].sort_values("VG")
            if len(s) < 2:
                continue
            ax.scatter(s.VG, np.abs(s.ID) * 1e6, s=10, color=color, alpha=0.5)
            vg_fine = np.linspace(FEATURE_BOUNDS["VG"][0], FEATURE_BOUNDS["VG"][1], 150)
            id_fine = predict_id(model, y_mean, y_std, vg_fine, np.full_like(vg_fine, vd),
                                  np.full_like(vg_fine, w), np.full_like(vg_fine, l))
            ax.plot(vg_fine, id_fine * 1e6, color=color, lw=2, label=f"VD={vd:g}V")
        ax.set_xlabel("VG (V)")
        ax.set_ylabel(r"$|I_D|$ ($\mu$A)")
        ax.set_yscale("log")
        ax.set_title(f"W={w:g}um, L={l:g}um  (dots=measured, lines=model)")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "id_vg_curves.png"))
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    train, val, test = load_and_split(DATA_PATH, seed=42)
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    y_mean = metrics["target_standardization"]["y_mean_log10_absID"]
    y_std = metrics["target_standardization"]["y_std_log10_absID"]

    d = np.load(os.path.join(OUT_DIR, "test_predictions.npz"))
    model = load_model(metrics["n_hidden"])

    scatter_plot(test, d)
    id_vd_curves(test, model, y_mean, y_std)
    id_vg_curves(test, model, y_mean, y_std)
    print("Saved plots to", PLOT_DIR)


if __name__ == "__main__":
    main()
