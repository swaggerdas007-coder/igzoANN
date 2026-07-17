"""Overview plots for scripts/select_best_devices.py's data_cleaned_2/:
one curve per (W, L) geometry (the single chosen best device) for each of
linear, saturation, and output sweeps.

Run after scripts/select_best_devices.py: python scripts/plot_best_devices.py
"""
import glob
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_DIR = os.path.join(REPO_ROOT, "data_cleaned_2")
PLOT_DIR = os.path.join(BEST_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def combos():
    return sorted({
        tuple(map(int, re.match(r"W(\d+)_L(\d+)_linear_best\.csv", os.path.basename(f)).groups()))
        for f in glob.glob(os.path.join(BEST_DIR, "W*_linear_best.csv"))
    })


def plot_transfer(kind, vd_label, out_name):
    cs = combos()
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (W, L) in enumerate(cs):
        df = pd.read_csv(os.path.join(BEST_DIR, f"W{W}_L{L}_{kind}_best.csv")).sort_values("VG")
        dev = df["device"].iloc[0]
        ax.plot(df["VG"], df["ID"].abs(), color=cmap(i / max(1, len(cs) - 1)), lw=1.4,
                label=f"W={W} L={L} ({dev})")
    ax.set_yscale("log")
    ax.set_xlabel("VG (V)")
    ax.set_ylabel("|ID| (A)")
    ax.set_title(f"data_cleaned_2: best-device Id-Vg {kind} curves ({vd_label}, VG in [-2,5])")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, out_name), dpi=130)
    plt.close(fig)


def plot_output_grid():
    cs = combos()
    ncols = 5
    nrows = -(-len(cs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(26, 4.6 * nrows))
    axes = axes.flatten()
    for ax in axes[len(cs):]:
        ax.axis("off")
    cmap = plt.get_cmap("viridis")
    for ax, (W, L) in zip(axes, cs):
        df = pd.read_csv(os.path.join(BEST_DIR, f"W{W}_L{L}_output_best.csv"))
        dev = df["device"].iloc[0]
        vg_levels = sorted(df["VG"].unique())
        for i, vgv in enumerate(vg_levels):
            curve = df[df["VG"] == vgv].sort_values("VD")
            ax.plot(curve["VD"], curve["ID"].abs(), color=cmap(i / max(1, len(vg_levels) - 1)),
                    lw=1.2, label=f"VG={vgv:g}" if i % 2 == 0 else None)
        ax.set_title(f"W={W} L={L} (device={dev})", fontsize=10)
        ax.set_xlabel("VD (V)", fontsize=8)
        ax.set_ylabel("|ID| (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5, loc="upper left", ncol=2)
    fig.suptitle("data_cleaned_2: best-device Id-Vd output curve families", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "best_id_vd.png"), dpi=110)
    plt.close(fig)


def main():
    plot_transfer("linear", "VD=0.1V", "best_id_vg_linear.png")
    plot_transfer("saturation", "VD=5V", "best_id_vg_saturation.png")
    plot_output_grid()
    print(f"saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
