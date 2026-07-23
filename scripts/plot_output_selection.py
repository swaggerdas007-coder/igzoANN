"""Diagnostic plots for scripts/select_clean_output_curves.py.

  - all_output_families_raw.png: one panel per (W, L), every raw device's
    FULL VG family plotted as a same-colored bundle (one color per device),
    so you can visually compare which device's family is cleanest/most
    uniformly spaced vs. noisy/crossing/dead. The chosen winner's bundle is
    drawn solid and on top; the rest are drawn thin and faded.
  - cleaned_output_meas.png: the 19 chosen output families only, one panel
    per geometry, VG members colored by a shared colormap (matches the
    style used for data_cleaned/'s cleaned_id_vd.png).

Run after scripts/select_clean_output_curves.py: python scripts/plot_output_selection.py
"""
import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from clean_transistor_curves import inventory  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "cleaned_output_meas")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE_COLORS = {
    "top1": "#4C72B0", "top1rep": "#8CA9D6",
    "top2": "#DD8452", "top2rep": "#F0B285",
    "bot1": "#55A868", "bot1rep": "#9BCBAA",
    "bot2": "#C44E52", "bot2rep": "#E29A9D",
}
DEVICE_ORDER = ["top1", "top1rep", "top2", "top2rep", "bot1", "bot1rep", "bot2", "bot2rep"]


def grid_axes(n):
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(26, 4.6 * nrows))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def plot_all_raw():
    recs = inventory()
    manifest = json.load(open(os.path.join(OUT_DIR, "output_selection_manifest.json")))
    winners = {(m["W"], m["L"]): m["chosen_device"] for m in manifest}

    combos = sorted({(W, L) for (W, L, _dev) in recs.keys()})
    fig, axes = grid_axes(len(combos))
    for ax, (W, L) in zip(axes, combos):
        winner = winners[(W, L)]
        devices_here = sorted({dev for (w, l, dev) in recs.keys() if w == W and l == L})
        for dev in DEVICE_ORDER:
            if dev not in devices_here or "output" not in recs[(W, L, dev)]:
                continue
            df = pd.read_csv(recs[(W, L, dev)]["output"])
            is_winner = dev == winner
            for vgv in sorted(df["VG"].unique()):
                curve = df[df["VG"] == vgv].sort_values("VD")
                ax.plot(curve["VD"], curve["ID"].abs(),
                        color=DEVICE_COLORS[dev],
                        lw=1.6 if is_winner else 0.6,
                        alpha=0.95 if is_winner else 0.35,
                        zorder=3 if is_winner else 1)
        # legend proxies (one line per device present)
        for dev in DEVICE_ORDER:
            if dev not in devices_here:
                continue
            label = f"{dev}{' [chosen]' if dev == winner else ''}"
            ax.plot([], [], color=DEVICE_COLORS[dev],
                    lw=1.6 if dev == winner else 0.6, label=label)
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 1e-2)
        ax.set_title(f"W={W} L={L}", fontsize=10)
        ax.set_xlabel("VD (V)", fontsize=8)
        ax.set_ylabel("|ID| (A)", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
    fig.suptitle("RAW output curve families by device (bold/opaque = chosen device, "
                 "faint = other candidates) -- one color per device", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "all_output_families_raw.png"), dpi=110)
    plt.close(fig)


def plot_chosen():
    manifest = json.load(open(os.path.join(OUT_DIR, "output_selection_manifest.json")))
    combos = sorted((m["W"], m["L"]) for m in manifest)
    fig, axes = grid_axes(len(combos))
    cmap = plt.get_cmap("viridis")
    for ax, (W, L) in zip(axes, combos):
        df = pd.read_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_output_clean.csv"))
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
    fig.suptitle("cleaned_output_meas: chosen cleanest/most-uniform output family per geometry", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "cleaned_output_meas.png"), dpi=110)
    plt.close(fig)


def main():
    plot_all_raw()
    plot_chosen()
    print(f"saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
