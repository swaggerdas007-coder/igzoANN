"""Diagnostic and verification plots for scripts/clean_transistor_curves.py.

Produces, for each of linear (Id-Vg @ VD=0.1V), saturation (Id-Vg @ VD=5V),
and output (Id-Vd family) sweeps, a before/after pair of overview grids (all
19 W,L geometries as small multiples):
  - overview_id_vg_linear_raw.png / overview_id_vg_saturation_raw.png /
    overview_id_vd_raw.png: every raw device replicate (VG sweeps already
    trimmed to the forward -2..5V leg), flagged ones dashed and labeled
    [BAD].
  - cleaned_id_vg_linear.png / cleaned_id_vg_saturation.png /
    cleaned_id_vd.png: only the retained data that ships in data_cleaned/.

Run after scripts/clean_transistor_curves.py: python scripts/plot_transistor_curves.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from clean_transistor_curves import load_kind  # noqa: E402

CLEAN_DIR = os.path.join(REPO_ROOT, "data_cleaned")
PLOT_DIR = os.path.join(CLEAN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Fixed categorical color assignment (never cycled) shared by every panel.
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


def plot_raw_transfer_overview(results, kind, vd_label, out_name):
    combos = sorted({(v["W"], v["L"]) for v in results.values()})
    fig, axes = grid_axes(len(combos))
    for ax, (W, L) in zip(axes, combos):
        for dev in DEVICE_ORDER:
            key = f"W{W}_L{L}_{dev}"
            if key not in results or f"{kind}_path" not in results[key]:
                continue
            r = results[key]
            df = load_kind(r[f"{kind}_path"], kind).sort_values("VG")
            ok = r[f"{kind}_ok"]
            ax.plot(df["VG"], df["ID"].abs(), "-" if ok else "--", color=DEVICE_COLORS[dev],
                    lw=1.2, label=f"{dev}{'' if ok else ' [BAD]'}", alpha=0.9)
        ax.set_yscale("log")
        ax.set_title(f"W={W} L={L}", fontsize=10)
        ax.set_xlabel("VG (V)", fontsize=8)
        ax.set_ylabel("|ID| (A)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"RAW Id-Vg {kind} curves ({vd_label}, VG trimmed to forward -2..5V) "
                 "— dashed/[BAD] = flagged & dropped", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, out_name), dpi=110)
    plt.close(fig)


def plot_raw_output_overview(results):
    combos = sorted({(v["W"], v["L"]) for v in results.values()})
    fig, axes = grid_axes(len(combos))
    for ax, (W, L) in zip(axes, combos):
        for dev in DEVICE_ORDER:
            key = f"W{W}_L{L}_{dev}"
            if key not in results or "output_path" not in results[key]:
                continue
            r = results[key]
            df = pd.read_csv(r["output_path"])
            ok = r["output_ok"]
            vg_top = sorted(df["VG"].unique())[-1]
            sub = df[df["VG"] == vg_top].sort_values("VD")
            ax.plot(sub["VD"], sub["ID"].abs(), "-" if ok else "--", color=DEVICE_COLORS[dev],
                    lw=1.2, label=f"{dev} VG={vg_top:g}{'' if ok else ' [BAD]'}", alpha=0.9)
        ax.set_title(f"W={W} L={L}", fontsize=10)
        ax.set_xlabel("VD (V)", fontsize=8)
        ax.set_ylabel("|ID| at max VG (A)", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=7)
    fig.suptitle("RAW Id-Vd output curves (top VG member) — dashed/[BAD] = flagged & dropped", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "overview_id_vd_raw.png"), dpi=110)
    plt.close(fig)


def _clean_combos():
    return sorted({
        tuple(map(int, re.match(r"W(\d+)_L(\d+)_linear_clean\.csv", os.path.basename(f)).groups()))
        for f in glob.glob(os.path.join(CLEAN_DIR, "W*_linear_clean.csv"))
    })


def plot_cleaned_transfer(kind, vd_label, out_name):
    combos = _clean_combos()
    fig, axes = grid_axes(len(combos))
    for ax, (W, L) in zip(axes, combos):
        df = pd.read_csv(os.path.join(CLEAN_DIR, f"W{W}_L{L}_{kind}_clean.csv"))
        for dev, sub in df.groupby("device"):
            sub = sub.sort_values("VG")
            ax.plot(sub["VG"], sub["ID"].abs(), color=DEVICE_COLORS.get(dev, "gray"), lw=1.3, label=dev)
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 1e-2)
        ax.set_title(f"W={W} L={L}", fontsize=10)
        ax.set_xlabel("VG (V)", fontsize=8)
        ax.set_ylabel("|ID| (A)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle(f"CLEANED Id-Vg {kind} curves ({vd_label}, VG in [-2,5], forward leg only) "
                 "— retained device replicates only", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, out_name), dpi=110)
    plt.close(fig)


def plot_cleaned_output():
    combos = _clean_combos()
    fig, axes = grid_axes(len(combos))
    cmap = plt.get_cmap("viridis")
    for ax, (W, L) in zip(axes, combos):
        df = pd.read_csv(os.path.join(CLEAN_DIR, f"W{W}_L{L}_output_clean.csv"))
        dev = sorted(df["device"].unique())[0]
        sub_dev = df[df["device"] == dev]
        vg_levels = sorted(sub_dev["VG"].unique())
        for i, vgv in enumerate(vg_levels):
            curve = sub_dev[sub_dev["VG"] == vgv].sort_values("VD")
            ax.plot(curve["VD"], curve["ID"].abs(), color=cmap(i / max(1, len(vg_levels) - 1)),
                    lw=1.2, label=f"VG={vgv:g}" if i % 2 == 0 else None)
        ax.set_title(f"W={W} L={L}  (device={dev})", fontsize=10)
        ax.set_xlabel("VD (V)", fontsize=8)
        ax.set_ylabel("|ID| (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5, loc="upper left", ncol=2)
    fig.suptitle("CLEANED Id-Vd output curve families (one representative retained device per W,L)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "cleaned_id_vd.png"), dpi=110)
    plt.close(fig)


def main():
    results = json.load(open(os.path.join(CLEAN_DIR, "qc_results.json")))
    plot_raw_transfer_overview(results, "linear", "VD=0.1V", "overview_id_vg_linear_raw.png")
    plot_raw_transfer_overview(results, "saturation", "VD=5V", "overview_id_vg_saturation_raw.png")
    plot_raw_output_overview(results)
    plot_cleaned_transfer("linear", "VD=0.1V", "cleaned_id_vg_linear.png")
    plot_cleaned_transfer("saturation", "VD=5V", "cleaned_id_vg_saturation.png")
    plot_cleaned_output()
    print(f"saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
