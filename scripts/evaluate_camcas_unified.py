"""Score the unified CAMCAS model (scripts/fit_camcas_unified.py) against this
repo's ANN and the thesis-style (L, W) polynomial-surface CAMCAS fit
(scripts/fit_camcas_model.py), on identical points with identical metrics:
log10|ID| MAE / RMSE / R^2 (the ANN's own metrics, outputs/metrics.json) and
on-state median relative error, broken down by geometry and sweep.

Usage:
    python scripts/evaluate_camcas_unified.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from camcas_unified import UnifiedParams, ids as camcas_ids  # noqa: E402
from plot_camcas_fit import CamcasModel  # noqa: E402
from src.dataset import load_and_split, scale_features  # noqa: E402
from src.model import TFTNet  # noqa: E402

DATA = os.path.join(REPO, "data_cleaned_2", "merged_ann_dataset.csv")
OUT_DIR = os.path.join(REPO, "outputs_camcas")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

INK = "#1f1f1f"
MUTED = "#8a8a85"
C_CAMCAS = "#2a78d6"
C_ANN = "#eb6834"
ON_STATE = 1e-11


def load_ann():
    with open(os.path.join(REPO, "outputs", "metrics.json")) as f:
        meta = json.load(f)
    model = TFTNet(n_inputs=4, n_hidden=meta["n_hidden"], n_outputs=1)
    model.load_state_dict(torch.load(os.path.join(REPO, "outputs", "model_weights.pt")))
    model.eval()
    ts = meta["target_standardization"]

    def predict(df):
        with torch.no_grad():
            z = model(torch.as_tensor(scale_features(df))).squeeze(1).numpy()
        return 10.0 ** (z * ts["y_std_log10_absID"] + ts["y_mean_log10_absID"])
    return predict


def thesis_surface_predict(df):
    with open(os.path.join(OUT_DIR, "camcas_coefficients.json")) as f:
        model = CamcasModel(json.load(f))
    out = np.empty(len(df))
    for (W, L, vd), idx in df.groupby(["W", "L", "VD"]).groups.items():
        rows = df.loc[idx]
        out[df.index.get_indexer(idx)] = model.ids(rows.VG.to_numpy(), vd, L, W)
    return np.maximum(np.abs(out), 1e-15)


def log_metrics(pred, log_true):
    err = np.log10(pred) - log_true
    return {"mae_dec": np.mean(np.abs(err)), "rmse_dec": np.sqrt(np.mean(err ** 2)),
            "r2": 1 - np.sum(err ** 2) / np.sum((log_true - log_true.mean()) ** 2)}


def median_rel_err(pred, true):
    on = np.abs(true) > ON_STATE
    return float(np.median(np.abs(pred[on] - true[on]) / np.abs(true[on]))) if on.any() else np.nan


def style(ax):
    ax.grid(True, color="#e4e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#4a4a46", labelsize=7)


def transfer_grid(df, sweep, golden, fname, title):
    geoms = sorted(df.groupby(["W", "L"]).groups)
    fig, axes = plt.subplots(4, 5, figsize=(17, 13), sharex=True, sharey=True)
    for ax, (W, L) in zip(axes.ravel(), geoms):
        s = df[(df.W == W) & (df.L == L) & (df.sweep == sweep)].sort_values("VG")
        style(ax)
        ax.semilogy(s.VG, s.abs_ID, "o", ms=3, color=INK, alpha=0.55, label="measured")
        ax.semilogy(s.VG, s.camcas, "-", lw=2, color=C_CAMCAS, label="CAMCAS unified")
        ax.semilogy(s.VG, s.ann, "--", lw=2, color=C_ANN, label="ANN")
        tag = "golden (fit)" if (W, L) in golden else "not in fit"
        ax.set_title(f"W={W:g} L={L:g} um  |  {tag}", fontsize=8.5,
                     color=INK if (W, L) in golden else MUTED)
        ax.set_ylim(1e-13, 1e-3)
    for ax in axes.ravel()[len(geoms):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("VG (V)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("|ID| (A)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)


def output_panels(df, geoms, fname):
    fig, axes = plt.subplots(1, len(geoms), figsize=(4.4 * len(geoms), 4.2), sharey=False)
    for ax, (W, L) in zip(axes, geoms):
        style(ax)
        s = df[(df.W == W) & (df.L == L) & (df.sweep == "output") & (df.VG >= 1)]
        for vg, g in s.groupby("VG"):
            g = g.sort_values("VD")
            ax.plot(g.VD, g.ID * 1e6, "o", ms=2.5, color=INK, alpha=0.45)
            ax.plot(g.VD, g.camcas * 1e6, "-", lw=2, color=C_CAMCAS)
            ax.plot(g.VD, g.ann * 1e6, "--", lw=2, color=C_ANN)
            ax.annotate(f"VG={vg:g}", (g.VD.iloc[-1], g.ID.iloc[-1] * 1e6), fontsize=6.5,
                        color="#4a4a46", xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_title(f"W={W:g} L={L:g} um", fontsize=9, color=INK)
        ax.set_xlabel("VD (V)", fontsize=8)
        ax.set_xlim(0, 5.7)
    axes[0].set_ylabel("ID (uA)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.6, label="measured"),
               plt.Line2D([], [], color=C_CAMCAS, lw=2, label="CAMCAS unified"),
               plt.Line2D([], [], color=C_ANN, lw=2, ls="--", label="ANN")]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Output curves, VG = 1-5 V", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close(fig)


def mae_bars(per_geom, fname):
    per_geom = per_geom.sort_values(["golden", "W", "L"], ascending=[False, True, True])
    labels = [f"{w:g}/{l:g}" for w, l in zip(per_geom.W, per_geom.L)]
    x = np.arange(len(per_geom))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    style(ax)
    ax.bar(x - 0.21, per_geom.camcas_mae_dec, 0.38, color=C_CAMCAS, label="CAMCAS unified",
           edgecolor="white", linewidth=1)
    ax.bar(x + 0.21, per_geom.ann_mae_dec, 0.38, color=C_ANN, label="ANN",
           edgecolor="white", linewidth=1)
    n_gold = int(per_geom.golden.sum())
    ax.axvline(n_gold - 0.5, color=MUTED, lw=1, ls=":")
    ax.text(n_gold / 2 - 0.5, ax.get_ylim()[1] * 0.95, "golden devices (fit)", ha="center",
            fontsize=8, color="#4a4a46")
    ax.text(n_gold + (len(x) - n_gold) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
            "atypical devices (not fit)", ha="center", fontsize=8, color="#4a4a46")
    ax.set_xticks(x, labels, rotation=0, fontsize=7.5)
    ax.set_xlabel("W / L (um)", fontsize=8)
    ax.set_ylabel("MAE, log10|ID| (decades)", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Per-geometry error, all sweeps (lower is better)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "camcas_unified_params.json")) as f:
        fitted = json.load(f)
    p = UnifiedParams(**fitted["params"])
    golden = {tuple(g) for g in fitted["golden_devices"]}

    df = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
    df["camcas"] = camcas_ids(df.VG, df.VD, df.W, df.L, p)
    df["ann"] = load_ann()(df)
    df["thesis"] = thesis_surface_predict(df)
    df["golden"] = [(w, l) in golden for w, l in zip(df.W, df.L)]

    _, _, test = load_and_split(DATA)
    test_keys = set(map(tuple, test.df[["VG", "VD", "W", "L"]].to_numpy()))
    df["ann_test"] = [k in test_keys for k in map(tuple, df[["VG", "VD", "W", "L"]].to_numpy())]

    models = {"camcas": "CAMCAS unified", "ann": "ANN", "thesis": "CAMCAS thesis-style surface"}
    subsets = {
        "all 19 geometries": df,
        "golden devices": df[df.golden],
        "atypical devices": df[~df.golden],
        "ANN test split": df[df.ann_test],
    }
    rows = []
    for sname, sub in subsets.items():
        for sweep in ("all", "linear", "saturation", "output"):
            ss = sub if sweep == "all" else sub[sub.sweep == sweep]
            for key, label in models.items():
                m = log_metrics(ss[key].to_numpy(), ss.log_ID.to_numpy())
                rows.append({"subset": sname, "sweep": sweep, "model": label, "n": len(ss),
                             **m, "median_rel_err_on": median_rel_err(ss[key].to_numpy(),
                                                                      ss.ID.to_numpy())})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "unified_vs_ann_summary.csv"), index=False)

    per_geom = []
    for (W, L), g in df.groupby(["W", "L"]):
        r = {"W": W, "L": L, "golden": (W, L) in golden}
        for key in models:
            r[f"{key}_mae_dec"] = log_metrics(g[key].to_numpy(), g.log_ID.to_numpy())["mae_dec"]
            for sweep in ("linear", "saturation"):
                gs = g[g.sweep == sweep]
                r[f"{key}_relerr_{sweep[:3]}"] = median_rel_err(gs[key].to_numpy(), gs.ID.to_numpy())
        per_geom.append(r)
    per_geom = pd.DataFrame(per_geom)
    per_geom.to_csv(os.path.join(OUT_DIR, "unified_vs_ann_per_geometry.csv"), index=False)

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.3f}"
    print(summary[summary.sweep == "all"].to_string(index=False, float_format=fmt))
    print()
    print(summary[summary.subset == "all 19 geometries"].to_string(index=False, float_format=fmt))
    print()
    print(per_geom.to_string(index=False, float_format=fmt))

    transfer_grid(df, "linear", golden, "unified_transfer_linear.png",
                  "Linear transfer (VD = 0.1 V): measured vs unified CAMCAS vs ANN")
    transfer_grid(df, "saturation", golden, "unified_transfer_saturation.png",
                  "Saturation transfer (VD = 5 V): measured vs unified CAMCAS vs ANN")
    output_panels(df, [(20, 20), (40, 10), (80, 5), (160, 20)], "unified_output_curves.png")
    mae_bars(per_geom, "unified_vs_ann_per_geometry.png")
    print(f"\nWrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
