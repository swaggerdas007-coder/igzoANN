"""Score the unified CAMCAS model (scripts/fit_camcas_unified.py) against the
deep ID ANN in verilogA/ntft_full.va (4 -> 32 -> 16 -> 1, trained by
trained_ANN/ann_train_deep.py) and the thesis-style (L, W) polynomial-surface
CAMCAS fit (scripts/fit_camcas_model.py), with identical metrics: log10|ID|
MAE / RMSE / R^2 and on-state median relative error.

The two models were fit to different physical devices for 14 of the 19
geometries: CAMCAS to data_cleaned_2 (one device per geometry picked on its
transfer curves), the ANN to cleaned_output_meas (one device per geometry
picked on its output curves, output sweeps only). So both models are scored
on BOTH datasets, and the output-curve panels use the 4 typical geometries
where the two datasets share the same measured device.

The ANN is evaluated straight from the weights in the .va file (same
min-max input scaling and clamping as the Verilog-A forward pass).

Usage:
    python scripts/evaluate_camcas_unified.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from camcas_unified import UnifiedParams, ids as camcas_ids  # noqa: E402
from plot_camcas_fit import CamcasModel  # noqa: E402

DATA_CAMCAS = os.path.join(REPO, "data_cleaned_2", "merged_ann_dataset.csv")
DATA_ANN = os.path.join(REPO, "cleaned_output_meas", "merged_output_dataset.csv")
ANN_VA = os.path.join(REPO, "verilogA", "ntft_full.va")
OUT_DIR = os.path.join(REPO, "outputs_camcas")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

INK = "#1f1f1f"
MUTED = "#8a8a85"
C_CAMCAS = "#2a78d6"
C_ANN = "#eb6834"
ON_STATE = 1e-11
MODELS = {"camcas": "CAMCAS unified", "ann": "ANN (ntft_full.va)",
          "thesis": "CAMCAS thesis-style surface"}


def load_ann():
    va = open(ANN_VA).read()

    def arr(name):
        vals = {int(i): float(v) for i, v in
                re.findall(rf"\b{name}\[(\d+)\]\s*=\s*([-\d.eE+]+);", va)}
        return np.array([vals[i] for i in range(len(vals))])

    def scalar(name):
        return float(re.search(rf"\b{name}\s*=\s*([-\d.eE+]+);", va).group(1))

    b1, b2 = arr("id_b1"), arr("id_b2")
    w1 = arr("id_w1").reshape(len(b1), 4)
    w2 = arr("id_w2").reshape(len(b2), len(b1))
    wo, bo = arr("id_wo"), scalar("id_bo")
    mean, std = scalar("id_y_mean"), scalar("id_y_std")
    bounds = {"VG": (-5.0, 5.0), "VD": (0.0, 5.0), "W": (5.0, 160.0), "L": (5.0, 20.0)}

    def predict(df):
        x = np.stack([np.clip((df[c].to_numpy(float) - lo) / (hi - lo), 0.0, 1.0)
                      for c, (lo, hi) in bounds.items()], axis=1)
        y1 = np.tanh(x @ w1.T + b1)
        y2 = np.tanh(y1 @ w2.T + b2)
        return 10.0 ** ((y2 @ wo + bo) * std + mean)
    return predict


def thesis_surface_predict(df):
    with open(os.path.join(OUT_DIR, "camcas_coefficients.json")) as f:
        model = CamcasModel(json.load(f))
    out = np.empty(len(df))
    for (W, L, vd), idx in df.groupby(["W", "L", "VD"]).groups.items():
        out[df.index.get_indexer(idx)] = model.ids(df.loc[idx, "VG"].to_numpy(), vd, L, W)
    return np.maximum(np.abs(out), 1e-15)


def load(path, golden, predict_ann, p):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    if "sweep" not in df:
        df["sweep"] = "output"
    df["camcas"] = camcas_ids(df.VG, df.VD, df.W, df.L, p)
    df["ann"] = predict_ann(df)
    df["thesis"] = thesis_surface_predict(df)
    df["golden"] = [(w, l) in golden for w, l in zip(df.W, df.L)]
    return df


def log_metrics(pred, log_true):
    err = np.log10(pred) - log_true
    return {"mae_dec": np.mean(np.abs(err)), "rmse_dec": np.sqrt(np.mean(err ** 2)),
            "r2": 1 - np.sum(err ** 2) / np.sum((log_true - log_true.mean()) ** 2)}


def median_rel_err(pred, true):
    on = np.abs(true) > ON_STATE
    return float(np.median(np.abs(pred[on] - true[on]) / np.abs(true[on]))) if on.any() else np.nan


def summarize(df, dataset):
    rows = []
    subsets = {"all 19": df, "golden 13": df[df.golden], "atypical 6": df[~df.golden]}
    for sname, sub in subsets.items():
        for sweep in ["all"] + sorted(sub.sweep.unique()):
            ss = sub if sweep == "all" else sub[sub.sweep == sweep]
            for key, label in MODELS.items():
                rows.append({"dataset": dataset, "subset": sname, "sweep": sweep, "model": label,
                             "n": len(ss), **log_metrics(ss[key].to_numpy(), ss.log_ID.to_numpy()),
                             "median_rel_err_on": median_rel_err(ss[key].to_numpy(),
                                                                 ss.ID.to_numpy())})
    return rows


def per_geometry(df, dataset):
    rows = []
    for (W, L), g in df.groupby(["W", "L"]):
        r = {"dataset": dataset, "W": W, "L": L, "golden": bool(g.golden.iloc[0])}
        for key in MODELS:
            r[f"{key}_mae_dec"] = log_metrics(g[key].to_numpy(), g.log_ID.to_numpy())["mae_dec"]
        rows.append(r)
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(True, color="#e4e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#4a4a46", labelsize=7)


def transfer_grid(df, sweep, same_device, fname, title):
    geoms = sorted(df.groupby(["W", "L"]).groups)
    fig, axes = plt.subplots(4, 5, figsize=(17, 13), sharex=True, sharey=True)
    for ax, (W, L) in zip(axes.ravel(), geoms):
        s = df[(df.W == W) & (df.L == L) & (df.sweep == sweep)].sort_values("VG")
        golden = bool(s.golden.iloc[0])
        style(ax)
        ax.semilogy(s.VG, s.abs_ID, "o", ms=3, color=INK, alpha=0.55, label="measured")
        ax.semilogy(s.VG, s.camcas, "-", lw=2, color=C_CAMCAS, label="CAMCAS unified")
        ax.semilogy(s.VG, s.ann, "--", lw=2, color=C_ANN, label="ANN (ntft_full.va)")
        tag = ("golden" if golden else "not in CAMCAS fit") + \
              ("" if (W, L) in same_device else "  |  ANN: other device")
        ax.set_title(f"W={W:g} L={L:g} um  |  {tag}", fontsize=8,
                     color=INK if golden else MUTED)
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


def output_panels(df, geoms, fname, title):
    fig, axes = plt.subplots(1, len(geoms), figsize=(4.4 * len(geoms), 4.2))
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
               plt.Line2D([], [], color=C_ANN, lw=2, ls="--", label="ANN (ntft_full.va)")]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close(fig)


def mae_bars(per_geom, same_device, fname):
    datasets = [("data_cleaned_2", "Scored on CAMCAS's fit devices (data_cleaned_2: linear + "
                 "saturation transfer + output curves)"),
                ("cleaned_output_meas", "Scored on the ANN's training devices "
                 "(cleaned_output_meas: output curves)")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    for ax, (ds, title) in zip(axes, datasets):
        pg = per_geom[per_geom.dataset == ds].sort_values(["golden", "W", "L"],
                                                           ascending=[False, True, True])
        labels = [f"{w:g}/{l:g}" + ("" if (w, l) in same_device else "*")
                  for w, l in zip(pg.W, pg.L)]
        x = np.arange(len(pg))
        style(ax)
        ax.bar(x - 0.21, pg.camcas_mae_dec, 0.38, color=C_CAMCAS, label="CAMCAS unified",
               edgecolor="white", linewidth=1)
        ax.bar(x + 0.21, pg.ann_mae_dec, 0.38, color=C_ANN, label="ANN (ntft_full.va)",
               edgecolor="white", linewidth=1)
        n_gold = int(pg.golden.sum())
        ax.axvline(n_gold - 0.5, color=MUTED, lw=1, ls=":")
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_ylabel("MAE, log10|ID| (decades)", fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK)
    top = axes[0].get_ylim()[1]
    for ax in axes:
        pg = per_geom[per_geom.dataset == "data_cleaned_2"]
        n_gold = int(pg.golden.sum())
        ax.text(n_gold / 2 - 0.5, top * 0.93, "golden devices (CAMCAS fit)", ha="center",
                fontsize=8, color="#4a4a46")
        ax.text(n_gold + (len(pg) - n_gold) / 2 - 0.5, top * 0.93, "atypical devices",
                ha="center", fontsize=8, color="#4a4a46")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    axes[1].set_xlabel("W / L (um)     * = the two datasets use different physical devices "
                       "at this geometry", fontsize=8)
    fig.suptitle("Per-geometry error (lower is better)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "camcas_unified_params.json")) as f:
        fitted = json.load(f)
    p = UnifiedParams(**fitted["params"])
    golden = {tuple(g) for g in fitted["golden_devices"]}
    predict_ann = load_ann()

    d2 = load(DATA_CAMCAS, golden, predict_ann, p)
    do = load(DATA_ANN, golden, predict_ann, p)
    dev_c = d2.groupby(["W", "L"]).device.first()
    dev_a = do.groupby(["W", "L"]).device.first()
    same_device = {g for g in dev_c.index if dev_c[g] == dev_a[g]}

    summary = pd.DataFrame(summarize(d2, "data_cleaned_2") + summarize(do, "cleaned_output_meas"))
    summary.to_csv(os.path.join(OUT_DIR, "unified_vs_ann_summary.csv"), index=False)
    per_geom = pd.concat([per_geometry(d2, "data_cleaned_2"),
                          per_geometry(do, "cleaned_output_meas")], ignore_index=True)
    per_geom["same_device"] = [(w, l) in same_device for w, l in zip(per_geom.W, per_geom.L)]
    per_geom.to_csv(os.path.join(OUT_DIR, "unified_vs_ann_per_geometry.csv"), index=False)

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.3f}"
    print(summary[summary.sweep == "all"].to_string(index=False, float_format=fmt))
    print()
    print(per_geom.to_string(index=False, float_format=fmt))
    shared = sorted(g for g in same_device if g in golden)
    for name, df in (("data_cleaned_2", d2), ("cleaned_output_meas", do)):
        sub = df[[(w, l) in same_device and (w, l) in golden for w, l in zip(df.W, df.L)]]
        sub = sub[sub.sweep == "output"]
        for key in ("camcas", "ann"):
            m = log_metrics(sub[key].to_numpy(), sub.log_ID.to_numpy())["mae_dec"]
            print(f"shared-device golden output curves ({name}), {MODELS[key]}: {m:.3f} dec")

    transfer_grid(d2, "linear", same_device, "unified_transfer_linear.png",
                  "Linear transfer (VD = 0.1 V, data_cleaned_2): measured vs unified CAMCAS vs ANN")
    transfer_grid(d2, "saturation", same_device, "unified_transfer_saturation.png",
                  "Saturation transfer (VD = 5 V, data_cleaned_2): measured vs unified CAMCAS vs ANN")
    output_panels(d2, shared[:4], "unified_output_curves.png",
                  "Output curves, VG = 1-5 V -- geometries where both models were fit to the "
                  "same measured device")
    mae_bars(per_geom, same_device, "unified_vs_ann_per_geometry.png")
    print(f"\nWrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
