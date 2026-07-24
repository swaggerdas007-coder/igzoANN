"""Cadence-style DC sweep analysis for the trained ANN / Verilog-A model.

Replicates a simple DC testbench: Source always at 0V, VG swept 0->5V in
1V steps, VD swept 0->5V in 0.2V steps (26 points), for every (W, L)
geometry. This is exactly the sweep a `dc` analysis in Spectre/Cadence
would run against verilogA/tft_ann_static.va.

For each L, produces one figure with one subplot per W, overlaying the
model's I-V family (lines) against the measured data in cleaned_output_meas/
(dots) where it exists. (W, L) combinations absent from the training data
are plotted too but flagged as "no measured data (extrapolated)" -- this is
the main tool for diagnosing whether specific L values behave badly because
the model never saw examples there.

Also computes, per (W, L) and aggregated per L:
  - RMSE/MAE against measured data (log10|ID| domain)
  - gds = dID/dVD via analytic autograd-free finite differences on the
    Cadence-style grid itself, and the fraction / worst value of gds < 0
    (negative output conductance -- the thing that destabilizes SPICE).

Usage:
    python scripts/dc_sweep_analysis.py [--weights trained_ANN/weights.json]
                                         [--out-dir trained_ANN/dc_analysis]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
MEAS_DIR = os.path.join(ROOT, "cleaned_output_meas")
ALL_W = [5, 10, 20, 40, 80, 160]
ALL_L = [5, 10, 15, 20]
VG_SWEEP = np.array([0, 1, 2, 3, 4, 5], dtype=float)
VD_SWEEP = np.round(np.arange(0.0, 5.0001, 0.2), 4)

BG = "#f7f7f5"
FG = "#1f1f1f"
MEAS_COLOR = "#4c72b0"


def load_model(weights_path):
    with open(weights_path) as f:
        w = json.load(f)
    bounds = w["input_scaling_minmax"]
    y_mean = w["target_transform"]["y_mean_log10_absID"]
    y_std = w["target_transform"]["y_std_log10_absID"]

    def scale(vg, vd, wid, l):
        vg_s = np.clip((vg - bounds["VG"][0]) / (bounds["VG"][1] - bounds["VG"][0]), 0, 1)
        vd_s = np.clip((vd - bounds["VD"][0]) / (bounds["VD"][1] - bounds["VD"][0]), 0, 1)
        w_s = np.clip((wid - bounds["W"][0]) / (bounds["W"][1] - bounds["W"][0]), 0, 1)
        l_s = np.clip((l - bounds["L"][0]) / (bounds["L"][1] - bounds["L"][0]), 0, 1)
        return np.stack(np.broadcast_arrays(vg_s, vd_s, w_s, l_s), axis=-1)

    if "layers" in w:
        # multi-hidden-layer format (see ann_train_deep.py)
        layers = [(np.array(l["w"]), np.array(l["b"])) for l in w["layers"]]
        wo = np.array(w["output"]["w"][0]); bo = w["output"]["b"][0]

        def id_model(vg, vd, wid, l):
            x = scale(vg, vd, wid, l)
            for lw, lb in layers:
                x = np.tanh(x @ lw.T + lb)
            ov = x @ wo + bo
            return 10.0 ** (ov * y_std + y_mean)
    else:
        # single-hidden-layer format (see ann_train.py)
        wh = np.array(w["wh"]); bh = np.array(w["bh"])
        wo = np.array(w["wo"][0]); bo = w["bo"][0]

        def id_model(vg, vd, wid, l):
            x = scale(vg, vd, wid, l)
            hy = np.tanh(x @ wh.T + bh)
            ov = hy @ wo + bo
            return 10.0 ** (ov * y_std + y_mean)

    return id_model, w


def load_measured(w, l):
    path = os.path.join(MEAS_DIR, f"W{w}_L{l}_output_clean.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def style_axes(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.25)


def gds_stats(id_model, vg, w, l):
    """gds = dID/dVD via central finite differences on a fine VD grid, at
    fixed VG/W/L. Returns (fraction negative, worst gds in S)."""
    vd_fine = np.linspace(0.01, 4.99, 200)
    ids = id_model(np.full_like(vd_fine, vg), vd_fine, w, l)
    gds = np.gradient(ids, vd_fine)
    return float((gds < 0).mean()), float(gds.min())


def analyze_geometry(id_model, w, l):
    meas = load_measured(w, l)
    VG, VD = np.meshgrid(VG_SWEEP, VD_SWEEP, indexing="ij")
    id_model_grid = id_model(VG, VD, w, l)

    row = {"W": w, "L": l, "has_data": meas is not None}
    gds_neg_fracs, gds_worsts = [], []
    for vg in VG_SWEEP:
        frac, worst = gds_stats(id_model, vg, w, l)
        gds_neg_fracs.append(frac)
        gds_worsts.append(worst)
    row["gds_negative_fraction"] = float(np.mean(gds_neg_fracs))
    row["worst_gds_S"] = float(np.min(gds_worsts))

    if meas is not None:
        m = meas[meas.VG.isin(VG_SWEEP.astype(int))].copy()
        pred = id_model(m.VG.to_numpy(float), m.VD.to_numpy(float), w, l)
        log_true = np.log10(np.abs(m.ID.to_numpy()).clip(min=1e-15))
        log_pred = np.log10(np.abs(pred).clip(min=1e-15))
        row["n_meas_points"] = int(len(m))
        row["rmse_log10ID"] = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
        row["mae_log10ID"] = float(np.mean(np.abs(log_true - log_pred)))
    else:
        row["n_meas_points"] = 0
        row["rmse_log10ID"] = None
        row["mae_log10ID"] = None
    return row, id_model_grid


def plot_by_L(id_model, out_dir, id_tag=""):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for l in ALL_L:
        ws_with_data = sorted({w for w in ALL_W if load_measured(w, l) is not None})
        ws_to_plot = ws_with_data if ws_with_data else ALL_W
        n = len(ws_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), dpi=140, squeeze=False)
        axes = axes[0]
        fig.patch.set_facecolor(BG)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(VG_SWEEP)))

        for ax, w in zip(axes, ws_to_plot):
            row, id_grid = analyze_geometry(id_model, w, l)
            all_rows.append(row)
            for i, vg in enumerate(VG_SWEEP):
                ax.plot(VD_SWEEP, id_grid[i] * 1e6, color=cmap[i], lw=2,
                        label=f"VG={vg:g}V", zorder=3)
            meas = load_measured(w, l)
            if meas is not None:
                for i, vg in enumerate(VG_SWEEP):
                    s = meas[meas.VG == int(vg)].sort_values("VD")
                    if len(s):
                        ax.scatter(s.VD, s.ID * 1e6, s=10, color=cmap[i], alpha=0.55,
                                   edgecolors="none", zorder=2)
                title_extra = f"n={row['n_meas_points']} rmse={row['rmse_log10ID']:.2f}dec"
            else:
                title_extra = "NO MEASURED DATA (extrapolated)"
            ax.set_title(f"W={w}um, L={l}um\n{title_extra}", fontsize=10)
            ax.set_xlabel("VD (V)")
            ax.set_ylabel(r"$I_D$ ($\mu$A)")
            style_axes(ax)
            if w == ws_to_plot[0]:
                ax.legend(fontsize=7, frameon=False, loc="upper left")

        fig.suptitle(f"L={l}um -- DC sweep (VG 0-5V/1V, VD 0-5V/0.2V, VS=0)  "
                     f"lines=model{id_tag}, dots=measured", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(out_dir, f"dc_sweep_L{l}.png"))
        plt.close(fig)
        print(f"Saved dc_sweep_L{l}.png ({n} widths)")

    return all_rows


def plot_all_19_grid(id_model, out_dir, id_tag=""):
    """Single figure, all 19 (W, L) combinations the device set actually
    covers, laid out L-by-row / W-by-column: raw measured data (dots) and
    the simulated Verilog-A/ANN model (lines) overlaid on each panel."""
    geoms_by_l = {l: sorted({w for w in ALL_W if load_measured(w, l) is not None}) for l in ALL_L}
    n_cols = max(len(ws) for ws in geoms_by_l.values())
    n_rows = len(ALL_L)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows), dpi=130)
    fig.patch.set_facecolor(BG)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(VG_SWEEP)))

    for r, l in enumerate(ALL_L):
        ws = geoms_by_l[l]
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(ws):
                ax.axis("off")
                continue
            w = ws[c]
            row, id_grid = analyze_geometry(id_model, w, l)
            for i, vg in enumerate(VG_SWEEP):
                ax.plot(VD_SWEEP, id_grid[i] * 1e6, color=cmap[i], lw=1.8, zorder=3)
            meas = load_measured(w, l)
            for i, vg in enumerate(VG_SWEEP):
                s = meas[meas.VG == int(vg)].sort_values("VD")
                if len(s):
                    ax.scatter(s.VD, s.ID * 1e6, s=8, color=cmap[i], alpha=0.5,
                               edgecolors="none", zorder=2)
            ax.set_title(f"W={w}, L={l}  rmse={row['rmse_log10ID']:.2f}dec", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("VD (V)", fontsize=8)
            if c == 0:
                ax.set_ylabel(r"$I_D$ ($\mu$A)", fontsize=8)
            ax.tick_params(labelsize=7)
            style_axes(ax)

    handles = [plt.Line2D([0], [0], color=cmap[i], lw=2, label=f"VG={vg:g}V")
               for i, vg in enumerate(VG_SWEEP)]
    fig.legend(handles=handles, loc="upper center", ncol=len(VG_SWEEP), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle(f"All 19 measured (W, L) geometries -- DC sweep (VG 0-5V/1V, VD 0-5V/0.2V, VS=0)\n"
                 f"lines=model{id_tag}, dots=measured", fontsize=13, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "dc_sweep_all19.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dc_sweep_all19.png ({n_rows}x{n_cols} grid, 19 panels)")


def summarize(rows, out_dir):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "dc_sweep_summary.csv"), index=False)

    print("\n=== per-(W,L) summary ===")
    print(df.to_string(index=False))

    print("\n=== aggregated by L ===")
    by_l = df.groupby("L").agg(
        mean_rmse_log10ID=("rmse_log10ID", "mean"),
        mean_gds_neg_fraction=("gds_negative_fraction", "mean"),
        worst_gds_S=("worst_gds_S", "min"),
        n_geometries_with_data=("has_data", "sum"),
        n_geometries_total=("W", "count"),
    )
    print(by_l.to_string())
    by_l.to_csv(os.path.join(out_dir, "dc_sweep_summary_by_L.csv"))
    return df, by_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "trained_ANN", "weights.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "trained_ANN", "dc_analysis"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    id_model, w = load_model(args.weights)
    rows = plot_by_L(id_model, args.out_dir, id_tag=args.tag)
    plot_all_19_grid(id_model, args.out_dir, id_tag=args.tag)
    df, by_l = summarize(rows, args.out_dir)


if __name__ == "__main__":
    main()
