"""DC analysis of verilogA/tft_ann_static.va, run in Python via
scripts/verilog_a_model.py (a direct numpy transcription of the .va
analog block -- see that file's docstring). Source is tied to 0V (VS=0)
throughout, matching the measured-data convention in cleaned_output_meas/.

Two sweep types, for all 19 (W, L) geometries with W >= L (the same 19
combinations covered by cleaned_output_meas/):

  * Vd sweep (output characteristics): VD swept 0->5V (0.05V step) with VG
    stepped 0,1,2,3,4,5V as a family of curves.
  * Vg sweep (transfer characteristics): VG swept -5->5V (0.05V step) with
    VD stepped 0.1,1,2,3,4,5V as a family of curves.

Outputs (under --out-dir, default trained_ANN/dc_analysis_verilogA/):
  - vd_sweep_data.csv, vg_sweep_data.csv   (long-format raw sweep data)
  - geometry_summary.csv                    (per-(W,L) Ion/Ioff/gds diagnostics)
  - dc_vd_sweep_all19.png                   (output-family grid, vs measured)
  - dc_vg_sweep_all19.png                   (transfer-family grid, vs measured)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from verilog_a_model import TftAnnModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
MEAS_DIR = os.path.join(ROOT, "cleaned_output_meas")
VA_PATH = os.path.join(ROOT, "verilogA", "tft_ann_static.va")

ALL_W = [5, 10, 20, 40, 80, 160]
ALL_L = [5, 10, 15, 20]
GEOMETRIES = [(w, l) for w in ALL_W for l in ALL_L if w >= l]   # 19 combos
assert len(GEOMETRIES) == 19

VG_FAMILY = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]          # VD-sweep parameter steps
VD_SWEEP = np.round(np.arange(0.0, 5.0001, 0.05), 4)

VD_FAMILY = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]          # VG-sweep parameter steps
VG_SWEEP = np.round(np.arange(-5.0, 5.0001, 0.05), 4)

BG = "#f7f7f5"
MEAS_COLOR = "#4c72b0"


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


def run_vd_sweep(model, w, l):
    """Output characteristics: ID(VD) family stepped by VG. Returns a
    long-format DataFrame and the gds-negative diagnostic."""
    rows = []
    gds_all = []
    for vg in VG_FAMILY:
        id_, gm, gds = model.evaluate(vg, VD_SWEEP, w, l)
        gds_all.append(gds)
        for vd, idv, gdsv in zip(VD_SWEEP, id_, gds):
            rows.append(dict(W=w, L=l, VG=vg, VD=vd, ID=idv, gds=gdsv))
    df = pd.DataFrame(rows)
    gds_all = np.concatenate(gds_all)
    return df, float((gds_all < 0).mean()), float(gds_all.min())


def run_vg_sweep(model, w, l):
    """Transfer characteristics: ID(VG) family stepped by VD."""
    rows = []
    for vd in VD_FAMILY:
        id_, gm, gds = model.evaluate(VG_SWEEP, vd, w, l)
        for vg, idv, gmv in zip(VG_SWEEP, id_, gm):
            rows.append(dict(W=w, L=l, VD=vd, VG=vg, ID=idv, gm=gmv))
    return pd.DataFrame(rows)


def geometry_summary(vd_df, vg_df, w, l, gds_neg_frac, worst_gds, meas):
    on_slice = vg_df[(vg_df.VD == 5.0)]
    ion = on_slice.ID.max()
    ioff = on_slice.loc[on_slice.VG.idxmin(), "ID"]
    row = dict(
        W=w, L=l,
        Ion_A=ion, Ioff_A=ioff, on_off_ratio=ion / ioff if ioff > 0 else np.nan,
        max_gm_S=vg_df.gm.max(),
        gds_negative_fraction=gds_neg_frac, worst_gds_S=worst_gds,
        has_measured_data=meas is not None,
    )
    if meas is not None:
        m = meas[meas.VG.isin([int(v) for v in VG_FAMILY])].copy()
        model_df = vd_df.set_index(["VG", "VD"])
        pred = []
        for _, r in m.iterrows():
            vd_nearest = VD_SWEEP[np.argmin(np.abs(VD_SWEEP - r.VD))]
            pred.append(model_df.loc[(float(r.VG), vd_nearest), "ID"])
        pred = np.array(pred)
        log_true = np.log10(np.abs(m.ID.to_numpy()).clip(min=1e-15))
        log_pred = np.log10(np.abs(pred).clip(min=1e-15))
        row["n_meas_points"] = int(len(m))
        row["rmse_log10ID_vs_meas"] = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    else:
        row["n_meas_points"] = 0
        row["rmse_log10ID_vs_meas"] = None
    return row


def plot_vd_grid(model, out_dir, tag):
    n_cols = max(len([w for w in ALL_W if (w, l) in GEOMETRIES]) for l in ALL_L)
    n_rows = len(ALL_L)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows), dpi=130)
    fig.patch.set_facecolor(BG)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(VG_FAMILY)))

    for r, l in enumerate(ALL_L):
        ws = sorted(w for w in ALL_W if (w, l) in GEOMETRIES)
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(ws):
                ax.axis("off")
                continue
            w = ws[c]
            for i, vg in enumerate(VG_FAMILY):
                id_, _, _ = model.evaluate(vg, VD_SWEEP, w, l)
                ax.plot(VD_SWEEP, id_ * 1e6, color=cmap[i], lw=1.8, zorder=3)
            meas = load_measured(w, l)
            if meas is not None:
                for i, vg in enumerate(VG_FAMILY):
                    s = meas[meas.VG == int(vg)].sort_values("VD")
                    if len(s):
                        ax.scatter(s.VD, s.ID * 1e6, s=8, color=cmap[i], alpha=0.5,
                                   edgecolors="none", zorder=2)
            ax.set_title(f"W={w}, L={l}", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("VD (V)", fontsize=8)
            if c == 0:
                ax.set_ylabel(r"$I_D$ ($\mu$A)", fontsize=8)
            ax.tick_params(labelsize=7)
            style_axes(ax)

    handles = [plt.Line2D([0], [0], color=cmap[i], lw=2, label=f"VG={vg:g}V")
               for i, vg in enumerate(VG_FAMILY)]
    fig.legend(handles=handles, loc="upper center", ncol=len(VG_FAMILY), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle(f"VD sweep (output family) -- all 19 W>=L geometries{tag}\n"
                 "lines = verilogA/tft_ann_static.va (run in Python), dots = measured",
                 fontsize=13, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "dc_vd_sweep_all19.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_vg_grid(model, out_dir, tag):
    n_cols = max(len([w for w in ALL_W if (w, l) in GEOMETRIES]) for l in ALL_L)
    n_rows = len(ALL_L)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows), dpi=130)
    fig.patch.set_facecolor(BG)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(VD_FAMILY)))

    for r, l in enumerate(ALL_L):
        ws = sorted(w for w in ALL_W if (w, l) in GEOMETRIES)
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(ws):
                ax.axis("off")
                continue
            w = ws[c]
            for i, vd in enumerate(VD_FAMILY):
                id_, _, _ = model.evaluate(VG_SWEEP, vd, w, l)
                ax.semilogy(VG_SWEEP, np.abs(id_), color=cmap[i], lw=1.8, zorder=3)
            meas = load_measured(w, l)
            if meas is not None:
                vd_nearest = 0.1
                s = meas[np.isclose(meas.VD, vd_nearest, atol=0.05)].sort_values("VG")
                if len(s):
                    ax.scatter(s.VG, np.abs(s.ID), s=14, color=cmap[0], alpha=0.6,
                               marker="x", zorder=2, label="meas VD=0.1V")
            ax.set_title(f"W={w}, L={l}", fontsize=9)
            ax.set_ylim(1e-13, 1e-3)
            if r == n_rows - 1:
                ax.set_xlabel("VG (V)", fontsize=8)
            if c == 0:
                ax.set_ylabel(r"$|I_D|$ (A)", fontsize=8)
            ax.tick_params(labelsize=7)
            style_axes(ax)

    handles = [plt.Line2D([0], [0], color=cmap[i], lw=2, label=f"VD={vd:g}V")
               for i, vd in enumerate(VD_FAMILY)]
    handles.append(plt.Line2D([0], [0], color=cmap[0], marker="x", lw=0, label="meas VD=0.1V"))
    fig.legend(handles=handles, loc="upper center", ncol=len(VD_FAMILY) + 1, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle(f"VG sweep (transfer family) -- all 19 W>=L geometries{tag}\n"
                 "lines = verilogA/tft_ann_static.va (run in Python), x = measured @ VD=0.1V",
                 fontsize=13, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "dc_vg_sweep_all19.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--va", default=VA_PATH)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "trained_ANN", "dc_analysis_verilogA"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = TftAnnModel(args.va)

    vd_dfs, vg_dfs, summary_rows = [], [], []
    for w, l in GEOMETRIES:
        vd_df, gds_neg_frac, worst_gds = run_vd_sweep(model, w, l)
        vg_df = run_vg_sweep(model, w, l)
        vd_dfs.append(vd_df)
        vg_dfs.append(vg_df)
        meas = load_measured(w, l)
        summary_rows.append(geometry_summary(vd_df, vg_df, w, l, gds_neg_frac, worst_gds, meas))
        print(f"W={w:>3}um L={l:>2}um: Ion={summary_rows[-1]['Ion_A']:.3e}A "
              f"Ioff={summary_rows[-1]['Ioff_A']:.3e}A "
              f"on/off={summary_rows[-1]['on_off_ratio']:.2e} "
              f"gds<0 frac={gds_neg_frac:.3f}")

    vd_all = pd.concat(vd_dfs, ignore_index=True)
    vg_all = pd.concat(vg_dfs, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    vd_all.to_csv(os.path.join(args.out_dir, "vd_sweep_data.csv"), index=False)
    vg_all.to_csv(os.path.join(args.out_dir, "vg_sweep_data.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "geometry_summary.csv"), index=False)
    print(f"\nWrote vd_sweep_data.csv ({len(vd_all)} rows), "
          f"vg_sweep_data.csv ({len(vg_all)} rows), geometry_summary.csv ({len(summary)} rows)")

    plot_vd_grid(model, args.out_dir, args.tag)
    plot_vg_grid(model, args.out_dir, args.tag)

    print("\n=== geometry summary ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
