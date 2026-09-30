"""Step 5 -- simulate the generated Verilog-A on every measured bias point of
all 19 devices and compare with the measurements.

Simulation: the .va is compiled by OpenVAF (through its Python front end
`verilogae`) and its analog block is evaluated at each (VGS, VDS) of the
measured sweeps, for the instance's (W, L). In the measurement every
terminal is driven by a source, so this is exactly the DC operating point a
circuit simulator (ngspice/Xyce via OSDI, Spectre) would compute.

Metrics (on-state, |ID_meas| > 1e-10 A; the off-state is the +-5 pA
instrument floor and is reported separately):
  mae_dec    mean |log10(ID_va) - log10(ID_meas)|  (0.1 dec ~ 26 %)
  rmse_dec, bias_dec (mean signed log error), med_rel (median |dI|/I)
reported per geometry, per sweep, for the 16 fit geometries and the 3
held-out ones, and binned by W, L, VGS, VDS and current level to expose
systematic errors. `own_mae_dec` is the floor from step 1: the same
equations with the device's own extracted parameters.

Usage: python step5_validate.py [variant] [va_path]
Outputs: step5_<variant>_*.csv, plots/step5_<variant>_*.png
"""
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALL_GEOMS, HOLDOUT, ON_LEVEL, OUT, PLOTS, VA_PATH, tag

INK, BLUE, ORANGE, GREY = "#1f1f1f", "#2a78d6", "#eb6834", "#8a8a85"


def load_va(path, m):
    import verilogae
    src = open(path).read().replace("real Ids_total;", "(*retrieve*) real Ids_total;")
    with tempfile.NamedTemporaryFile("w", suffix=".va", delete=False) as tmp:
        tmp.write(src)
    fn = verilogae.load(tmp.name).functions["Ids_total"]

    def ids(vgs, vds, W, L):
        vgs, vds = np.asarray(vgs, float), np.asarray(vds, float)
        return fn.eval(temperature=300.0, voltages={"br_gs": vgs, "br_ds": vds},
                       W=float(W), L=float(L), m=m)
    return ids


def measured():
    frames = []
    for w, l in ALL_GEOMS:
        for sweep in ("linear", "saturation", "output"):
            d = pd.read_csv(os.path.join(OUT, "selected_curves", f"{tag(w, l)}_{sweep}.csv"))
            frames.append(pd.DataFrame(dict(W=w, L=l, sweep=sweep, VG=d.VG, VD=d.VD, ID=d.ID)))
    df = pd.concat(frames, ignore_index=True)
    df["split"] = ["holdout" if (w, l) in HOLDOUT else "train" for w, l in zip(df.W, df.L)]
    return df[df.VD > 0].reset_index(drop=True)


def simulate(df, ids):
    out = np.empty(len(df))
    for (w, l), g in df.groupby(["W", "L"]):
        out[g.index] = ids(g.VG.to_numpy(), g.VD.to_numpy(), w, l)
    df = df.copy()
    df["ID_va"] = out
    df["on"] = np.abs(df.ID) > ON_LEVEL
    df["err_dec"] = np.log10(np.maximum(np.abs(df.ID_va), 1e-16)) - np.log10(
        np.maximum(np.abs(df.ID), 1e-16))
    df["rel"] = np.abs(df.ID_va - df.ID) / np.abs(df.ID)
    return df


def stats(g):
    e = g.err_dec
    return pd.Series(dict(n=len(g), mae_dec=np.mean(np.abs(e)), rmse_dec=np.sqrt(np.mean(e ** 2)),
                          bias_dec=np.mean(e), med_rel=np.median(g.rel),
                          p90_rel=np.quantile(g.rel, 0.9)))


def tables(df):
    on = df[df.on]
    summary = pd.concat([
        on.groupby(["split", "sweep"]).apply(stats, include_groups=False),
        on.groupby(["split"]).apply(stats, include_groups=False).assign(sweep="all")
          .set_index("sweep", append=True)]).sort_index()
    offs = df[~df.on].groupby("split").apply(stats, include_groups=False)
    per_geom = on.groupby(["W", "L", "split", "sweep"]).apply(stats, include_groups=False) \
        .reset_index()
    own = pd.read_csv(os.path.join(OUT, "step1_parameter_table.csv"))[
        ["W", "L", "own_mae_dec_lin", "own_mae_dec_sat", "own_mae_dec_out"]]
    per_geom = per_geom.merge(own, on=["W", "L"])
    per_geom["own_mae_dec"] = [r[f"own_mae_dec_{r.sweep[:3]}"] for _, r in per_geom.iterrows()]
    per_geom = per_geom.drop(columns=["own_mae_dec_lin", "own_mae_dec_sat", "own_mae_dec_out"])
    tr = on[on.split == "train"].copy()
    tr["VGS_bin"] = pd.cut(tr.VG, [-5, 0, 1, 2, 3, 4, 5.01])
    tr["VDS_bin"] = pd.cut(tr.VD, [0, 0.11, 0.5, 1, 2, 3, 4, 5.01])
    tr["I_bin"] = pd.cut(np.log10(np.abs(tr.ID)), [-10, -9, -8, -7, -6, -5, -3])
    systematic = pd.concat({k: tr.groupby(k, observed=True).apply(stats, include_groups=False)
                            for k in ("W", "L", "VGS_bin", "VDS_bin", "I_bin")})
    return summary, offs, per_geom, systematic


def plot_transfer(df, variant, logy=True):
    fig, axes = plt.subplots(4, 5, figsize=(21, 15), sharex=True)
    for ax, (w, l) in zip(axes.ravel(), ALL_GEOMS):
        for sweep, c in (("linear", BLUE), ("saturation", ORANGE)):
            s = df[(df.W == w) & (df.L == l) & (df.sweep == sweep)].sort_values("VG")
            y, yv = np.abs(s.ID), np.abs(s.ID_va)
            if not logy:
                sc = 1e6
                y, yv = y * sc, yv * sc
            ax.plot(s.VG, y, "o", ms=2.6, color=c, alpha=0.45)
            ax.plot(s.VG, yv, "-", lw=1.8, color=c, label=f"{sweep} (VD={s.VD.iloc[0]:g} V)")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-13, 1e-3)
        hold = (w, l) in HOLDOUT
        g = df[(df.W == w) & (df.L == l) & df.on & (df.sweep != "output")]
        ax.set_title(f"W={w} L={l} um{'  [UNSEEN]' if hold else ''}   "
                     f"MAE {np.mean(np.abs(g.err_dec)):.3f} dec", fontsize=9,
                     color=ORANGE if hold else INK)
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("VGS (V)")
    fig.suptitle(f"Step 5 [{variant}]: transfer curves, measured (dots) vs Verilog-A (lines)"
                 + ("" if logy else "  -- linear scale, ID in uA"), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"step5_{variant}_transfer_{'log' if logy else 'lin'}.png"),
                dpi=75)
    plt.close(fig)


def plot_output(df, variant):
    fig, axes = plt.subplots(4, 5, figsize=(21, 15))
    cmap = plt.get_cmap("viridis")
    for ax, (w, l) in zip(axes.ravel(), ALL_GEOMS):
        s = df[(df.W == w) & (df.L == l) & (df.sweep == "output") & (df.VG >= 1)]
        for vg, g in s.groupby("VG"):
            g = g.sort_values("VD")
            c = cmap((vg - 1) / 4.2)
            ax.plot(g.VD, g.ID * 1e6, "o", ms=2.2, color=c, alpha=0.45)
            ax.plot(g.VD, g.ID_va * 1e6, "-", lw=1.7, color=c)
            ax.annotate(f"{vg:g}", (5.05, g.ID.iloc[-1] * 1e6), fontsize=6)
        hold = (w, l) in HOLDOUT
        g = df[(df.W == w) & (df.L == l) & df.on & (df.sweep == "output")]
        ax.set_title(f"W={w} L={l} um{'  [UNSEEN]' if hold else ''}   "
                     f"MAE {np.mean(np.abs(g.err_dec)):.3f} dec", fontsize=9,
                     color=ORANGE if hold else INK)
        ax.set_xlim(0, 5.5)
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"Step 5 [{variant}]: output curves VGS = 1..5 V, measured (dots) vs "
                 "Verilog-A (lines), ID in uA", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"step5_{variant}_output.png"), dpi=75)
    plt.close(fig)


def plot_systematic(df, per_geom, variant):
    on = df[df.on]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for sweep, c in (("linear", BLUE), ("saturation", ORANGE), ("output", GREY)):
        s = on[(on.sweep == sweep) & (on.split == "train")]
        axes[0, 0].plot(s.VG + np.random.uniform(-0.03, 0.03, len(s)), s.err_dec, ".", ms=2,
                        color=c, alpha=0.4, label=sweep)
        axes[0, 1].plot(s.VD, s.err_dec, ".", ms=2, color=c, alpha=0.4)
        axes[0, 2].semilogx(np.abs(s.ID), s.err_dec, ".", ms=2, color=c, alpha=0.4)
    axes[0, 0].set(xlabel="VGS (V)", ylabel="log10(ID_va/ID_meas)", title="error vs VGS (16 fit)")
    axes[0, 1].set(xlabel="VDS (V)", title="error vs VDS")
    axes[0, 2].set(xlabel="|ID| measured (A)", title="error vs current level")
    axes[0, 0].legend(markerscale=5)
    g = per_geom.groupby(["W", "L", "split"]).mae_dec.mean().reset_index()
    for split, mk in (("train", "o"), ("holdout", "s")):
        s = g[g.split == split]
        axes[1, 0].scatter(s.W, s.mae_dec, c=s.L, marker=mk, cmap="viridis", s=50, vmin=5, vmax=20)
        axes[1, 1].scatter(s.L, s.mae_dec, c=np.log2(s.W), marker=mk, cmap="plasma", s=50,
                           vmin=2, vmax=7.5)
    axes[1, 0].set(xscale="log", xlabel="W (um), colour = L", ylabel="MAE (dec)",
                   title="per-geometry MAE vs W (squares = unseen)")
    axes[1, 1].set(xlabel="L (um), colour = log2 W", title="per-geometry MAE vs L")
    pg = per_geom.groupby(["W", "L", "split"])[["mae_dec", "own_mae_dec"]].mean().reset_index()
    lab = [f"{w}/{l}" for w, l in zip(pg.W, pg.L)]
    x = np.arange(len(pg))
    axes[1, 2].bar(x - 0.2, pg.mae_dec, 0.4, color=[ORANGE if s == "holdout" else BLUE
                                                    for s in pg.split], label="scaled model")
    axes[1, 2].bar(x + 0.2, pg.own_mae_dec, 0.4, color=GREY, label="own-parameter floor")
    axes[1, 2].set_xticks(x, lab, rotation=70, fontsize=7)
    axes[1, 2].set(ylabel="MAE (dec)", title="scaled model vs own-parameter floor")
    axes[1, 2].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    for ax in axes[0]:
        ax.axhline(0, color=INK, lw=0.8)
    fig.suptitle(f"Step 5 [{variant}]: systematic error analysis (on-state points)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"step5_{variant}_errors.png"), dpi=85)
    plt.close(fig)


def run(variant, va_path, plots=True, quiet=False):
    with open(os.path.join(OUT, f"step3_coefficients_{variant}.json")) as fh:
        m = json.load(fh)["m"]
    df = simulate(measured(), load_va(va_path, m))
    summary, offs, per_geom, systematic = tables(df)
    pref = os.path.join(OUT, f"step5_{variant}")
    summary.to_csv(pref + "_summary.csv", float_format="%.4g")
    per_geom.to_csv(pref + "_per_geometry.csv", index=False, float_format="%.4g")
    systematic.to_csv(pref + "_systematic.csv", float_format="%.4g")
    if plots:
        df.to_csv(pref + "_points.csv.gz", index=False, float_format="%.5g")
        plot_transfer(df, variant, True)
        plot_transfer(df, variant, False)
        plot_output(df, variant)
        plot_systematic(df, per_geom, variant)
    if not quiet:
        pd.set_option("display.width", 200)
        f = lambda v: f"{v:.3f}"
        print(summary.to_string(float_format=f))
        print("\noff-state (instrument floor):\n", offs.to_string(float_format=f))
        print(per_geom.pivot_table(index=["W", "L", "split"], columns="sweep",
                                   values=["mae_dec", "own_mae_dec"]).to_string(float_format=f))
        print(systematic.to_string(float_format=f))
    return df, summary, per_geom


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "thesis"
    run(variant, sys.argv[2] if len(sys.argv) > 2 else VA_PATH)
