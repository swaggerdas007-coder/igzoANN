"""Step 0 -- pick ONE physical device per geometry whose transfer (linear
VD=0.1 V and saturation VD=5 V) AND output (Id-Vd family) curves are all
proper transistor curves. Transfer and output come from the same device so
the parameters extracted from the transfer curves (step 1) describe the
device whose output curve the model is checked against (step 5).

Candidates: every device replicate in data_cleaned/ (already past the basic
QC in scripts/clean_transistor_curves.py). Each is scored on:

  transfer  von_*       turn-on (thesis derivative-onset rule, common.onset)
            drop_*      largest fractional dip once on (noise/non-monotonic)
            leak        median |ID| well below turn-on (off-state leakage)
            top_ratio   ID(5 V)/ID(4 V) in saturation (compliance clipping)
  output    out_drop    largest fractional dip of any on Id-Vd trace
            out_cross   VG-ordering violations at VD = 1..5 V
            out_rough   curvature roughness of the on traces
            c_lin/c_sat output-vs-transfer agreement at (5 V, 0.1 V) and
                        (5 V, 5 V): same device, same bias -> same current

Hard rejects are listed in REJECT. Among survivors the thesis rule applies
(Sec. 3.3.2: keep the data "most closely aligned with ideal IGZO TFT
behavior ... minimal dispersion"): the smallest distance of the turn-on
from the population's typical turn-on, plus the output/transfer
inconsistency and output roughness. The late-turn-on bot2 die site is a
wafer-position effect, not a geometry effect, and would otherwise be
confused with W/L scaling in step 3.

Outputs: outputs_camcas_thesis/step0_candidates.csv, step0_selection.csv,
selected_curves/*.csv, plots/step0_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ALL_GEOMS, HOLDOUT, ON_LEVEL, OUT, PLOTS, load_sweep,
                    max_rel_drop, onset, tag)

REJECT = {
    "noisy lin on-state": lambda r: r.drop_lin > 0.05,
    "noisy sat on-state": lambda r: r.drop_sat > 0.05,
    "off-state leakage": lambda r: r.leak > 2e-11,
    "compliance clipping": lambda r: r.top_ratio < 1.1,
    "non-monotonic output": lambda r: r.out_drop > 0.05,
    "crossing output": lambda r: r.out_cross > 0,
    "output/transfer drift":
        lambda r: max(abs(np.log(r.c_lin)), abs(np.log(r.c_sat))) > np.log(1.5),
}


def transfer_metrics(g):
    von, _, k = onset(g.VG, g.ID)
    a = np.abs(g.ID.to_numpy())
    on = a[k:]
    on = on[on > ON_LEVEL]
    off = a[g.VG.to_numpy() < von - 0.3]
    return von, (max_rel_drop(on) if len(on) > 2 else 1.0), \
        (float(np.median(off)) if len(off) >= 3 else np.nan), a


def output_metrics(o):
    drops, rough = [], []
    for vg, t in o[o.VG >= 1].groupby("VG"):
        i = t.sort_values("VD").ID.to_numpy()
        if i[-1] < 1e-9:
            continue
        drops.append(max_rel_drop(i))
        rough.append(np.sum(np.abs(np.diff(i, 2))) / (i.max() - i.min()))
    cross = 0
    for vd in (1.0, 2.0, 3.0, 4.0, 5.0):
        s = o[np.isclose(o.VD, vd) & (o.VG >= 0)].sort_values("VG").ID.to_numpy()
        cross += int(np.sum(np.diff(s) < 0))
    return max(drops), float(np.median(rough)), cross


def candidates():
    rows = []
    for w, l in ALL_GEOMS:
        lin, sat, out = (load_sweep(w, l, s) for s in ("linear", "saturation", "output"))
        for dev in sorted(lin.device.unique()):
            gl, gs, o = lin[lin.device == dev], sat[sat.device == dev], out[out.device == dev]
            von_l, drop_l, leak_l, al = transfer_metrics(gl)
            von_s, drop_s, leak_s, as_ = transfer_metrics(gs)
            od, orough, ocross = output_metrics(o)
            i55 = o[(o.VG == 5) & np.isclose(o.VD, 5.0)].ID.iloc[0]
            i501 = o[(o.VG == 5) & np.isclose(o.VD, 0.1)].ID.iloc[0]
            rows.append(dict(
                W=w, L=l, device=dev, von_lin=von_l, von_sat=von_s,
                drop_lin=drop_l, drop_sat=drop_s,
                leak=np.nanmax([leak_l, leak_s]),
                top_ratio=as_[-1] / as_[-11],
                out_drop=od, out_rough=orough, out_cross=ocross,
                c_lin=abs(i501) / al[-1], c_sat=abs(i55) / as_[-1]))
    df = pd.DataFrame(rows)
    df["reject"] = [", ".join(k for k, f in REJECT.items() if f(r)) for r in df.itertuples()]
    ok = df.reject == ""
    typ = df.loc[ok, ["von_lin", "von_sat"]].median()
    df["score"] = (np.abs(df.von_lin - typ.von_lin) + np.abs(df.von_sat - typ.von_sat)
                   + np.abs(np.log(df.c_lin)) + np.abs(np.log(df.c_sat)) + 5 * df.out_rough)
    df.attrs["typ"] = typ
    return df


def plot_selection(sel, cand):
    fig, axes = plt.subplots(4, 5, figsize=(21, 15))
    for ax, (w, l) in zip(axes.ravel(), ALL_GEOMS):
        dev = sel.loc[(sel.W == w) & (sel.L == l), "device"].iloc[0]
        for sweep, ls in (("linear", "-"), ("saturation", "--")):
            d = load_sweep(w, l, sweep)
            for dv, g in d.groupby("device"):
                chosen = dv == dev
                ax.semilogy(g.VG, np.abs(g.ID), ls, lw=2.2 if chosen else 0.8,
                            color="#1f1f1f" if chosen else "#bbbbbb", zorder=3 if chosen else 1)
        c = cand[(cand.W == w) & (cand.L == l)]
        txt = "\n".join(f"{r.device}: {'OK' if not r.reject else r.reject}" for r in c.itertuples())
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=6.5, va="top", clip_on=True)
        ax.set_title(f"W={w} L={l}  -> {dev}" + ("  [held out]" if (w, l) in HOLDOUT else ""),
                     fontsize=9)
        ax.set_ylim(1e-14, 1e-3)
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Step 0: transfer curves of every candidate (grey) and the selected device "
                 "(black; solid VD=0.1 V, dashed VD=5 V)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step0_transfer_selection.png"), dpi=80)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(21, 15))
    for ax, (w, l) in zip(axes.ravel(), ALL_GEOMS):
        dev = sel.loc[(sel.W == w) & (sel.L == l), "device"].iloc[0]
        o = load_sweep(w, l, "output", dev)
        for vg, t in o[o.VG >= 0].groupby("VG"):
            ax.plot(t.VD, t.ID * 1e6, lw=1.4)
            ax.annotate(f"{vg:g}", (5.02, t.ID.iloc[-1] * 1e6), fontsize=6)
        ax.set_title(f"W={w} L={l}  {dev}", fontsize=9)
        ax.set_xlim(0, 5.4)
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Step 0: selected output families (VG = 0..5 V, ID in uA)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step0_output_selection.png"), dpi=80)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    cand = candidates()
    cand.to_csv(os.path.join(OUT, "step0_candidates.csv"), index=False, float_format="%.4g")
    ok = cand[cand.reject == ""]
    sel = ok.loc[ok.groupby(["W", "L"]).score.idxmin()].sort_values(["W", "L"])
    missing = set(ALL_GEOMS) - set(zip(sel.W, sel.L))
    assert not missing, f"no acceptable device for {missing}"
    sel.to_csv(os.path.join(OUT, "step0_selection.csv"), index=False, float_format="%.4g")

    cdir = os.path.join(OUT, "selected_curves")
    os.makedirs(cdir, exist_ok=True)
    for r in sel.itertuples():
        for sweep in ("linear", "saturation", "output"):
            load_sweep(r.W, r.L, sweep, r.device).to_csv(
                os.path.join(cdir, f"{tag(r.W, r.L)}_{sweep}.csv"), index=False)
    plot_selection(sel, cand)

    pd.set_option("display.width", 220)
    print("typical turn-on (median of acceptable candidates):", dict(cand.attrs["typ"]))
    print(cand.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nSELECTED:")
    print(sel[["W", "L", "device", "von_lin", "von_sat", "c_lin", "c_sat", "out_rough",
               "score"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
