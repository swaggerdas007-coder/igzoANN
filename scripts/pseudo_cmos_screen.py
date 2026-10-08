"""DC screen of unipolar a-IGZO NOT gates: topology comparison + full sizing sweep.

Stage 1 -- all six topologies at one reference sizing, to establish which style
           is worth optimising at all.
Stage 2 -- every (W1,W2,W3,W4) on the measured width grid for the two
           pseudo-CMOS variants: 6^4 = 1296 sizings each, exact DC.

Bootstrapping is a purely *dynamic* effect (a capacitor does nothing in DC), so
pC and pCb share a VTC and so do pCz and pCzb. This script therefore screens the
DC behaviour; scripts/pseudo_cmos_best.py takes the survivors into transient and
is where the bootstrap cap is actually chosen.

    python scripts/pseudo_cmos_screen.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import (KINDS, W_GRID, L_LOGIC, Sizing, screen_metrics,  # noqa: E402
                             screen_vtc, vtc)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)
VDD_REF = 3.0
NPTS = 201
COLS = ["kind", "w1", "w2", "w3", "w4", "gain", "gain_vm", "voh", "vol",
        "swing", "vm", "snm", "nmh", "nml", "p_static", "i_lo", "i_hi",
        "tpd_proxy", "pdp_proxy", "valid"]


def verify():
    """The fast screen must reproduce the full nodal engine."""
    worst = 0.0
    for w1, w2, w3, w4, kind in [(5, 160, 160, 160, "pC"), (20, 80, 40, 160, "pC"),
                                 (160, 20, 160, 40, "pC"), (80, 160, 160, 160, "pCz"),
                                 (10, 40, 80, 20, "pC"), (40, 160, 20, 80, "pCz")]:
        S = Sizing(w1=w1e(w1), w2=w2e(w2), w3=w3e(w3), w4=w4e(w4))
        ref = vtc(S, kind, VDD_REF, npts=61)
        m = np.isfinite(ref["vout"])
        vx, vo, _ = screen_vtc(w1e(w1), w2e(w2), w3e(w3), w4e(w4), kind,
                               VDD_REF, ref["vin"])
        worst = max(worst, float(np.max(np.abs(vo[m] - ref["vout"][m]))),
                    float(np.max(np.abs(vx[m] - ref["vx"][m]))))
    print(f"fast screen vs full nodal engine: max |dV| = {worst:.2e} V")
    assert worst < 1e-4, "fast DC screen disagrees with the nodal engine"


def w1e(x):
    return x * 1e-6


w2e = w3e = w4e = w1e


def _row(a):
    w1, w2, w3, w4, kind = a
    r = screen_metrics(w1, w2, w3, w4, kind, VDD_REF, npts=NPTS)
    r["w1"], r["w2"], r["w3"], r["w4"] = w1 * 1e6, w2 * 1e6, w3 * 1e6, w4 * 1e6
    return {k: r[k] for k in COLS}


def topology_table():
    print("\n--- topology comparison at W1/W2/W3/W4 = 5/160/160/160 um, "
          f"L = {L_LOGIC*1e6:.0f} um, VDD = {VDD_REF} V ---")
    s = Sizing(w1=5e-6, w2=160e-6, w3=160e-6, w4=160e-6)
    rows, curves = [], {}
    for k in KINDS:
        r = vtc(s, k, VDD_REF, npts=241)
        curves[k] = (r["vin"], r["vout"], r["vx"])
        rows.append(dict(kind=k, ndev=2 if k in ("pE", "pD") else 4,
                         gain=r["gain"], voh=r["voh"], vol=r["vol"],
                         swing=r["swing"], swing_pct=100 * r["swing"] / VDD_REF,
                         vm=r["vm"], snm=r["snm"], p_static_uW=r["p_static"] * 1e6,
                         tpd_proxy_us=r["tpd_proxy"] * 1e6, valid=r["valid"]))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    df.to_csv(os.path.join(OUT, "topology_comparison.csv"), index=False)
    return df, curves


def plot_topologies(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, (vi, vo, vx) in curves.items():
        ax[0].plot(vi, vo, label=k, lw=1.8)
        if np.isfinite(vx).any():
            ax[1].plot(vi, vx, label=k, lw=1.8)
    ax[0].plot([0, VDD_REF], [0, VDD_REF], "k:", lw=.8, label="Vout = Vin")
    ax[0].set(xlabel="$V_{IN}$ [V]", ylabel="$V_{OUT}$ [V]",
              title=f"Unipolar a-IGZO NOT gates, VDD = {VDD_REF} V\n"
                    "5/160/160/160 um, L = 5 um")
    ax[1].set(xlabel="$V_{IN}$ [V]", ylabel="internal node $V_X$ [V]",
              title="first-stage node X (4T styles)")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "topology_comparison.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def full_screen():
    jobs = [(a, b, c, d, k) for k in ("pC", "pCz")
            for a, b, c, d in itertools.product(W_GRID, repeat=4)]
    print(f"\n--- full sizing screen: {len(jobs)} DC VTCs ({NPTS} pts each) ---")
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_row, jobs, chunksize=16)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sizing_screen.csv"), index=False)
    print(f"wrote {OUT}/sizing_screen.csv  ({len(df)} rows, "
          f"{int(df.valid.sum())} cascadable)")
    return df


def report(df):
    for k in ("pC", "pCz"):
        d = df[df.kind == k]
        print(f"\n{k}: {int(d.valid.sum())}/{len(d)} sizings cascadable")
        if not d.valid.any():
            continue
        v = d[d.valid]
        print(f"  max |dVout/dVin|   {v.gain.min():.2f} .. {v.gain.max():.2f}")
        print(f"  loop gain at VM    {v.gain_vm.min():.2f} .. {v.gain_vm.max():.2f}"
              f"   <- the one that decides whether a cascade regenerates")
        latch = d[(d.gain > 50) & (d.gain_vm < 1)]
        if len(latch):
            print(f"  {len(latch)} sizings reach |dVout/dVin| > 50 yet have loop gain "
                  f"< 1 at their own trip point -- they latch, and ranking on "
                  f"max slope would have picked them")
        print(f"  swing     {v.swing.min():.2f} .. {v.swing.max():.2f} V "
              f"({v.swing.max()/VDD_REF*100:.0f}% of VDD)")
        print(f"  SNM       {v.snm.min():.3f} .. {v.snm.max():.3f} V")
        print(f"  P_static  {v.p_static.min()*1e6:.3f} .. {v.p_static.max()*1e6:.1f} uW")
    v = df[df.valid].copy()
    print("\n--- best by max gain ---")
    print(v.nlargest(8, "gain")[["kind", "w1", "w2", "w3", "w4", "gain", "voh",
                                 "vol", "snm", "p_static", "tpd_proxy"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.4g}"))
    print("\n--- best by max SNM ---")
    print(v.nlargest(8, "snm")[["kind", "w1", "w2", "w3", "w4", "gain", "voh",
                                "vol", "snm", "p_static", "tpd_proxy"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.4g}"))
    print("\n--- fastest (min delay proxy) ---")
    print(v.nsmallest(8, "tpd_proxy")[["kind", "w1", "w2", "w3", "w4", "gain",
                                       "voh", "vol", "snm", "p_static", "tpd_proxy"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.4g}"))
    print("\n--- best power-delay product ---")
    print(v.nsmallest(8, "pdp_proxy")[["kind", "w1", "w2", "w3", "w4", "gain",
                                       "voh", "vol", "snm", "p_static", "tpd_proxy"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.4g}"))


def plot_screen(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = df[df.valid]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    sc = ax[0].scatter(v.p_static * 1e6, v.gain, c=np.log10(v.tpd_proxy * 1e6),
                       s=14, cmap="viridis")
    ax[0].set(xscale="log", xlabel="static power [uW]", ylabel="max |dVout/dVin|",
              title="gain vs static power")
    plt.colorbar(sc, ax=ax[0], label="log10 tpd proxy [us]")
    sc = ax[1].scatter(v.tpd_proxy * 1e6, v.snm, c=np.log10(v.p_static * 1e6),
                       s=14, cmap="plasma")
    ax[1].set(xscale="log", xlabel="tpd proxy [us]", ylabel="butterfly SNM [V]",
              title="noise margin vs speed")
    plt.colorbar(sc, ax=ax[1], label="log10 P [uW]")
    for k, c in (("pC", "C0"), ("pCz", "C3")):
        d = v[v.kind == k]
        ax[2].scatter(d.tpd_proxy * 1e6, d.p_static * 1e6, s=14, label=k, alpha=.6, c=c)
    ax[2].set(xscale="log", yscale="log", xlabel="tpd proxy [us]",
              ylabel="static power [uW]", title="the real tradeoff")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"a-IGZO pseudo-CMOS NOT gate: {len(v)} cascadable sizings "
                 f"of {len(df)} screened, VDD = {VDD_REF} V", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "sizing_screen.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    verify()
    _, curves = topology_table()
    plot_topologies(curves)
    df = full_screen()
    report(df)
    plot_screen(df)
