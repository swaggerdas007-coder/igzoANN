"""5-stage pseudo-CMOS bootstrapped ring oscillator: supply-voltage / frequency study.

Sweeps VDD over the whole range the compact model can be held responsible for
and reports, at each supply, the oscillation frequency, output swing, supply
power, energy per transition, how far the bootstrap pushes node X above VDD, and
whether any device left the model's trained box. From that it picks the usable
supply window and the best operating ranges.

Three error bars are measured rather than assumed, because a ring-oscillator
frequency is exactly the kind of number that is quoted to three digits and
deserves none of them:

  numerical   the frequency is re-run at four time-step limits and Richardson
              extrapolated, since backward Euler's period error is O(dt).
  charge      `ddt(C*V)` (what the .va actually tells a simulator) vs using the
              calibrated C as an incremental capacitance. The README puts the
              difference in effective dQ/dV at up to 303%.
  capacitance CGD/CGS were trained at four geometries, none shorter than
              L = 15 um, so L = 5 um logic is an area-scaling extrapolation.
              Re-run with every device cap x0.5 and x2.

    python scripts/ring_osc5.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import Sizing                       # noqa: E402
from src.ring import run_ring_auto                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)

NSTAGE = 5
VDD_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
            2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
KEEP = ["vdd", "freq", "period", "swing", "swing_pct", "vmin", "vmax", "x_max",
        "x_min", "boost", "power", "i_supply", "e_per_cycle", "e_per_transition",
        "tpd", "jitter", "n_periods", "mode_index", "oob_frac", "steps",
        "cap_mode", "cap_scale", "dvmax", "oscillates", "reason"]
FALLBACK = dict(w1=5.0, w2=40.0, w3=160.0, w4=80.0, cboot=2e-12)


def gate():
    p = os.path.join(OUT, "recommended_gate.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        g = dict(w1=float(r.w1), w2=float(r.w2), w3=float(r.w3),
                 w4=float(r.w4), cboot=float(r.cboot))
    else:
        print("recommended_gate.csv missing -- using the fallback sizing")
        g = dict(FALLBACK)
    print(f"gate: W1/W2/W3/W4 = {g['w1']:.0f}/{g['w2']:.0f}/{g['w3']:.0f}/"
          f"{g['w4']:.0f} um, L = 5 um, Cboot = {g['cboot']*1e12:.2f} pF")
    return g


def sizing(g, cboot=None):
    return Sizing(w1=g["w1"] * 1e-6, w2=g["w2"] * 1e-6, w3=g["w3"] * 1e-6,
                  w4=g["w4"] * 1e-6,
                  cboot=g["cboot"] if cboot is None else cboot)


def _job(a):
    g, vdd, kind, cap_mode, cap_scale, dvmax, nstage = a
    s = sizing(g)
    try:
        r, _ = run_ring_auto(s, kind, nstage, vdd, cap_mode=cap_mode,
                             cap_scale=cap_scale, dvmax=dvmax, max_steps=400000)
    except Exception as e:
        r = dict(vdd=vdd, oscillates=False, reason=f"{type(e).__name__}: {e}")
    d = {k: r.get(k, np.nan) for k in KEEP}
    d["vdd"] = vdd
    d["nstage"] = nstage
    return d


def main():
    g = gate()
    kind = "pCb" if g["cboot"] > 0 else "pC"

    print(f"\n--- VDD sweep, {NSTAGE}-stage ring, {len(VDD_GRID)} supplies ---")
    jobs = [(g, v, kind, "incremental", 1.0, 0.04, NSTAGE) for v in VDD_GRID]
    with Pool(os.cpu_count()) as pool:
        base = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    base["variant"] = "nominal"

    print("--- charge-formulation and capacitance-magnitude variants ---")
    jobs = ([(g, v, kind, "cv", 1.0, 0.04, NSTAGE) for v in VDD_GRID]
            + [(g, v, kind, "incremental", 0.5, 0.04, NSTAGE) for v in VDD_GRID]
            + [(g, v, kind, "incremental", 2.0, 0.04, NSTAGE) for v in VDD_GRID])
    with Pool(os.cpu_count()) as pool:
        var = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    var["variant"] = (["cap_mode=cv"] * len(VDD_GRID)
                      + ["cap x0.5"] * len(VDD_GRID) + ["cap x2"] * len(VDD_GRID))

    print("--- time-step convergence at VDD = 3 V ---")
    jobs = [(g, 3.0, kind, "incremental", 1.0, d, NSTAGE)
            for d in (0.08, 0.04, 0.02, 0.01)]
    with Pool(4) as pool:
        conv = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    conv["variant"] = "dt_refine"
    df = pd.concat([base, var, conv], ignore_index=True)
    df.to_csv(os.path.join(OUT, "ring5_vdd_sweep.csv"), index=False)
    print(f"wrote {OUT}/ring5_vdd_sweep.csv")

    print("\ndt refinement (backward Euler, period error is O(dt)):")
    for _, r in conv.iterrows():
        print(f"  dvmax = {r.dvmax:5.3f} V -> {r.steps:6.0f} steps, "
              f"f = {r.freq:10.1f} Hz")
    c = conv.dropna(subset=["freq"]).sort_values("dvmax")
    if len(c) >= 2:
        f_h, f_h2 = c.freq.iloc[1], c.freq.iloc[0]
        rich = 2 * f_h2 - f_h
        print(f"  Richardson-extrapolated f = {rich:,.1f} Hz; the dvmax = 0.04 V "
              f"value used for the sweep is {(base[base.vdd==3.0].freq.iloc[0]/rich-1)*100:+.2f}% off")

    b = base.copy()
    osc = b[b.oscillates].copy()
    print(f"\n--- VDD sweep results ({len(osc)}/{len(b)} supplies oscillate) ---")
    cols = ["vdd", "freq", "swing", "swing_pct", "vmin", "vmax", "x_max", "boost",
            "power", "e_per_transition", "tpd", "mode_index", "oob_frac"]
    pr = b[cols + ["oscillates"]].copy()
    pr["freq"] = pr.freq / 1e3
    pr["power"] = pr.power * 1e6
    pr["e_per_transition"] = pr.e_per_transition * 1e12
    pr["tpd"] = pr.tpd * 1e6
    pr.columns = ["VDD", "f[kHz]", "swing[V]", "swing%", "VOL", "VOH", "Xmax",
                  "boost", "P[uW]", "E/tr[pJ]", "tpd[us]", "mode", "oob", "osc"]
    print(pr.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # operating windows
    print("\n--- usable supply window ---")
    ok = osc[osc.oob_frac < 0.02]
    if len(ok):
        print(f"  oscillates at all                 : "
              f"{osc.vdd.min():.2f} .. {osc.vdd.max():.2f} V")
        print(f"  ... and model stays in its box    : "
              f"{ok.vdd.min():.2f} .. {ok.vdd.max():.2f} V")
        for thr in (50, 80, 90):
            s = ok[ok.swing_pct >= thr]
            if len(s):
                print(f"  ... and swing >= {thr}% of VDD      : "
                      f"{s.vdd.min():.2f} .. {s.vdd.max():.2f} V "
                      f"(f = {s.freq.min()/1e3:.1f} .. {s.freq.max()/1e3:.1f} kHz)")
        e = ok.loc[ok.e_per_transition.idxmin()]
        print(f"  minimum energy/transition         : {e.e_per_transition*1e12:.2f} pJ "
              f"at VDD = {e.vdd:.2f} V (f = {e.freq/1e3:.1f} kHz, "
              f"swing {e.swing_pct:.0f}%)")
        fm = ok.loc[ok.freq.idxmax()]
        print(f"  maximum frequency                 : {fm.freq/1e3:.1f} kHz at "
              f"VDD = {fm.vdd:.2f} V (swing {fm.swing_pct:.0f}%, "
              f"P = {fm.power*1e6:.0f} uW)")
        bad = osc[osc.oob_frac >= 0.02]
        if len(bad):
            print(f"  model-box violation above VDD     = {bad.vdd.min():.2f} V "
                  f"(node X is bootstrapped past the 5 V training limit)")
    # f vs VDD power law
    w = ok[(ok.swing_pct > 60) & (ok.freq > 0)]
    if len(w) > 4:
        p = np.polyfit(np.log(w.vdd), np.log(w.freq), 1)
        print(f"\n  f scales as VDD^{p[0]:.2f} over "
              f"{w.vdd.min():.2f}-{w.vdd.max():.2f} V "
              f"(fit residual {np.std(np.log(w.freq) - np.polyval(p, np.log(w.vdd))):.3f} in ln f)")

    print("\n--- model-uncertainty band on f ---")
    for v in (1.0, 2.0, 3.0, 4.0):
        row = base[base.vdd == v]
        if not len(row) or not bool(row.oscillates.iloc[0]):
            continue
        f0 = float(row.freq.iloc[0])
        bits = [f"nominal {f0/1e3:7.1f} kHz"]
        for lbl in ("cap_mode=cv", "cap x0.5", "cap x2"):
            r = var[(var.vdd == v) & (var.variant == lbl)]
            if len(r) and bool(r.oscillates.iloc[0]):
                bits.append(f"{lbl} {float(r.freq.iloc[0])/1e3:7.1f} kHz "
                            f"({(float(r.freq.iloc[0])/f0-1)*100:+6.1f}%)")
        print(f"  VDD = {v:.1f} V: " + " | ".join(bits))
    return g, kind, df, base, var


def plots(g, kind, base, var):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.ring import run_ring_auto

    b = base[base.oscillates].sort_values("vdd")
    inbox = b[b.oob_frac < 0.02]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
    A = ax.ravel()
    A[0].semilogy(b.vdd, b.freq / 1e3, "o-", c="C0", label="nominal")
    for lbl, c, m in (("cap_mode=cv", "C2", "s"), ("cap x0.5", "C1", "^"),
                      ("cap x2", "C3", "v")):
        d = var[(var.variant == lbl) & (var.oscillates)].sort_values("vdd")
        if len(d):
            A[0].semilogy(d.vdd, d.freq / 1e3, m + "--", c=c, ms=4, lw=1, label=lbl)
    A[0].set(xlabel="VDD [V]", ylabel="f [kHz]", title="oscillation frequency")
    A[0].legend(fontsize=8)
    A[1].plot(b.vdd, b.swing_pct, "o-", c="C0", label="swing / VDD")
    A[1].plot(b.vdd, 100 * b.vmax / b.vdd, "s--", c="C2", ms=4, label="VOH / VDD")
    A[1].plot(b.vdd, 100 * b.vmin / b.vdd, "v--", c="C3", ms=4, label="VOL / VDD")
    A[1].axhline(90, color="k", ls=":", lw=.8)
    A[1].set(xlabel="VDD [V]", ylabel="% of VDD", title="output levels")
    A[1].legend(fontsize=8)
    A[2].plot(b.vdd, b.x_max, "o-", c="C0", label="max $V_X$")
    A[2].plot(b.vdd, b.vdd, "k:", lw=.9, label="VDD")
    A[2].axhline(5.0, color="C3", ls="--", lw=1, label="model limit 5 V")
    A[2].set(xlabel="VDD [V]", ylabel="V [V]",
             title="bootstrap: node X vs the supply")
    A[2].legend(fontsize=8)
    A[3].loglog(b.vdd, b.power * 1e6, "o-", c="C0")
    A[3].set(xlabel="VDD [V]", ylabel="P [uW]", title="average supply power")
    A[4].semilogy(b.vdd, b.e_per_transition * 1e12, "o-", c="C0")
    if len(inbox):
        k = inbox.e_per_transition.idxmin()
        A[4].plot(inbox.vdd[k], inbox.e_per_transition[k] * 1e12, "*",
                  ms=16, c="C3", label="minimum")
        A[4].legend(fontsize=8)
    A[4].set(xlabel="VDD [V]", ylabel="energy / transition [pJ]",
             title="switching energy")
    A[5].plot(b.vdd, b.oob_frac * 100, "o-", c="C3")
    A[5].set(xlabel="VDD [V]", ylabel="% of time steps out of model box",
             title="compact-model validity")
    for a in A:
        a.grid(alpha=.3, which="both")
    fig.suptitle(f"{NSTAGE}-stage a-IGZO pseudo-CMOS bootstrapped ring: "
                 f"W1/W2/W3/W4 = {g['w1']:.0f}/{g['w2']:.0f}/{g['w3']:.0f}/{g['w4']:.0f} um, "
                 f"L = 5 um, Cboot = {g['cboot']*1e12:.2f} pF", y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "ring5_vdd_sweep.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # waveforms at 3 V, bootstrapped vs not
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for row, cb, ttl in ((0, g["cboot"], f"with Cboot = {g['cboot']*1e12:.2f} pF"),
                         (1, 0.0, "same sizing, Cboot = 0")):
        s = sizing(g, cboot=cb)
        r, (t, V, net, outs, xs) = run_ring_auto(
            s, "pCb" if cb else "pC", NSTAGE, 3.0)
        if not r["oscillates"]:
            continue
        t0 = t[-1] - 2.2 * r["period"]
        m = t >= t0
        tt = (t[m] - t0) * 1e6
        ax[row].plot(tt, V[m, net._idx[xs[0]]], c="C1", lw=1.6,
                     label="$V_X$ (gate of pull-up M3)")
        ax[row].plot(tt, V[m, net._idx[outs[0]]], c="C0", lw=2, label="$V_{OUT}$")
        ax[row].axhline(3.0, color="k", ls=":", lw=.9, label="VDD")
        ax[row].set(ylabel="V [V]",
                    title=f"{ttl}: f = {r['freq']/1e3:.1f} kHz, "
                          f"swing = {r['swing_pct']:.1f}% of VDD, "
                          f"max $V_X$ = {r['x_max']:.2f} V")
        ax[row].legend(fontsize=8, loc="center right")
        ax[row].grid(alpha=.3)
    ax[1].set_xlabel("time [us]")
    fig.suptitle("What the bootstrap capacitor does, VDD = 3 V", y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "ring5_bootstrap_waveforms.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    g, kind, df, base, var = main()
    plots(g, kind, base, var)
