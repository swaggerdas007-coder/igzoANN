"""How far the a-IGZO pseudo-CMOS bootstrapped ring can be pushed: stage count.

Four questions, in order:

  1. How few stages?   N = 1 is negative feedback (it settles at the trip point),
     N = 2, 4, 6 ... latch. The floor is N = 3, and that is also where the
     highest frequency lives.
  2. How many stages?  N is swept to 101. For static (ratioed) logic there is no
     charge-retention limit, so the expectation is f * N = const; the point of
     measuring it is to find where that *stops* holding and why.
  3. Which mode?       A long ring also supports multi-wave modes at k x the
     fundamental (k odd). Each N is run from both initial conditions --
     "alternating" rails, which is the highest spatial mode, and a 1 mV nudge off
     the symmetric DC point, which grows the fundamental -- and the measured
     mode index says which one survived.
  4. Supply window vs N: the VDD sweep of ring_osc5.py, repeated at several N.

    python scripts/ring_osc_stages.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import Sizing                   # noqa: E402
from src.ring import run_ring, run_ring_auto        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)

N_ODD = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101]
N_EVEN = [2, 4, 6, 10]
VDD_REF = 3.0
VDD_GRID = [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
N_FOR_VDD = [3, 5, 11, 31]
KEEP = ["nstage", "vdd", "freq", "period", "swing", "swing_pct", "vmin", "vmax",
        "x_max", "boost", "power", "i_supply", "e_per_cycle", "e_per_transition",
        "tpd", "jitter", "n_periods", "mode_index", "oob_frac", "steps", "init",
        "oscillates", "reason"]
FALLBACK = dict(w1=5.0, w2=20.0, w3=160.0, w4=40.0, cboot=0.0)


def gate():
    p = os.path.join(OUT, "recommended_gate.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        return dict(w1=float(r.w1), w2=float(r.w2), w3=float(r.w3),
                    w4=float(r.w4), cboot=float(r.cboot))
    print("recommended_gate.csv missing -- using the fallback sizing")
    return dict(FALLBACK)


def sizing(g):
    return Sizing(w1=g["w1"] * 1e-6, w2=g["w2"] * 1e-6, w3=g["w3"] * 1e-6,
                  w4=g["w4"] * 1e-6, cboot=g["cboot"])


def _job(a):
    g, n, vdd, init, tpd_ref = a
    kind = "pCb" if g["cboot"] > 0 else "pC"
    s = sizing(g)
    # start the window near the expected period instead of letting the auto
    # stretcher burn two short runs: T ~ 2*N*tpd, and we want ~9 of them
    # f measured to go as VDD^0.93, so the period scales ~1/VDD; start the
    # window at ~9 expected periods so the auto-stretcher rarely has to fire
    t_end = max(9.0 * 2.0 * n * tpd_ref * (VDD_REF / vdd) ** 1.1, 20e-6)
    try:
        r, _ = run_ring_auto(s, kind, n, vdd, t_end=t_end, init=init,
                             max_steps=900000, tries=2)
    except Exception as e:
        r = dict(oscillates=False, reason=f"{type(e).__name__}: {e}")
    d = {k: r.get(k, np.nan) for k in KEEP}
    d.update(nstage=n, vdd=vdd, init=init)
    return d


def main():
    g = gate()
    kind = "pCb" if g["cboot"] > 0 else "pC"
    print(f"gate: W1/W2/W3/W4 = {g['w1']:.0f}/{g['w2']:.0f}/{g['w3']:.0f}/"
          f"{g['w4']:.0f} um, L = 5 um, Cboot = {g['cboot']*1e12:.2f} pF, "
          f"kind = {kind}")

    r5, _ = run_ring_auto(sizing(g), kind, 5, VDD_REF)
    tpd_ref = r5["tpd"]
    print(f"reference: 5-stage at {VDD_REF} V -> f = {r5['freq']/1e3:.1f} kHz, "
          f"tpd = {tpd_ref*1e9:.0f} ns/stage\n")

    print(f"--- stage-count sweep at VDD = {VDD_REF} V "
          f"(odd N from both initial conditions, plus even N) ---")
    jobs = ([(g, n, VDD_REF, init, tpd_ref) for n in N_ODD
             for init in ("alternating", "perturb")]
            + [(g, n, VDD_REF, "alternating", tpd_ref) for n in N_EVEN])
    with Pool(os.cpu_count()) as pool:
        ns = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    ns.to_csv(os.path.join(OUT, "ring_stage_sweep.csv"), index=False)

    pr = ns.copy()
    pr["fN"] = pr.freq * pr.nstage
    cols = ["nstage", "init", "freq", "fN", "swing_pct", "vmax", "tpd",
            "mode_index", "power", "e_per_transition", "jitter", "oscillates"]
    d = pr[cols].copy()
    d["freq"] = d.freq / 1e3
    d["fN"] = d.fN / 1e3
    d["tpd"] = d.tpd * 1e9
    d["power"] = d.power * 1e6
    d["e_per_transition"] = d.e_per_transition * 1e12
    d["jitter"] = d.jitter * 1e9
    d.columns = ["N", "init", "f[kHz]", "f*N[kHz]", "swing%", "VOH", "tpd[ns]",
                 "mode", "P[uW]", "E/tr[pJ]", "jit[ns]", "osc"]
    print(d.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    o = ns[ns.oscillates & (ns.init == "perturb")].sort_values("nstage")
    if len(o) > 2:
        fn = o.freq * o.nstage
        print(f"\n  f*N over N = {int(o.nstage.min())}..{int(o.nstage.max())}: "
              f"{fn.min()/1e3:.1f} .. {fn.max()/1e3:.1f} kHz "
              f"(spread {(fn.max()/fn.min()-1)*100:.1f}%)")
        print(f"  tpd/stage: {o.tpd.min()*1e9:.0f} .. {o.tpd.max()*1e9:.0f} ns "
              f"-- flat means the delay per stage does not care how long the ring is")
        print(f"  swing: {o.swing_pct.min():.1f} .. {o.swing_pct.max():.1f} % of VDD")
        print(f"  total power scales {o.power.iloc[0]*1e6:.1f} uW at N={int(o.nstage.iloc[0])} "
              f"-> {o.power.iloc[-1]*1e6:.1f} uW at N={int(o.nstage.iloc[-1])} "
              f"({o.power.iloc[-1]/o.power.iloc[0]:.1f}x for "
              f"{o.nstage.iloc[-1]/o.nstage.iloc[0]:.1f}x the stages)")
        ep = o.e_per_transition * 1e12
        print(f"  energy/transition: {ep.min():.2f} .. {ep.max():.2f} pJ, "
              f"rising ~linearly with N -- this logic is static-power dominated, "
              f"so a longer (slower) ring integrates that current over a longer "
              f"period and each transition costs more")
    alt = ns[ns.oscillates & (ns.init == "alternating")]
    hi = alt[alt.mode_index > 2]
    print(f"\n  multi-wave modes from the alternating start: "
          f"{len(hi)}/{len(alt)} runs came up on mode k > 1"
          + (f" -- N = {sorted(hi.nstage.unique().tolist())}" if len(hi) else ""))
    ev = ns[ns.nstage.isin(N_EVEN)]
    print(f"  even N: {int(ev.oscillates.sum())}/{len(ev)} oscillate "
          f"(N = {sorted(ev.nstage.tolist())}) -- an even ring is a latch")

    print(f"\n--- VDD sweep at N = {N_FOR_VDD} ---")
    jobs = [(g, n, v, "perturb", tpd_ref) for n in N_FOR_VDD for v in VDD_GRID]
    with Pool(os.cpu_count()) as pool:
        vs = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    vs.to_csv(os.path.join(OUT, "ring_stage_vdd_sweep.csv"), index=False)
    for n in N_FOR_VDD:
        d = vs[(vs.nstage == n)].sort_values("vdd")
        ok = d[d.oscillates & (d.oob_frac < 0.02)]
        if not len(ok):
            print(f"  N = {n:3d}: never oscillated inside the model box")
            continue
        s90 = ok[ok.swing_pct >= 90]
        print(f"  N = {n:3d}: oscillates {ok.vdd.min():.1f}-{ok.vdd.max():.1f} V, "
              f"f = {ok.freq.min()/1e3:7.2f}-{ok.freq.max()/1e3:7.2f} kHz, "
              f"P = {ok.power.min()*1e6:6.1f}-{ok.power.max()*1e6:6.1f} uW"
              + (f", swing>=90% from {s90.vdd.min():.1f} V" if len(s90) else
                 f", swing peaks at {ok.swing_pct.max():.0f}%"))
    return g, ns, vs


def plots(g, ns, vs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
    A = ax.ravel()
    for init, c, m in (("perturb", "C0", "o"), ("alternating", "C1", "s")):
        d = ns[ns.oscillates & (ns.init == init)].sort_values("nstage")
        if len(d):
            A[0].loglog(d.nstage, d.freq / 1e3, m + "-", c=c, ms=5, label=init)
            A[1].semilogx(d.nstage, d.freq * d.nstage / 1e3, m + "-", c=c, ms=5,
                          label=init)
            A[2].semilogx(d.nstage, d.swing_pct, m + "-", c=c, ms=5, label=init)
            A[4].loglog(d.nstage, d.power * 1e6, m + "-", c=c, ms=5, label=init)
            A[5].semilogx(d.nstage, d.mode_index, m + "-", c=c, ms=5, label=init)
    d = ns[ns.oscillates & (ns.init == "perturb")].sort_values("nstage")
    if len(d):
        A[0].loglog(d.nstage, d.freq.iloc[0] * d.nstage.iloc[0] / d.nstage / 1e3,
                    "k:", lw=1, label=r"$\propto 1/N$")
        A[3].semilogx(d.nstage, d.tpd * 1e9, "o-", c="C0", ms=5, label="perturb")
    A[0].set(xlabel="stages N", ylabel="f [kHz]", title="frequency vs ring length")
    A[1].set(xlabel="stages N", ylabel=r"$f \times N$ [kHz]",
             title=r"$f \times N$ = $1/(2 t_{pd})$: constant if the gate is")
    A[2].set(xlabel="stages N", ylabel="swing [% of VDD]", title="output swing")
    A[3].set(xlabel="stages N", ylabel="$t_{pd}$ per stage [ns]",
             title="delay per stage")
    A[4].set(xlabel="stages N", ylabel="total power [uW]",
             title="supply power (linear in N)")
    A[5].set(xlabel="stages N", ylabel="mode index k",
             title="spatial mode (1 = fundamental)")
    A[5].axhline(1, color="k", ls=":", lw=.8)
    for a in A:
        a.grid(alpha=.3, which="both")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle(f"How far the ring goes: a-IGZO pseudo-CMOS, "
                 f"{g['w1']:.0f}/{g['w2']:.0f}/{g['w3']:.0f}/{g['w4']:.0f} um, "
                 f"L = 5 um, Cboot = {g['cboot']*1e12:.2f} pF, VDD = {VDD_REF} V",
                 y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "ring_stage_sweep.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for n, c in zip(N_FOR_VDD, ("C0", "C1", "C2", "C3")):
        d = vs[(vs.nstage == n) & vs.oscillates].sort_values("vdd")
        if not len(d):
            continue
        ax[0].semilogy(d.vdd, d.freq / 1e3, "o-", c=c, ms=4, label=f"N = {n}")
        ax[1].plot(d.vdd, d.swing_pct, "o-", c=c, ms=4, label=f"N = {n}")
        ax[2].semilogy(d.vdd, d.e_per_transition * 1e12, "o-", c=c, ms=4,
                       label=f"N = {n}")
    ax[0].set(xlabel="VDD [V]", ylabel="f [kHz]", title="frequency vs supply")
    ax[1].set(xlabel="VDD [V]", ylabel="swing [% of VDD]", title="swing vs supply")
    ax[1].axhline(90, color="k", ls=":", lw=.8)
    ax[2].set(xlabel="VDD [V]", ylabel="energy / transition [pJ]",
              title="switching energy vs supply")
    for a in ax:
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("Supply-voltage window at several ring lengths", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "ring_stage_vdd_sweep.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    g, ns, vs = main()
    plots(g, ns, vs)
