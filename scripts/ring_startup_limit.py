"""Startup of the ring: the minimum supply it needs, and how long it takes.

Part 1 -- minimum supply voltage vs ring length, against the Barkhausen prediction.

An N-stage ring starts only if each stage's small-signal gain clears
|A| >= sec(pi/N): the loop needs 180 degrees of phase, each stage can only
contribute pi/N of it from its own pole, and the rest has to come from the
inversion, which costs gain. So a *short* ring needs a *higher* supply -- N = 3
needs |A| >= 2, N = 31 needs only 1.005.

This takes the measured DC gain-vs-VDD curve of the recommended gate, inverts
sec(pi/N) through it to predict each ring's minimum supply, and checks the
prediction by bisecting the real transient startup threshold: a 1 mV nudge off
the symmetric DC point either grows or decays.

    python scripts/ring_startup_limit.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic_sim import period_from_crossings          # noqa: E402
from src.pseudo_cmos import Sizing, screen_metrics        # noqa: E402
from src.ring import run_ring                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
N_LIST = [3, 5, 7, 11, 21, 31]
N_TIME = [5, 11, 21, 31, 51, 71]
VLO, VHI, NBISECT = 0.45, 3.2, 7


def gate():
    r = pd.read_csv(os.path.join(OUT, "recommended_gate.csv")).iloc[0]
    return Sizing(w1=r.w1 * 1e-6, w2=r.w2 * 1e-6, w3=r.w3 * 1e-6,
                  w4=r.w4 * 1e-6, cboot=float(r.cboot))


TPD_REF, VDD_REF = 515e-9, 3.0


def est_period(n, vdd):
    """T = 2*N*tpd, with tpd ~ 1/VDD (f was measured to go as VDD^1.00)."""
    return 2.0 * n * TPD_REF * (VDD_REF / vdd)


def starts(s, n, vdd, nper=40):
    """True if a 1 mV perturbation off the symmetric DC point grows to a swing.

    dt_max must be tied to the *period*, not to the window: left at the default
    the stepper takes ~7 steps per period at these window lengths and backward
    Euler's own damping then beats the physical growth rate, so every ring looks
    stable. That is a property of the integrator, not of the circuit.
    """
    T = est_period(n, vdd)
    try:
        r, (t, V, net, outs, xs) = run_ring(
            s, "pCb" if s.cboot > 0 else "pC", n, vdd, nper * T,
            dvmax=0.01, dt_max=T / 150.0, init="perturb", max_steps=400000)
    except Exception:
        return False, 0.0
    o = V[:, net._idx[outs[0]]]
    late = t >= t[-1] * 0.7
    sw = float(o[late].max() - o[late].min())
    return bool(sw > 0.2 * vdd), sw


def _job(a):
    s, n = a
    lo, hi = VLO, VHI
    if not starts(s, n, hi)[0]:
        return dict(nstage=n, vmin_sim=np.nan)
    if starts(s, n, lo)[0]:
        return dict(nstage=n, vmin_sim=VLO)
    for _ in range(NBISECT):
        mid = 0.5 * (lo + hi)
        if starts(s, n, mid)[0]:
            hi = mid
        else:
            lo = mid
    return dict(nstage=n, vmin_sim=0.5 * (lo + hi), vmin_lo=lo, vmin_hi=hi)


def _startup_time(a):
    """Cycles needed to grow from a 1 mV nudge to 90% of the final amplitude."""
    s, n, tpd = a
    T = 2.0 * n * tpd
    try:
        r, (t, V, net, outs, xs) = run_ring(
            s, "pCb" if s.cboot > 0 else "pC", n, 3.0, 120.0 * T,
            dvmax=0.02, dt_max=T / 120.0, init="perturb", max_steps=900000)
    except Exception as e:
        return dict(nstage=n, cycles=np.nan, reason=str(e))
    o = V[:, net._idx[outs[0]]]
    final = float(o[t >= t[-1] * 0.85].max() - o[t >= t[-1] * 0.85].min())
    if final < 0.3:
        return dict(nstage=n, cycles=np.nan, final_swing=final,
                    reason="did not start within 120 periods")
    # running peak-to-peak over a one-period window
    env = np.array([np.ptp(o[(t >= tt - T) & (t <= tt)]) if tt > T else 0.0
                    for tt in t])
    k = np.argmax(env >= 0.9 * final)
    return dict(nstage=n, cycles=float(t[k] / T), t_start=float(t[k]),
                final_swing=final, period=T, reason="")


def main():
    s = gate()
    print(f"gate: {s.w1*1e6:.0f}/{s.w2*1e6:.0f}/{s.w3*1e6:.0f}/{s.w4*1e6:.0f} um, "
          f"L = {s.l*1e6:.0f} um, Cboot = {s.cboot*1e12:.2f} pF\n")

    vg = np.arange(0.40, 3.01, 0.05)
    gains = np.array([screen_metrics(s.w1, s.w2, s.w3, s.w4, "pC", float(v),
                                     npts=301)["gain"] for v in vg])
    print("DC small-signal gain of one stage vs supply:")
    for v, g in zip(vg[::4], gains[::4]):
        print(f"  VDD = {v:.2f} V -> |A| = {g:.3f}")

    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_job, [(s, n) for n in N_LIST], chunksize=1)
    df = pd.DataFrame(rows)
    df["a_req"] = 1.0 / np.cos(np.pi / df.nstage)
    df["vmin_pred"] = [float(np.interp(a, gains, vg)) if a <= gains.max() else np.nan
                       for a in df.a_req]
    df.to_csv(os.path.join(OUT, "ring_startup_limit.csv"), index=False)

    print("\n  N   |A| needed = sec(pi/N)   VDD predicted   VDD measured (bisected)")
    for _, r in df.iterrows():
        print(f" {int(r.nstage):3d}        {r.a_req:7.4f}            "
              f"{r.vmin_pred:6.3f} V        {r.vmin_sim:6.3f} V"
              f"   (bracket {r.get('vmin_lo', np.nan):.3f}-{r.get('vmin_hi', np.nan):.3f})")
    d = df.dropna(subset=["vmin_pred", "vmin_sim"])
    if len(d):
        err = np.abs(d.vmin_sim - d.vmin_pred)
        print(f"\n  prediction error: max {err.max()*1e3:.0f} mV, "
              f"mean {err.mean()*1e3:.0f} mV over N = {sorted(d.nstage.tolist())}")
    print("\n  -> a SHORTER ring needs a HIGHER supply. The 3-stage ring, the fastest "
          "one,\n     is also the one that dies first as VDD is lowered.")

    print("\n--- part 2: how long does a ring take to start from a 1 mV nudge? ---")
    tpd = 515e-9
    with Pool(os.cpu_count()) as pool:
        st = pd.DataFrame(pool.map(_startup_time, [(s, n, tpd) for n in N_TIME],
                                   chunksize=1))
    st.to_csv(os.path.join(OUT, "ring_startup_time.csv"), index=False)
    print("    N   period      startup       cycles to 90% amplitude")
    for _, r in st.iterrows():
        if not np.isfinite(r.get("cycles", np.nan)):
            print(f"  {int(r.nstage):3d}   {'':8s}  {'':10s}   not started within "
                  f"120 periods ({r.reason})")
            continue
        print(f"  {int(r.nstage):3d}   {r.period*1e6:7.2f} us  "
              f"{r.t_start*1e6:8.1f} us   {r.cycles:6.1f}")
    d2 = st.dropna(subset=["cycles"])
    if len(d2) > 2:
        p = np.polyfit(np.log(d2.nstage), np.log(d2.t_start), 1)
        p2 = np.polyfit(np.log(d2.nstage), np.log(d2.cycles), 1)
        print(f"\n  startup time ~ N^{p[0]:.2f}, i.e. N^{p2[0]:.2f} in *cycles* -- "
              f"the period only grows as N,\n     so a long ring needs "
              f"disproportionately many cycles to come up from noise. That is why "
              f"the\n     perturbation start in ring_osc_stages.py fails for "
              f"N >= 71 in a 9-period window\n     while the full-rail "
              f"'alternating' start, which begins at amplitude, does not.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(vg, gains, "C0-", lw=2)
    ax[0].axhline(1.0, color="k", ls=":", lw=.8, label="|A| = 1")
    for _, r in df.iterrows():
        ax[0].plot([r.vmin_pred], [r.a_req], "o", ms=6,
                   label=f"N = {int(r.nstage)}: sec(pi/N) = {r.a_req:.3f}")
    ax[0].set(xlabel="VDD [V]", ylabel="stage gain |A|",
              title="one stage's DC gain sets the floor")
    ax[0].legend(fontsize=7)
    ax[1].plot(df.nstage, df.vmin_pred, "s--", c="C1", label="sec(pi/N) prediction")
    ax[1].plot(df.nstage, df.vmin_sim, "o-", c="C0", label="bisected from transient")
    ax[1].set(xscale="log", xlabel="stages N", ylabel="minimum VDD [V]",
              title="minimum supply vs ring length")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle("Why a short ring needs a bigger supply", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "ring_startup_limit.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
