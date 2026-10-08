"""Pick the best-performing bootstrapped pseudo-CMOS NOT gate, by ring simulation.

The DC screen cannot choose a bootstrapped gate: a capacitor does nothing in DC.
Worse, the delay proxy counts M3's CGS as load when it is actually the gate's
*intrinsic* bootstrap capacitor (it sits between X and OUT). So the sizing is
decided here, on the real figure of merit -- a 5-stage ring of the gate -- in two
phases:

  A  base sizing, Cboot = 0: which (W1,W2,W3,W4) oscillates fastest with the
     largest swing.
  B  bootstrap cap sweep on the phase-A leaders.

What phase A is expected to find, from the capacitance budget at node X:
CGS(M3) couples OUT into X (bootstrap) while CGD(M2) couples IN into X against
it, and at W2 = W3 = 160 um the two are both ~1.5 pF and cancel. Shrinking W2
and keeping W3 large should therefore be what lets the boost happen at all.

    python scripts/pseudo_cmos_best.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import Sizing, screen_metrics                 # noqa: E402
from src.ring import run_ring_auto                                 # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)
VDD = 3.0
NSTAGE = 5
KEEP = ["kind", "w1", "w2", "w3", "w4", "cboot", "freq", "swing", "swing_pct",
        "vmin", "vmax", "x_max", "boost", "power", "e_per_transition", "tpd",
        "jitter", "n_periods", "oob_frac", "steps", "oscillates", "reason"]

PH_A = dict(w1=(5e-6, 10e-6, 20e-6), w2=(10e-6, 20e-6, 40e-6, 80e-6),
            w3=(40e-6, 80e-6, 160e-6), w4=(20e-6, 40e-6, 80e-6, 160e-6))
CBOOT = (0.0, 0.5e-12, 1e-12, 2e-12, 4e-12, 8e-12)
# Phase C removes a selection bias in A: A is run at Cboot = 0, which rewards a
# big W3 purely because M3's own CGS *is* the bootstrap capacitor. Once an
# explicit cap is allowed, a smaller (cheaper, lower-crowbar) pull-up might win,
# so the W3/Cboot plane is re-swept jointly instead of sequentially.
PH_C = dict(w1=(5e-6, 10e-6), w2=(10e-6, 20e-6, 40e-6),
            w3=(40e-6, 80e-6, 160e-6), w4=(20e-6, 40e-6))
CBOOT_C = (1e-12, 2e-12, 4e-12)


def _job(a):
    w1, w2, w3, w4, cb, kind = a
    s = Sizing(w1=w1, w2=w2, w3=w3, w4=w4, cboot=cb)
    try:
        r, _ = run_ring_auto(s, kind, NSTAGE, VDD)
    except Exception as e:                                  # keep the sweep alive
        r = dict(kind=kind, w1=w1 * 1e6, w2=w2 * 1e6, w3=w3 * 1e6, w4=w4 * 1e6,
                 cboot=cb, oscillates=False, reason=f"{type(e).__name__}: {e}")
    d = {k: r.get(k, np.nan) for k in KEEP}
    d.update(kind=kind, w1=w1 * 1e6, w2=w2 * 1e6, w3=w3 * 1e6, w4=w4 * 1e6, cboot=cb)
    # DC metrics of the same sizing, for the record
    try:
        m = screen_metrics(w1, w2, w3, w4, "pC" if kind in ("pC", "pCb") else "pCz",
                           VDD, npts=1001)
        d.update(dc_gain=m["gain"], dc_gain_vm=m["gain_vm"], dc_voh=m["voh"],
                 dc_vol=m["vol"], dc_snm=m["snm"], dc_pstatic=m["p_static"])
    except Exception:
        pass
    return d


def show(df, n=14, by="score"):
    c = [c for c in ["kind", "w1", "w2", "w3", "w4", "cboot", "freq", "swing_pct",
                     "vmax", "x_max", "boost", "power", "dc_snm", "score"]
         if c in df.columns]
    d = df.copy()
    d["cboot"] = d.cboot * 1e12
    d["freq"] = d.freq / 1e3
    d["power"] = d.power * 1e6
    print(d.nlargest(n, by)[c].to_string(index=False,
                                         float_format=lambda x: f"{x:8.3f}"))


def score(df):
    """Rank for a ring oscillator: speed, but only with a usable swing.

    A gate that rings fast at 55% swing is not a logic gate, so swing enters as
    a hard-ish weight rather than a tiebreak: score = f * (swing fraction)^3.
    """
    return df.freq * (df.swing_pct / 100.0) ** 3


def phase_c():
    jobs = [(a, b, c, d, cb, "pCb") for a, b, c, d in
            itertools.product(PH_C["w1"], PH_C["w2"], PH_C["w3"], PH_C["w4"])
            for cb in CBOOT_C]
    print(f"phase C: joint W3 x Cboot re-sweep, {len(jobs)} runs")
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(OUT, "ring5_sizing_phaseC.csv"), index=False)
    oc = c[c.oscillates].copy()
    oc["score"] = score(oc)
    print(f"\n{len(oc)}/{len(c)} oscillate. Top by f*swing^3:")
    show(oc, n=14)
    print("\nbest at each W3 (does a smaller pull-up win once Cboot is free?):")
    for w3 in sorted(oc.w3.unique()):
        d = oc[oc.w3 == w3].nlargest(1, "score").iloc[0]
        print(f"  W3 = {w3:5.0f} um: {d.w1:.0f}/{d.w2:.0f}/{d.w3:.0f}/{d.w4:.0f}, "
              f"Cboot = {d.cboot*1e12:.0f} pF -> f = {d.freq/1e3:6.1f} kHz, "
              f"swing = {d.swing_pct:5.1f}%, P = {d.power*1e6:6.1f} uW, "
              f"score = {d.score:,.0f}")
    return oc


# Selecting on ring performance alone is a trap twice over.
#
# First, a ring oscillates with a gate that is a poor *logic* gate: the loop only
# has to clear unity gain. Maximising f*swing^3 lands on W2 = 10 um, 225 kHz --
# whose loop gain is 1.63, barely regenerative.
#
# Second, "gain" has to mean the slope at the *trip point*, not max |dVout/dVin|.
# The zero-VGS variant peaks at |A| = 220 at VIN = 4.5 mV, nowhere near its own
# VM = 0.238 V where the slope is only -0.53 -- so its symmetric ring point is
# stable and it latches, despite the best DC gain and noise margin in the study.
#
# So the rule is stated up front rather than tuned: among designs that swing at
# least SWING_MIN of VDD, take the (frequency, loop-gain-at-VM) Pareto front, and
# off it choose the FASTEST point whose loop gain clears GAIN_MIN -- i.e. the
# fastest gate that still regenerates 2x per stage. Cboot is then the smallest
# capacitor on that sizing reaching SWING_PICK.
SWING_MIN, GAIN_MIN, SWING_PICK = 90.0, 2.0, 92.0


def pick(ob, vdd=VDD):
    d = ob[ob.swing_pct >= SWING_MIN].copy()
    if not len(d):
        print("  nothing reaches the swing floor -- falling back to raw score")
        d = ob.copy()
    # fastest full-swing run per sizing, then the Pareto front in (f, gain_vm)
    per = d.sort_values("freq", ascending=False).groupby(
        ["w1", "w2", "w3", "w4"], as_index=False).first().sort_values(
        "freq", ascending=False)
    front, best = [], -np.inf
    for _, r in per.iterrows():
        if r.dc_gain_vm > best:
            front.append(r)
            best = r.dc_gain_vm
    f = pd.DataFrame(front)
    print(f"\n  Pareto front, speed vs loop gain at VM (swing >= {SWING_MIN:.0f}%):")
    print(f[["w1", "w2", "w3", "w4", "cboot", "freq", "swing_pct", "dc_gain_vm",
             "dc_snm", "power"]]
          .assign(cboot=lambda x: x.cboot * 1e12, freq=lambda x: x.freq / 1e3,
                  power=lambda x: x.power * 1e6)
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    ok = f[f.dc_gain_vm >= GAIN_MIN]
    if not len(ok):
        print(f"  nothing on the front clears loop gain {GAIN_MIN}")
        ok = f
    chosen = ok.nlargest(1, "freq").iloc[0]
    same = d[(d.w1 == chosen.w1) & (d.w2 == chosen.w2) &
             (d.w3 == chosen.w3) & (d.w4 == chosen.w4)]
    good = same[same.swing_pct >= SWING_PICK]
    best = (good.nsmallest(1, "cboot") if len(good) else same.nlargest(1, "swing_pct")).iloc[0]
    print(f"\n  fastest front point with loop gain >= {GAIN_MIN}: "
          f"{chosen.w1:.0f}/{chosen.w2:.0f}/{chosen.w3:.0f}/{chosen.w4:.0f}; "
          f"smallest Cboot on it reaching {SWING_PICK:.0f}% swing is "
          f"{best.cboot*1e12:.2f} pF")
    return best


def announce(best):
    print("\n=== recommended bootstrapped pseudo-CMOS NOT gate ===")
    for k in ("w1", "w2", "w3", "w4"):
        print(f"  {k.upper()} = {best[k]:.0f} um")
    print(f"  L     = 5 um (all devices)")
    print(f"  Cboot = {best.cboot*1e12:.2f} pF")
    print(f"  -> 5-stage ring at VDD=3: f = {best.freq/1e3:.1f} kHz, "
          f"swing = {best.swing_pct:.1f}% of VDD, X peaks at {best.x_max:.3f} V "
          f"({best.boost:+.3f} V vs VDD), P = {best.power*1e6:.1f} uW")
    pd.DataFrame([best]).to_csv(os.path.join(OUT, "recommended_gate.csv"), index=False)
    print(f"wrote {OUT}/recommended_gate.csv")


if __name__ == "__main__":
    if "--phase-c" in sys.argv:
        oc = phase_c()
        prev = pd.read_csv(os.path.join(OUT, "ring5_sizing_phaseB.csv"))
        prev = prev[prev.oscillates].copy()
        prev["score"] = score(prev)
        allr = pd.concat([prev, oc], ignore_index=True)
        announce(pick(allr))
        sys.exit(0)
    jobs = [(a, b, c, d, 0.0, "pC") for a, b, c, d in
            itertools.product(PH_A["w1"], PH_A["w2"], PH_A["w3"], PH_A["w4"])]
    # a few pCz points too: its DC gain is 25x pC's, so it is worth knowing
    # whether the style can ring at all once the bootstrap supplies the charge
    jobs += [(a, b, c, d, cb, "pCzb") for a, b, c, d, cb in
             [(5e-6, 10e-6, 160e-6, 80e-6, 0.0), (5e-6, 10e-6, 160e-6, 80e-6, 2e-12),
              (20e-6, 10e-6, 80e-6, 40e-6, 0.0), (20e-6, 10e-6, 80e-6, 40e-6, 2e-12),
              (5e-6, 10e-6, 40e-6, 20e-6, 2e-12), (20e-6, 10e-6, 40e-6, 20e-6, 2e-12)]]
    print(f"phase A: {len(jobs)} 5-stage ring transients at VDD = {VDD} V")
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(OUT, "ring5_sizing_phaseA.csv"), index=False)
    osc = a[a.oscillates].copy()
    osc["score"] = score(osc)
    print(f"\n{len(osc)}/{len(a)} oscillate. Top by f*swing^3:")
    show(osc)
    print("\nTop by raw frequency:")
    show(osc, by="freq")
    print("\nTop by swing:")
    show(osc, by="swing_pct")
    pz = a[(a.kind == "pCzb")]
    print(f"\npCzb (zero-VGS load) points: {int(pz.oscillates.sum())}/{len(pz)} oscillate")
    if len(pz):
        print(pz[["w1", "w2", "w3", "w4", "cboot", "freq", "swing_pct", "oscillates",
                  "reason"]].to_string(index=False))

    base = osc.nlargest(8, "score")[["w1", "w2", "w3", "w4"]].drop_duplicates()
    jobs_b = [(r.w1 * 1e-6, r.w2 * 1e-6, r.w3 * 1e-6, r.w4 * 1e-6, cb, "pCb")
              for _, r in base.iterrows() for cb in CBOOT]
    print(f"\nphase B: bootstrap cap sweep, {len(jobs_b)} runs "
          f"({len(base)} sizings x {len(CBOOT)} caps)")
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(_job, jobs_b, chunksize=2)
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(OUT, "ring5_sizing_phaseB.csv"), index=False)
    ob = b[b.oscillates].copy()
    ob["score"] = score(ob)
    print("\nphase B, top by f*swing^3:")
    show(ob, n=16)
    oc = phase_c()
    announce(pick(pd.concat([ob, oc], ignore_index=True)))
