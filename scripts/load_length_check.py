"""Does lengthening the stage-1 load cut the ring's power? (It is 80% static.)

Everything else in this study fixes L = 5 um on all four devices, because that
is the only length where the ANN's W scaling is monotonic. That argument is
about *comparing widths* at a given length, and the stage-1 load M1 is the one
device that only ever uses a single width -- so its length can be varied without
ever relying on the suspect W scaling.

It is worth checking because M1 sets the static current, and static power is
~80% of this ring's total (7.6 uW/gate DC against 49.1 uW for five gates). The
other logic study on this branch (src/logic.py) independently picked a 5/20 um
load for exactly this reason.

Model caveat: on the measured geometry grid W = 5 um exists only at L = 5 um, so
W5/L10, W5/L15 and W5/L20 are extrapolated corners. W20 at L = 10-20 um is
measured, and is included as the trustworthy comparison.

    python scripts/load_length_check.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import Sizing, screen_metrics          # noqa: E402
from src.ring import run_ring_auto                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
VDD, NSTAGE = 3.0, 5
W1 = (5e-6, 10e-6, 20e-6)
L1 = (5e-6, 10e-6, 15e-6, 20e-6)
CB = (1e-12, 2e-12)
MEASURED = {(5, 5), (10, 5), (10, 10), (20, 5), (20, 10), (20, 15), (20, 20)}


def base():
    r = pd.read_csv(os.path.join(OUT, "recommended_gate.csv")).iloc[0]
    return dict(w2=r.w2 * 1e-6, w3=r.w3 * 1e-6, w4=r.w4 * 1e-6)


def _job(a):
    w1, l1, cb, b = a
    s = Sizing(w1=w1, w2=b["w2"], w3=b["w3"], w4=b["w4"], l=5e-6, l1=l1, cboot=cb)
    d = dict(w1=w1 * 1e6, l1=l1 * 1e6, cboot=cb,
             measured=(round(w1 * 1e6), round(l1 * 1e6)) in MEASURED)
    try:
        m = screen_metrics(w1, b["w2"], b["w3"], b["w4"], "pC", VDD, npts=801, l1=l1)
        d.update(dc_gain_vm=m["gain_vm"], dc_voh=m["voh"], dc_vol=m["vol"],
                 dc_snm=m["snm"], dc_pstatic=m["p_static"])
        r, _ = run_ring_auto(s, "pCb", NSTAGE, VDD)
        d.update(freq=r.get("freq", np.nan), swing_pct=r.get("swing_pct", np.nan),
                 power=r.get("power", np.nan), x_max=r.get("x_max", np.nan),
                 boost=r.get("boost", np.nan),
                 e_per_transition=r.get("e_per_transition", np.nan),
                 oscillates=r["oscillates"])
    except Exception as e:
        d.update(oscillates=False, reason=f"{type(e).__name__}: {e}")
    return d


if __name__ == "__main__":
    b = base()
    print(f"stage-1 load sweep around W2/W3/W4 = {b['w2']*1e6:.0f}/{b['w3']*1e6:.0f}/"
          f"{b['w4']*1e6:.0f} um, L = 5 um, VDD = {VDD} V, {NSTAGE} stages")
    jobs = [(w, l, c, b) for w, l, c in itertools.product(W1, L1, CB)]
    with Pool(os.cpu_count()) as pool:
        df = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    df.to_csv(os.path.join(OUT, "load_length_check.csv"), index=False)
    o = df[df.oscillates].copy()
    o["pct_saved"] = 100 * (1 - o.power / float(
        o[(o.w1 == 5) & (o.l1 == 5) & (o.cboot == 1e-12)].power.iloc[0]))
    cols = ["w1", "l1", "cboot", "measured", "freq", "swing_pct", "power",
            "pct_saved", "dc_pstatic", "dc_gain_vm", "dc_snm", "dc_voh", "boost"]
    p = o[cols].copy()
    p["cboot"] = p.cboot * 1e12
    p["freq"] = p.freq / 1e3
    p["power"] = p.power * 1e6
    p["dc_pstatic"] = p.dc_pstatic * 1e6
    print(p.sort_values(["cboot", "w1", "l1"]).to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"))

    ref = o[(o.w1 == 5) & (o.l1 == 5) & (o.cboot == 1e-12)].iloc[0]
    print(f"\nreference (the recommended gate, L1 = 5 um): "
          f"{ref.freq/1e3:.1f} kHz, {ref.swing_pct:.1f}% swing, "
          f"{ref.power*1e6:.1f} uW")
    good = o[(o.swing_pct >= 90) & (o.dc_gain_vm >= 2.0)]

    def report(g, label):
        print(f"{label}: W1/L1 = {g.w1:.0f}/{g.l1:.0f} um, "
              f"Cboot = {g.cboot*1e12:.0f} pF -> {g.freq/1e3:.1f} kHz, "
              f"{g.swing_pct:.1f}% swing, {g.power*1e6:.1f} uW, loop gain "
              f"{g.dc_gain_vm:.2f}, SNM {g.dc_snm:.3f} V  "
              f"[{100*(g.power/ref.power-1):+.0f}% power, "
              f"{100*(g.freq/ref.freq-1):+.0f}% frequency vs the reference]")

    if len(good):
        report(good.loc[good.power.idxmin()],
               "lowest power with swing >= 90% and loop gain >= 2  ")
        gm = good[good.measured]
        if len(gm):
            report(gm.loc[gm.power.idxmin()],
                   "   ... restricted to measured geometries        ")
        else:
            print("   ... no measured geometry clears both bars")
    print("\nNote the non-monotonic L1 = 15 um row: it costs more power than "
          "L1 = 10 um and\nL1 = 20 um both. W = 5 um was only measured at "
          "L = 5 um, so the model's L scaling\nat that width is an "
          "extrapolation and is not even ordered correctly.")
