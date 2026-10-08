"""Does the zero-VGS-load variant (pCz/pCzb) ever ring, given enough time?

It has by far the best DC numbers -- gain ~118 and a 1.16 V butterfly eye at
65 nW, against ~4 and 0.76 V at 8 uW for the saturated-load pC -- so the claim
that it is unusable needs more than a 60 us window to stand up. Here it gets a
window 300x longer, with and without a bootstrap capacitor.

    python scripts/pczb_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic_sim import eval_batch                      # noqa: E402
from src.pseudo_cmos import Sizing, screen_metrics, node_caps   # noqa: E402
from src.ring import run_ring                             # noqa: E402

VDD = 3.0


def main():
    for (w1, w2, w3, w4) in [(5, 10, 160, 40), (20, 10, 80, 40)]:
        s0 = Sizing(w1=w1e(w1), w2=w1e(w2), w3=w1e(w3), w4=w1e(w4))
        m = screen_metrics(s0.w1, s0.w2, s0.w3, s0.w4, "pCz", VDD, npts=401)
        cx, _ = node_caps(s0, "pCz", VDD)
        # current M1 can supply to node X: a zero-VGS load is just the device's
        # own off-current, so the X slew rate is I/C
        i1 = float(np.ravel(eval_batch(np.array([0.0]), np.array([VDD * 0.5]),
                                       s0.w1, s0.l)["id"])[0])
        print(f"\npCz {w1}/{w2}/{w3}/{w4} um: DC gain {m['gain']:.0f}, "
              f"SNM {m['snm']:.2f} V, P_static {m['p_static']*1e9:.0f} nW")
        print(f"  M1 (zero-VGS load) sources {i1*1e9:.2f} nA into Cx = {cx*1e12:.2f} pF "
              f"-> dVx/dt = {i1/cx*1e-6:.4f} V/us; a {VDD/2:.1f} V swing needs "
              f"{VDD/2*cx/i1*1e3:.1f} ms")
        for cb, kind in ((0.0, "pCz"), (2e-12, "pCzb")):
            s = Sizing(w1=s0.w1, w2=s0.w2, w3=s0.w3, w4=s0.w4, cboot=cb)
            r, _ = run_ring(s, kind, 5, VDD, 20e-3, dvmax=0.05, max_steps=400000)
            print(f"  {kind:5s} Cboot={cb*1e12:.0f}pF, 20 ms window: "
                  + (f"f = {r['freq']:.1f} Hz, swing = {r['swing_pct']:.1f}%"
                     if r["oscillates"] else
                     f"NO OSCILLATION ({r['reason']}); node 0 ended at "
                     f"{r.get('vmin', float('nan')):.3f}..{r.get('vmax', float('nan')):.3f} V"
                     f", {r['steps']} steps, complete={r['complete']}"))


def w1e(x):
    return x * 1e-6


if __name__ == "__main__":
    main()
