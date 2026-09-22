"""Design-space sweep for the n-only a-IGZO differential pair.

Sweeps geometry and bias for both loads, keeps only points where every device
stays inside the ANN's trained bias box (no clamped input), and reports the
gain / bandwidth / power Pareto.

Run:  python scripts/diffpair_sweep.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diffpair import build, analyse, solve_rl, OUT1, OUT2, TAIL  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")
os.makedirs(OUT, exist_ok=True)

VDD = 5.0
CLOAD = 1e-12
RL_MAX = 10e6        # a thin-film load resistor you can actually lay out
ITAIL_MIN = 0.5e-6   # below this the tail device's own ro wrecks the CMRR
# the CGD/CGS nets were fit to only (20,20),(40,20),(160,15),(160,20) um --
# anything else is capacitance extrapolation, so flag it.
CAP_TRAINED = {(20, 20), (40, 20), (160, 15), (160, 20)}


def rows_res():
    W1 = [20e-6, 40e-6, 80e-6, 160e-6]
    L1 = [5e-6, 10e-6, 15e-6, 20e-6]
    VCM = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    VB = [0.8, 1.0, 1.2, 1.5, 2.0]
    VDROP = [1.0, 1.5, 2.0, 2.5]
    out = []
    for w1, l1, vcm, vb, vdrop in itertools.product(W1, L1, VCM, VB, VDROP):
        try:
            c, rl = solve_rl(w1, l1, 40e-6, 20e-6, VDD, vcm, vb, vdrop, CLOAD)
        except Exception:
            continue
        if c is None or not (1e3 < rl < RL_MAX):
            continue
        r = analyse(c, VDD, CLOAD)
        if r is None or r["itail"] < ITAIL_MIN:
            continue
        r.update(load="res", W1=w1 * 1e6, L1=l1 * 1e6, vcm=vcm, vb=vb,
                 vdrop=vdrop, rl=rl, WL=np.nan, LL=np.nan,
                 cap_extrapolated=(round(w1 * 1e6), round(l1 * 1e6)) not in CAP_TRAINED)
        out.append(r)
    return out


def rows_diode():
    W1 = [40e-6, 80e-6, 160e-6]
    L1 = [5e-6, 10e-6, 15e-6, 20e-6]
    LOADS = [(5e-6, 20e-6), (10e-6, 20e-6), (20e-6, 20e-6), (40e-6, 20e-6)]
    VCM = [0.8, 1.0, 1.2, 1.5, 2.0]
    VB = [0.8, 1.0, 1.2, 1.5, 2.0]
    out = []
    for w1, l1, (wl, ll), vcm, vb in itertools.product(W1, L1, LOADS, VCM, VB):
        try:
            c = build("diode", w1, l1, 40e-6, 20e-6, VDD, vcm, vb, wl=wl, ll=ll, cload=CLOAD)
            r = analyse(c, VDD, CLOAD)
        except Exception:
            continue
        if r is None or r["itail"] < ITAIL_MIN:
            continue
        r.update(load="diode", W1=w1 * 1e6, L1=l1 * 1e6, vcm=vcm, vb=vb,
                 vdrop=np.nan, rl=np.nan, WL=wl * 1e6, LL=ll * 1e6,
                 cap_extrapolated=((round(w1 * 1e6), round(l1 * 1e6)) not in CAP_TRAINED
                                   or (round(wl * 1e6), round(ll * 1e6)) not in CAP_TRAINED))
        out.append(r)
    return out


def main():
    rows = rows_res() + rows_diode()
    df = pd.DataFrame(rows)
    df["itail_uA"] = df.itail * 1e6
    df["power_uW"] = df.power * 1e6
    df["f3db_kHz"] = df.f3db / 1e3
    df["gbw_kHz"] = df.gbw / 1e3
    df = df.sort_values("av_db", ascending=False)
    df.to_csv(os.path.join(OUT, "diffpair_sweep.csv"), index=False)
    print(f"{len(df)} valid operating points (all devices inside the trained box)\n")

    cols = ["load", "W1", "L1", "WL", "LL", "vcm", "vb", "vdrop", "rl",
            "itail_uA", "vout", "av", "av_db", "f3db_kHz", "gbw_kHz",
            "power_uW", "cmrr_db", "feedthrough_limited", "cap_extrapolated"]
    fmt = dict(float_format=lambda x: f"{x:.3g}", index=False)

    for load in ("res", "diode"):
        d = df[df.load == load]
        print(f"===== {load} load: top 8 by gain =====")
        print(d.head(8)[cols].to_string(**fmt))
        u = d[~d.feedthrough_limited]
        print(f"\n----- {load} load: top 8 by GBW (usable -3dB corner only) -----")
        print(u.sort_values("gbw", ascending=False).head(8)[cols].to_string(**fmt))
        print()

    print("===== best compromise: gain >= 10 dB, maximise GBW =====")
    good = df[(df.av_db >= 10) & (~df.feedthrough_limited)].sort_values("gbw", ascending=False)
    print(good.head(10)[cols].to_string(**fmt))

    print("\n===== SAME, but only geometries the C-V nets were actually trained on =====")
    print("    (CGD/CGS saw only (20,20),(40,20),(160,15),(160,20) um -- everything")
    print("     above leans on extrapolated capacitance, so its bandwidth is a guess)")
    tru = df[(~df.cap_extrapolated) & (~df.feedthrough_limited)]
    print("\n-- top 6 by gain --")
    print(tru.sort_values("av_db", ascending=False).head(6)[cols].to_string(**fmt))
    print("\n-- top 6 by GBW --")
    print(tru.sort_values("gbw", ascending=False).head(6)[cols].to_string(**fmt))
    print(f"\nwrote {OUT}/diffpair_sweep.csv")


if __name__ == "__main__":
    main()
