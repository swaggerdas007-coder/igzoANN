"""Two-stage a-IGZO differential amplifier -- getting past the ~25 dB
single-stage ceiling to 40 dB.

A resistor-loaded stage has Av = (gm/ID) * V_drop. a-IGZO's gm/ID tops out
near 8 /V and V_drop can't exceed ~3 V, so one stage cannot beat ~25 dB no
matter what W/L you pick. Two identical-topology stages in series can.

Nodes: 1 o1a  2 o2a  3 tailA   4 o1b  5 o2b  6 tailB
       7 in1  8 in2  9 vdd    10 vbA 11 vbB
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.circuit import Circuit, model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")

O1A, O2A, TA, O1B, O2B, TB, IN1, IN2, VDD_N, VBA, VBB = range(1, 12)
VDD, CLOAD = 5.0, 1e-12
RMAX = 10e6


def build(w1, l1, ra, w2, l2, rb, vcm, vba, vbb, cload=CLOAD):
    c = Circuit(11)
    c.fixed = {IN1: vcm, IN2: vcm, VDD_N: VDD, VBA: vba, VBB: vbb}
    c.add_tft("M1", O1A, IN1, TA, w1, l1)
    c.add_tft("M2", O2A, IN2, TA, w1, l1)
    c.add_tft("M3", TA, VBA, 0, 40e-6, 20e-6)
    c.add_res(VDD_N, O1A, ra)
    c.add_res(VDD_N, O2A, ra)
    # stage B input is stage A's output; cross the pair so the two inversions
    # do not leave the overall amplifier inverting
    c.add_tft("M4", O1B, O2A, TB, w2, l2)
    c.add_tft("M5", O2B, O1A, TB, w2, l2)
    c.add_tft("M6", TB, VBB, 0, 40e-6, 20e-6)
    c.add_res(VDD_N, O1B, rb)
    c.add_res(VDD_N, O2B, rb)
    c.add_cap(O1B, 0, cload)
    c.add_cap(O2B, 0, cload)
    return c


def solve(w1, l1, w2, l2, vcm, vba, vbb, drop_a, drop_b):
    """Fixed-point on both load resistors to hit the two target output swings."""
    ra, rb = 1e6, 1e6
    for _ in range(12):
        c = build(w1, l1, ra, w2, l2, rb, vcm, vba, vbb)
        v, ok = c.solve_dc(guess={O1A: VDD - drop_a, O2A: VDD - drop_a, TA: 0.5,
                                  O1B: VDD - drop_b, O2B: VDD - drop_b, TB: 1.5})
        if not ok:
            return None
        ia = float(np.ravel(model().id_(*c.tfts[0].bias(v), w1, l1))[0])
        ib = float(np.ravel(model().id_(*c.tfts[3].bias(v), w2, l2))[0])
        if ia < 1e-10 or ib < 1e-10:
            return None
        ra_n, rb_n = drop_a / ia, drop_b / ib
        if abs(ra_n - ra) / ra < 1e-5 and abs(rb_n - rb) / rb < 1e-5:
            ra, rb = ra_n, rb_n
            break
        ra, rb = 0.5 * ra + 0.5 * ra_n, 0.5 * rb + 0.5 * rb_n
    if not (1e3 < ra < RMAX and 1e3 < rb < RMAX):
        return None
    c = build(w1, l1, ra, w2, l2, rb, vcm, vba, vbb)
    v, ok = c.solve_dc(guess={O1A: VDD - drop_a, O2A: VDD - drop_a, TA: 0.5,
                              O1B: VDD - drop_b, O2B: VDD - drop_b, TB: 1.5})
    if not ok:
        return None
    for t in c.tfts:
        vgs, vds = t.bias(v)
        o = t.op(v)
        if not (-5 <= vgs <= 5 and 0.05 <= vds <= 5) or o["clamped_vg"] or o["clamped_vd"]:
            return None
        if t.name in ("M1", "M2", "M4", "M5") and (o["gds"] <= 0 or o["gm"] / o["gds"] < 2):
            return None

    f = np.logspace(0, 8, 500)
    vac = c.ac(v, f, {IN1: 0.5, IN2: -0.5, VDD_N: 0.0, VBA: 0.0, VBB: 0.0})
    a_tot = vac[:, O1B] - vac[:, O2B]
    a_stg1 = abs(vac[0, O1A] - vac[0, O2A])
    a0 = abs(a_tot[0])
    mag = np.abs(a_tot) / a0
    below = np.where(mag < 1 / np.sqrt(2))[0]
    f3 = float(np.interp(1 / np.sqrt(2), [mag[below[0]], mag[below[0] - 1]],
                         [f[below[0]], f[below[0] - 1]])) if len(below) else np.inf
    itot = float(np.ravel(model().id_(*c.tfts[2].bias(v), 40e-6, 20e-6))[0]) + \
        float(np.ravel(model().id_(*c.tfts[5].bias(v), 40e-6, 20e-6))[0])
    return dict(c=c, v=v, ra=ra, rb=rb, av=a0, av_db=20 * np.log10(a0),
                av1_db=20 * np.log10(a_stg1), av2_db=20 * np.log10(a0 / a_stg1),
                f3db=f3, power=itot * VDD, f=f, resp=a_tot,
                vo_a=v[O1A], vo_b=v[O1B])


def main():
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Brute-forcing both stages is wasteful: stage A's optimum is already
    # known from the single-stage sweep, and it is stage B -- whose input
    # common mode is pinned at stage A's output -- that limits the total.
    # So fix a few stage-A settings and search stage B properly.
    STAGE_A = [((160e-6, 15e-6), 0.8, 1.2, 2.5),    # out CM 2.5 V, 22.8 dB
               ((160e-6, 15e-6), 0.8, 1.2, 3.0),    # out CM 2.0 V, more room for B
               ((160e-6, 20e-6), 0.8, 1.2, 2.8)]
    best, rows = None, []
    for (w1, l1), vcm, vba, da in STAGE_A:
        for (w2, l2), vbb, db in itertools.product(
                [(160e-6, 15e-6), (160e-6, 20e-6), (80e-6, 20e-6)],
                [1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                [2.0, 2.5, 2.8, 3.0, 3.2]):
            try:
                r = solve(w1, l1, w2, l2, vcm, vba, vbb, da, db)
            except Exception:
                continue
            if not r:
                continue
            r.update(vcm=vcm, vba=vba, vbb=vbb, da=da, db=db,
                     W=w1 * 1e6, L=l1 * 1e6, W2=w2 * 1e6, L2=l2 * 1e6)
            rows.append({k: v for k, v in r.items()
                         if k not in ("c", "v", "f", "resp")})
            if best is None or r["av_db"] > best["av_db"]:
                best = r
                print(f"  best gain so far {r['av_db']:5.1f} dB "
                      f"(A {r['av1_db']:.1f} + B {r['av2_db']:.1f}), "
                      f"f3dB {r['f3db']/1e3:.1f} kHz", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("av_db", ascending=False)
    df.to_csv(os.path.join(OUT, "twostage_sweep.csv"), index=False)
    cols = ["W", "L", "W2", "L2", "vcm", "vba", "vbb", "da", "db", "ra", "rb",
            "av_db", "av1_db", "av2_db", "f3db", "power"]
    print(f"\n{len(df)} valid two-stage points\n")
    print("=== highest gain ===")
    print(df.head(5)[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    at40 = df[df.av_db >= 40].sort_values("f3db", ascending=False)
    print(f"\n=== >= 40 dB, fastest first ({len(at40)} points) ===")
    print(at40.head(8)[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    b = best
    print(f"===== two-stage: pair A {b['W']:.0f}/{b['L']:.0f} um, "
          f"pair B {b['W2']:.0f}/{b['L2']:.0f} um =====")
    print(f"  V_CM,in = {b['vcm']} V,  V_B(A) = {b['vba']} V,  V_B(B) = {b['vbb']} V")
    print(f"  R_A = {b['ra']/1e6:.2f} MOhm  (stage A out CM {b['vo_a']:.2f} V)")
    print(f"  R_B = {b['rb']/1e6:.2f} MOhm  (stage B out CM {b['vo_b']:.2f} V)")
    print(f"  stage A {b['av1_db']:.1f} dB  +  stage B {b['av2_db']:.1f} dB")
    print(f"  TOTAL Av = {b['av']:.0f} ({b['av_db']:.1f} dB), f-3dB = {b['f3db']/1e3:.1f} kHz, "
          f"GBW = {b['av']*b['f3db']/1e6:.2f} MHz")
    print(f"  power = {b['power']*1e6:.1f} uW")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(b["f"], 20 * np.log10(np.abs(b["resp"])), lw=2, label="two-stage")
    ax.axhline(b["av_db"] - 3, ls=":", c="k", lw=1)
    ax.axhline(40, ls="--", c="r", lw=1, label="40 dB target")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("|Ad| [dB]"); ax.grid(alpha=.3)
    ax.set_title(f"two-stage a-IGZO diff amp: {b['av_db']:.1f} dB, "
                 f"f-3dB {b['f3db']/1e3:.0f} kHz, {b['power']*1e6:.0f} uW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twostage.png"), dpi=130)
    print(f"wrote {OUT}/twostage.png")


if __name__ == "__main__":
    main()
