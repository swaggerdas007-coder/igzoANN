"""Size the bootstrapped pseudo-CMOS NAND2 at the target operating point.

Bench: NAND2 under test driving fan-out 3 (three NAND2 inputs -- the heaviest
node inside the D flip-flop), A toggling at f, B at f/2 so all four input
combinations and both stack positions are exercised. For every sizing we
record VOH (worst high level), VOL, propagation delays measured from the
input edge that caused each output edge, and supply power.

    python scripts/logic_sizing.py            # -> outputs/logic_test/sizing_sweep.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic import Netlist                      # noqa: E402
from src.logic_meas import crossings, avg_power    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "logic_test")
um = 1e-6


def bench(sz, vdd=5.0, f=50e3, fo=3, ncyc=3, form="cdv"):
    n = Netlist(sz)
    n.vdc("vdd", vdd)
    n.vpulse("a", 0, vdd, 1 / f, delay=0.25 / f)
    n.vpulse("b", 0, vdd, 2 / f, delay=0.25 / f)
    n.nand("dut", ["a", "b"], "y")
    for k in range(fo):
        n.nand(f"l{k}", ["y", "vdd"], f"z{k}")
    r = n.simulate(ncyc / f, form=form)
    return r


def measure(r, vdd, f):
    t, y = r["time"], r["y"]
    mid = vdd / 2
    ins = np.sort(np.concatenate([crossings(t, r[k], mid, d) for k in ("a", "b") for d in (1, -1)]))
    res = {}
    for name, d in (("tpHL", -1), ("tpLH", +1)):
        ds = []
        for to in crossings(t, y, mid, d):
            prev = ins[ins < to]
            if len(prev):
                ds.append(to - prev[-1])
        res[name] = max(ds) if ds else np.nan
    # levels just before each input event (settled state of that half-cycle)
    T = 1 / f
    ts = np.arange(0.25 * T + 0.5 * T - 0.02 * T, t[-1], 0.5 * T)
    a = np.interp(ts, t, r["a"]) > mid
    b = np.interp(ts, t, r["b"]) > mid
    yv = np.interp(ts, t, y)
    hi = yv[~(a & b)]
    lo = yv[a & b]
    res["VOH"] = hi.min() if len(hi) else np.nan
    res["VOL"] = lo.max() if len(lo) else np.nan
    res["P_avg_uW"] = avg_power(t, r["i(vdd)"], vdd, T, t[-1]) * 1e6
    # functional: output must be the NAND at every sample
    res["ok"] = bool(np.all((yv > mid) == ~(a & b)))
    return res


def one(args):
    m1, m2, m3, m4, cb = args
    sz = dict(m1=(m1[0] * um, m1[1] * um), m2=(m2[0] * um, m2[1] * um),
              m3=(m3[0] * um, m3[1] * um), m4=(m4[0] * um, m4[1] * um), cb=cb * 1e-12)
    try:
        r = bench(sz)
        res = measure(r, 5.0, 50e3)
    except Exception as e:                       # noqa: BLE001
        res = dict(ok=False, err=str(e)[:80])
    return dict(m1=f"{m1[0]}/{m1[1]}", m2=f"{m2[0]}/{m2[1]}", m3=f"{m3[0]}/{m3[1]}",
                m4=f"{m4[0]}/{m4[1]}", cb_pF=cb, **res)


def main():
    os.makedirs(OUT, exist_ok=True)
    grid = list(itertools.product(
        [(20, 5), (40, 5), (80, 5)],                     # m1 stage-1 stack
        [(5, 20), (5, 10), (5, 5), (10, 5)],             # m2 diode load
        [(40, 5), (80, 5), (160, 5)],                    # m3 output stack
        [(40, 5), (80, 5), (160, 5)],                    # m4 pull-up
        [0.0, 0.5, 1.0, 2.0],                            # Cb (pF)
    ))
    # the winners sat on the small edge of the m1/m3/m4 grid: extend it down
    grid += [g for g in itertools.product(
        [(10, 5), (20, 5)], [(5, 20)], [(20, 5), (40, 5)], [(20, 5), (40, 5)], [1.0, 2.0])
        if g not in grid]
    print(len(grid), "sizings")
    rows = []
    with Pool(4) as p:
        for k, row in enumerate(p.imap_unordered(one, grid)):
            rows.append(row)
            if k % 25 == 0:
                print(k, row, flush=True)
    df = pd.DataFrame(rows)
    df["tp_max_us"] = df[["tpHL", "tpLH"]].max(axis=1) * 1e6
    df.to_csv(os.path.join(OUT, "sizing_sweep.csv"), index=False)
    good = df[df.ok & (df.VOH > 4.0) & (df.VOL < 0.5)].sort_values("tp_max_us")
    print(good.head(20).to_string())


if __name__ == "__main__":
    main()
