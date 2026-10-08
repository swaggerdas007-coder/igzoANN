"""Operating-window (shmoo) analysis of the 3-bit ripple counter: VDD x clock
frequency, same sizing as the 5 V / 50 kHz design.

A point passes when the decoded count, sampled just before every rising clock
edge, runs 1,2,...,7,0,1 and every sampled bit is a clean logic level
(high >= 0.7 VDD, low <= 0.3 VDD).

    python scripts/logic_shmoo.py        # -> outputs/logic_test/shmoo.csv, shmoo.png
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logic_circuits import (counter_bench, counter_measure, OUT, C, SERIES,   # noqa: E402
                            style)

VDDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
FREQS = [10e3, 20e3, 50e3, 100e3, 150e3, 200e3, 250e3, 300e3, 400e3, 500e3]
# extra points that pin down the pass/fail edge found on the coarse grid
REFINE = ([(2.5, f) for f in (30e3, 40e3)] +
          [(v, f) for v in (3.0, 3.5, 4.0) for f in (60e3, 70e3, 80e3, 90e3)] +
          [(v, f) for v in (4.5, 5.0) for f in (110e3, 120e3, 130e3, 140e3)])


def one(args):
    vdd, f = args
    row = dict(vdd=vdd, f_kHz=f / 1e3)
    try:
        r = counter_bench(vdd, f)
        m = counter_measure(r, vdd, f)
        clean = (m["VOH"] is not None and m["VOL"] is not None and
                 m["VOH"] >= 0.7 * vdd and m["VOL"] <= 0.3 * vdd)
        row.update(functional=m["functional"], clean_levels=bool(clean),
                   pass_=bool(m["functional"] and clean), VOH=m["VOH"], VOL=m["VOL"],
                   clk_to_q2_us=m["clk_to_q2_us"], P_avg_uW=m["P_avg_uW"],
                   E_per_clock_nJ=m["E_per_clock_nJ"], count=" ".join(map(str, m["count"])))
    except Exception as e:                       # noqa: BLE001
        row.update(functional=False, clean_levels=False, pass_=False, error=str(e)[:80])
    print(row, flush=True)
    return row


def plot(df, path):
    style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    coarse = df[df.f_kHz.isin([f / 1e3 for f in FREQS])]
    fi = {f: k for k, f in enumerate(sorted(coarse.f_kHz.unique()))}
    vi = {v: k for k, v in enumerate(sorted(coarse.vdd.unique()))}
    for _, r in coarse.iterrows():
        x, y = fi[r.f_kHz], vi[r.vdd]
        if r.pass_:
            col, txt = C["aqua"], "pass"
        elif r.functional:
            col, txt = C["yellow"], "levels"
        else:
            col, txt = C["red"], "fail"
        ax.add_patch(plt.Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, color=col, lw=0))
        ax.text(x, y, txt, ha="center", va="center", fontsize=7, color="#ffffff" if txt != "levels" else C["ink"])
    ax.plot(fi[50.0], vi[5.0], marker="s", ms=26, mfc="none", mec=C["ink"], mew=1.6)
    ax.set_xticks(list(fi.values()), [f"{f:g}" for f in fi])
    ax.set_yticks(list(vi.values()), [f"{v:g}" for v in vi])
    ax.set_xlim(-0.6, len(fi) - 0.4)
    ax.set_ylim(-0.6, len(vi) - 0.4)
    ax.grid(False)
    ax.set_xlabel("clock frequency (kHz)")
    ax.set_ylabel("VDD (V)")
    ax.set_title("3-bit counter shmoo (box = requested 5 V / 50 kHz)")
    ax.text(0, -0.2, "pass: counts 0-7 with clean levels  |  levels: counts, but a bit outside 0.3/0.7 VDD"
            "  |  fail: wrong count", transform=ax.transAxes, fontsize=7, color=C["ink2"])

    ax = axs[1]
    fmax = df[df.pass_].groupby("vdd").f_kHz.max().reindex(sorted(df.vdd.unique()))
    ax.plot(fmax.index, fmax.values, marker="o", color=C["blue"], ms=6)
    for v, fm in fmax.items():
        if np.isfinite(fm):
            ax.annotate(f"{fm:g}", (v, fm), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=C["ink2"])
    ax.axhline(50, color=C["ink2"], lw=0.8, ls=":")
    ax.text(fmax.index.max(), 50, "50 kHz target ", va="top", ha="right", fontsize=8, color=C["ink2"])
    ax.set_xlabel("VDD (V)")
    ax.set_ylabel("highest passing clock (kHz)")
    ax.set_title("max clock frequency vs supply (10 kHz resolution)")

    ax = axs[2]
    for k, f in enumerate([20.0, 50.0, 100.0]):
        s = df[(df.f_kHz == f) & df.pass_].sort_values("vdd")
        ax.plot(s.vdd, s.E_per_clock_nJ, marker="o", ms=5, color=SERIES[k], label=f"{f:g} kHz")
    ax.set_xlabel("VDD (V)")
    ax.set_ylabel("energy per clock (nJ)")
    ax.set_title("energy per clock cycle (passing points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    csv = os.path.join(OUT, "shmoo.csv")
    old = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame(columns=["vdd", "f_kHz"])
    done = {(round(v, 3), round(f, 3)) for v, f in zip(old.vdd, old.f_kHz)}
    grid = [(v, f) for v, f in list(itertools.product(VDDS, FREQS)) + REFINE
            if (round(v, 3), round(f / 1e3, 3)) not in done]
    with Pool(4) as p:
        rows = p.map(one, grid)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df = df.sort_values(["vdd", "f_kHz"]).reset_index(drop=True)
    df["pass_"] = df["pass_"].astype(bool)
    df["functional"] = df["functional"].astype(bool)
    df.to_csv(csv, index=False)
    plot(df, os.path.join(OUT, "shmoo.png"))
    print(df.pivot(index="vdd", columns="f_kHz", values="pass_").to_string())
    print(df[df.pass_].groupby("vdd").f_kHz.max())


if __name__ == "__main__":
    main()
