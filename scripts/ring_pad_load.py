"""What a bond pad does to a ring oscillator, and why "VDD / GND / OUT" needs care.

A ring oscillator really is a three-terminal device: it self-starts, needs no
input and no bias, so power-on is the enable. The catch is what "OUT" is wired
to. Tapping a ring node straight to a pad hangs the pad's capacitance on one
stage of the loop, and that capacitance is not small compared with what the
stage normally drives -- so the ring slows down and the frequency you measure is
not the frequency of the circuit you taped out.

Two cases, 11 stages at VDD = 3 V:

  direct    C_pad straight onto the output of stage 0.
  buffered  stage 0 also drives a 1x -> 4x -> 16x fan-up chain of the same
            inverter, and C_pad hangs off the end of that.

    python scripts/ring_pad_load.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic_sim import Net, period_from_crossings      # noqa: E402
from src.pseudo_cmos import Sizing, add_inverter, input_cap   # noqa: E402
from src.va_model import TFTModel                         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
NSTAGE, VDD = 11, 3.0
CPADS = [0.0, 0.5e-12, 1e-12, 2e-12, 5e-12, 10e-12, 20e-12, 50e-12]
BUF = (1, 4, 16)          # fan-up multipliers of the output buffer chain
TPD_REF = 515e-9
COLORS = {"direct": "#2a78d6", "buffered": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"


def gate():
    r = pd.read_csv(os.path.join(OUT, "recommended_gate.csv")).iloc[0]
    return Sizing(w1=r.w1 * 1e-6, w2=r.w2 * 1e-6, w3=r.w3 * 1e-6,
                  w4=r.w4 * 1e-6, cboot=float(r.cboot))


def build(s, cpad, buffered):
    net = Net()
    net.node("vdd")
    outs = [f"o{i}" for i in range(NSTAGE)]
    for nm in outs:
        net.node(nm)
    ins = outs[-1:] + outs[:-1]
    xs = [add_inverter(net, f"s{i}_", s, ins[i], outs[i], "pCb" and "vdd", "gnd",
                       "pCb") for i in range(NSTAGE)]
    pad = outs[0]
    if buffered:
        src = outs[0]
        for k, m in enumerate(BUF):
            dst = f"b{k}"
            net.node(dst)
            add_inverter(net, f"b{k}_", s, src, dst, "vdd", "gnd", "pCb", mult=m)
            src = dst
        pad = src
    if cpad > 0:
        net.cap(pad, "gnd", cpad)
    net.force("vdd", VDD)
    net.build()
    v0 = np.zeros(net.n)
    v0[net._idx["vdd"]] = VDD
    for i, o in enumerate(outs):
        lv = VDD if i % 2 == 0 else 0.0
        v0[net._idx[o]] = lv
        v0[net._idx[xs[i]]] = lv
    return net, outs, pad, v0


def _job(a):
    s, cpad, buffered = a
    net, outs, pad, v0 = build(s, cpad, buffered)
    T = 2.0 * NSTAGE * TPD_REF
    t_end = 14.0 * T * (1.0 + cpad / 2e-12)        # slower rings need more window
    t, V = net.transient(v0, t_end, dvmax=0.04, dt_max=T / 400.0,
                         max_steps=600000)
    d = dict(cpad=cpad * 1e12, buffered=buffered, steps=net.stats["steps"],
             complete=net.stats["complete"])
    # measure on a ring node that is NOT the tapped one
    probe = V[:, net._idx[outs[NSTAGE // 2]]]
    pm = period_from_crossings(t, probe, t_start=0.55 * t[-1])
    if pm is None:
        d.update(freq=np.nan, swing=np.nan, pad_swing=np.nan, oscillates=False)
        return d
    m = t >= 0.55 * t[-1]
    pv = V[m, net._idx[pad]]
    d.update(freq=pm["freq"], swing=pm["swing"], ring_swing_pct=100 * pm["swing"] / VDD,
             pad_swing=float(pv.max() - pv.min()),
             pad_swing_pct=100 * float(pv.max() - pv.min()) / VDD,
             oscillates=True)
    iv = net._idx["vdd"]
    sel = np.where(m)[0][::4]
    cur = [float(np.sum(net.devices(V[k])["id"][net.nd == iv])
                 - np.sum(net.devices(V[k])["id"][net.ns == iv])) for k in sel]
    d["power"] = VDD * float(np.mean(cur))
    return d


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = gate()
    m = TFTModel()
    cin = input_cap(s, "pCb", VDD * 0.5)
    o = m.op(np.array([VDD * .5]), np.array([VDD * .5]), s.w3, s.l)
    cself = float(o["cgd"][0]) + float(
        m.op(np.array([VDD * .5]), np.array([VDD * .5]), s.w4, s.l)["cgd"][0])
    print(f"ring node capacitance, for scale: the next stage's input is "
          f"{cin*1e12:.2f} pF,\n  plus {cself*1e12:.2f} pF of self-load and the "
          f"{s.cboot*1e12:.1f} pF bootstrap cap -> about "
          f"{(cin+cself+s.cboot)*1e12:.1f} pF per node.")
    print(f"A bond pad is typically 0.5-5 pF; a scope probe adds ~10 pF and a "
          f"cable ~100 pF.\n")

    jobs = [(s, c, b) for b in (False, True) for c in CPADS]
    print(f"{len(jobs)} transients: {NSTAGE} stages at VDD = {VDD} V, "
          f"C_pad = {[c*1e12 for c in CPADS]} pF, direct vs "
          f"{'x'.join(str(b) for b in BUF)} buffer")
    with Pool(os.cpu_count()) as pool:
        df = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    df.to_csv(os.path.join(OUT, "ring_pad_load.csv"), index=False)

    f0 = float(df[(~df.buffered) & (df.cpad == 0)].freq.iloc[0])
    df["f_rel"] = 100 * df.freq / f0
    for b, lbl in ((False, "direct tap"), (True, f"buffered {'x'.join(map(str,BUF))}")):
        d = df[df.buffered == b].sort_values("cpad")
        print(f"\n--- {lbl} ---")
        t = pd.DataFrame({"C_pad [pF]": d.cpad, "f [kHz]": d.freq / 1e3,
                          "f / f_unloaded [%]": d.f_rel,
                          "ring swing [% VDD]": d.ring_swing_pct,
                          "pad swing [% VDD]": d.pad_swing_pct,
                          "P [uW]": d.power * 1e6})
        print(t.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for b, lbl in ((False, "direct tap"), (True, f"buffered ({'x'.join(map(str,BUF))})")):
        d = df[(df.buffered == b) & df.oscillates].sort_values("cpad")
        c = COLORS["buffered" if b else "direct"]
        ax[0].plot(d.cpad, d.f_rel, "-o", color=c, lw=2, ms=6, mec=SURFACE, mew=1.2)
        ax[1].plot(d.cpad, d.pad_swing_pct, "-o", color=c, lw=2, ms=6,
                   mec=SURFACE, mew=1.2)
        ax[0].annotate(lbl, xy=(d.cpad.iloc[-1], d.f_rel.iloc[-1]),
                       xytext=(-6, 10 if b else -14), textcoords="offset points",
                       color=INK, fontsize=9, ha="right", fontweight="medium")
        ax[1].annotate(lbl, xy=(d.cpad.iloc[-1], d.pad_swing_pct.iloc[-1]),
                       xytext=(-6, 10 if b else -14), textcoords="offset points",
                       color=INK, fontsize=9, ha="right", fontweight="medium")
    ax[0].axhline(100, color=INK2, ls=(0, (4, 3)), lw=1)
    ax[0].text(49, 101.5, "unloaded ring", color=INK2, fontsize=8, ha="right")
    for a, xl, yl, ti in ((ax[0], "pad + probe capacitance on OUT [pF]",
                           "ring frequency [% of unloaded]",
                           "what the pad does to the ring"),
                          (ax[1], "pad + probe capacitance on OUT [pF]",
                           "signal at the pad [% of VDD]",
                           "what reaches the pad")):
        a.set_facecolor(SURFACE)
        a.set_xlabel(xl, color=INK2, fontsize=9)
        a.set_ylabel(yl, color=INK2, fontsize=9)
        a.set_title(ti, color=INK, fontsize=10.5, pad=8)
        a.grid(True, color=GRID, lw=.7)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=8.5, length=3)
        a.set_xlim(-2, 52)
    h = [plt.Line2D([], [], color=COLORS[k], lw=2.4, marker="o", ms=6,
                    mec=SURFACE, mew=1.2,
                    label="OUT wired straight to the pad" if k == "direct"
                    else f"OUT through a {'x'.join(map(str,BUF))} buffer")
         for k in ("direct", "buffered")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=9.5, labelcolor=INK)
    fig.suptitle(f"Why a ring oscillator's third pin needs a buffer "
                 f"({NSTAGE} stages, VDD = {VDD} V)", color=INK, fontsize=12.5,
                 y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(OUT, "ring_pad_load.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nwrote {p}")
