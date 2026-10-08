"""Cross-check src/logic_sim.py against src/tran.py on the same gate.

The two engines were written independently, from the same `.va`, and differ in
almost everything that could hide a mistake:

  integrator     backward Euler (logic_sim)        vs  variable-step BDF2/Gear-2 (tran)
  step control   max node voltage change per step  vs  the same, plus breakpoints
  net forward    va_model.op(), reverse-accumulated  vs  tran._net_eval(), its own
  t=0 state      explicit initial condition        vs  pseudo-transient settle
  shunt          none                              vs  1e-13 S on every node

src/tran.py has already been checked against ngspice-42 -- an independent
solver with its own Gear integrator and LTE control -- to 17 mV peak on a NAND2
transient (outputs/logic_test/ngspice_check.json). So agreement here carries
that validation across to the ring-oscillator results.

The one deliberate difference is left in: logic_sim subtracts ID(VGS, 0) so the
device is odd and continuous at VDS = 0, and blends CGD/CGS through the swap.
Both are sub-nA effects against the ~10 uA switching currents.

    python scripts/crosscheck_engines.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd                                        # noqa: E402

from src.logic_sim import Net, crossings                   # noqa: E402
from src.pseudo_cmos import add_inverter, Sizing           # noqa: E402
from src.tran import Circuit, pulse_fn                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
VDD, NSTAGE = 3.0, 3
PER, TR = 40e-6, 1e-6
TSTOP = 2 * PER


def sizing():
    p = os.path.join(OUT, "recommended_gate.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        return Sizing(w1=r.w1 * 1e-6, w2=r.w2 * 1e-6, w3=r.w3 * 1e-6,
                      w4=r.w4 * 1e-6, cboot=float(r.cboot))
    return Sizing(w1=5e-6, w2=20e-6, w3=160e-6, w4=40e-6, cboot=1e-12)


_VIN, _BPS = pulse_fn(0.0, VDD, PER / 4, TR, TR, PER / 2 - TR, PER)


def vin(t):
    return _VIN(t)


def run_mine(s):
    net = Net()
    for n in ("vdd", "in"):
        net.node(n)
    outs = [f"o{i}" for i in range(NSTAGE)]
    ins = ["in"] + outs[:-1]
    xs = [add_inverter(net, f"s{i}_", s, ins[i], outs[i], "vdd", "gnd", "pCb")
          for i in range(NSTAGE)]
    net.force("vdd", VDD)
    net.force("in", vin(0.0))
    net.build()
    v0 = np.zeros(net.n)
    v0[net._idx["vdd"]] = VDD
    for i, o in enumerate(outs):
        lv = VDD if i % 2 == 0 else 0.0
        v0[net._idx[o]] = lv
        v0[net._idx[xs[i]]] = lv
    # settle at the t=0 input before timing anything
    net.transient(v0, PER / 4, dvmax=0.02, dt_max=PER / 400)
    v, ok = net.solve_dc(guess=v0)
    t, V = net.transient(v, TSTOP, dvmax=0.01, dt_max=PER / 2000,
                         drive=lambda tt, n: n.force("in", vin(tt)),
                         max_steps=400000)
    return t, {o: V[:, net._idx[o]] for o in outs}, {xs[0]: V[:, net._idx[xs[0]]]}


def run_theirs(s):
    c = Circuit()
    outs = [f"o{i}" for i in range(NSTAGE)]
    ins = ["in"] + outs[:-1]
    L = s.l
    for i in range(NSTAGE):
        x = f"x{i}"
        c.add_tft(f"s{i}M1", "vdd", "vdd", x, s.w1, L)
        c.add_tft(f"s{i}M2", x, ins[i], "0", s.w2, L)
        c.add_tft(f"s{i}M3", "vdd", x, outs[i], s.w3, L)
        c.add_tft(f"s{i}M4", outs[i], ins[i], "0", s.w4, L)
        if s.cboot:
            c.add_cap(f"s{i}Cb", outs[i], x, s.cboot)
    c.add_source("vdd", lambda t: VDD)
    c.add_source("in", _VIN, breakpoints=_BPS(TSTOP))
    v0 = {"vdd": VDD}
    for i, o in enumerate(outs):
        v0[o] = VDD if i % 2 == 0 else 0.0
        v0[f"x{i}"] = v0[o]
    r = c.tran(TSTOP, v0=v0, dv_step=0.01, h_max=PER / 2000,
               form="cdv", symmetric=True)
    return r["time"], {o: r[o] for o in outs}, {"x0": r["x0"]}


def main():
    s = sizing()
    print(f"gate: {s.w1*1e6:.0f}/{s.w2*1e6:.0f}/{s.w3*1e6:.0f}/{s.w4*1e6:.0f} um, "
          f"L = {s.l*1e6:.0f} um, Cboot = {s.cboot*1e12:.2f} pF, "
          f"{NSTAGE}-stage chain, VDD = {VDD} V\n")
    ta, A, XA = run_mine(s)
    tb, B, XB = run_theirs(s)
    print(f"logic_sim (backward Euler): {len(ta):6d} points")
    print(f"tran      (BDF2 / Gear-2) : {len(tb):6d} points\n")

    grid = np.linspace(PER / 4, TSTOP, 4000)
    rows = []
    for k in A:
        a = np.interp(grid, ta, A[k])
        b = np.interp(grid, tb, B[k])
        rows.append(dict(node=k, max_abs_mV=np.max(np.abs(a - b)) * 1e3,
                         rms_mV=np.sqrt(np.mean((a - b) ** 2)) * 1e3,
                         swing_V=float(a.max() - a.min())))
    xa = np.interp(grid, ta, list(XA.values())[0])
    xb = np.interp(grid, tb, list(XB.values())[0])
    rows.append(dict(node="x0 (bootstrap node)",
                     max_abs_mV=np.max(np.abs(xa - xb)) * 1e3,
                     rms_mV=np.sqrt(np.mean((xa - xb) ** 2)) * 1e3,
                     swing_V=float(xa.max() - xa.min())))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    lvl = VDD / 2
    print("\n50% crossing times of the last stage [us]:")
    ca = crossings(ta, A[f"o{NSTAGE-1}"], lvl, rising=None)
    cb = crossings(tb, B[f"o{NSTAGE-1}"], lvl, rising=None)
    n = min(len(ca), len(cb))
    for i in range(n):
        print(f"  edge {i}: logic_sim {ca[i]*1e6:9.4f}   tran {cb[i]*1e6:9.4f}   "
              f"delta {(ca[i]-cb[i])*1e9:+8.1f} ns")
    if n:
        d = np.abs(ca[:n] - cb[:n])
        print(f"  max edge disagreement {d.max()*1e9:.1f} ns on a "
              f"{PER*1e6:.0f} us period = {d.max()/PER*100:.3f}%")
    print(f"\nmax over all nodes: {df.max_abs_mV.max():.1f} mV peak, "
          f"{df.rms_mV.max():.1f} mV rms, on swings of ~{df.swing_V.max():.2f} V")
    print(f"(src/tran.py vs ngspice-42 on a NAND2, for scale: 16.7 mV peak, "
          f"1.7 mV rms)")
    df.to_csv(os.path.join(OUT, "engine_crosscheck.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(ta * 1e6, A["o0"], "C0-", lw=2.2, label="logic_sim (backward Euler)")
    ax[0].plot(tb * 1e6, B["o0"], "C3--", lw=1.2, label="tran (BDF2 / Gear-2)")
    ax[0].plot(ta * 1e6, list(XA.values())[0], "C1-", lw=1.6, label="logic_sim $V_X$")
    ax[0].plot(tb * 1e6, list(XB.values())[0], "k--", lw=1.0, label="tran $V_X$")
    ax[0].axhline(VDD, color="k", ls=":", lw=.8)
    ax[0].set(ylabel="V [V]", title="stage 1 output and bootstrap node")
    ax[1].plot(grid * 1e6, (np.interp(grid, ta, A["o0"])
                            - np.interp(grid, tb, B["o0"])) * 1e3, "C0-", lw=1.2,
               label="$V_{OUT}$ difference")
    ax[1].plot(grid * 1e6, (xa - xb) * 1e3, "C1-", lw=1.2, label="$V_X$ difference")
    ax[1].set(xlabel="time [us]", ylabel="difference [mV]",
              title="two independent engines on the same netlist")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "engine_crosscheck.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
