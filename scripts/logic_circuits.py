"""Simulate the bootstrapped pseudo-CMOS NAND2/NAND3, the 6-NAND D flip-flop
and the 3-bit ripple counter at one operating point, measure them and plot.

    python scripts/logic_circuits.py                 # 5 V, 50 kHz
    python scripts/logic_circuits.py --vdd 4 --f 20e3 --tag 4V_20k

Model: verilogA/ntft_full.va nets in the capacitor-current form
(verilogA/ntft_full_cdv.va), solved by src/tran.py. Outputs land in
outputs/logic_test/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.logic import Netlist, dff_nodeset, nand_nodeset, SIZING   # noqa: E402
from src.logic_meas import crossings, transition, avg_power            # noqa: E402

OUT = os.path.join(ROOT, "outputs", "logic_test")

# reference palette (dataviz skill), fixed slot order
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"]]


def style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "lines.linewidth": 1.6, "font.size": 9, "legend.frameon": False,
        "axes.titlesize": 10, "axes.titleweight": "bold",
    })


# ------------------------------------------------------------------ NAND
def nand_bench(n, vdd, f, fo=3):
    """Truth-table walk: input k toggles at f/2^k; FO3 NAND2 load."""
    T = 1 / f
    ins = ["a", "b", "c"][:n]
    net = Netlist()
    net.vdc("vdd", vdd)
    for k, a in enumerate(ins):
        net.vpulse(a, 0, vdd, T * 2 ** k, delay=0.25 * T)
    net.nand("dut", ins, "y")
    for k in range(fo):
        net.nand(f"l{k}", ["y", "vdd"], f"z{k}")
    tstop = 0.25 * T + T * 2 ** (n - 1) + 0.25 * T
    t0 = time.time()
    r = net.simulate(tstop)
    r["cpu_s"] = time.time() - t0
    r["ins"], r["ntft"] = ins, net.ntft
    return r


def nand_measure(r, vdd, f):
    T = 1 / f
    t, y = r["time"], r["y"]
    mid = vdd / 2
    edges = np.sort(np.concatenate([crossings(t, r[a], mid, d) for a in r["ins"] for d in (1, -1)]))
    m = {}
    for name, d in (("tpHL_us", -1), ("tpLH_us", +1)):
        ds = [to - edges[edges < to][-1] for to in crossings(t, y, mid, d) if np.any(edges < to)]
        m[name] = float(np.max(ds) * 1e6) if ds else None
    ts = np.arange(0.25 * T + 0.48 * T, t[-1], 0.5 * T)
    allhi = np.all([np.interp(ts, t, r[a]) > mid for a in r["ins"]], axis=0)
    yv = np.interp(ts, t, y)
    m["VOH"] = float(yv[~allhi].min())
    m["VOL"] = float(yv[allhi].max())
    m["functional"] = bool(np.all((yv > mid) == ~allhi))
    m["t_rise_us"] = float(np.max(transition(t, y, m["VOL"], m["VOH"], +1)) * 1e6)
    m["t_fall_us"] = float(np.max(transition(t, y, m["VOL"], m["VOH"], -1)) * 1e6)
    m["P_bench_uW"] = float(avg_power(t, r["i(vdd)"], vdd, 0.25 * T, t[-1]) * 1e6)
    m["X_max"] = float(r["dut_x"].max())
    return m


def nand_vtc(n, vdd, npts=101):
    """DC transfer with the swept input at the top or bottom of the stack."""
    out = {}
    for pos in ("top", "bottom"):
        net = Netlist()
        net.vdc("vdd", vdd)
        ins = [f"i{k}" for k in range(n)]
        sw = ins[0] if pos == "top" else ins[-1]
        for a in ins:
            net.vdc(a, vdd if a != sw else 0.0)
        net.nand("dut", ins, "y")
        for k in range(3):
            net.nand(f"l{k}", ["y", "vdd"], f"z{k}")
        c = net.circuit()
        vin = np.linspace(0, vdd, npts)
        r = c.dc_sweep(sw, vin, v0={"y": vdd, "dut_x": vdd})
        out[pos] = (vin, r["y"], r["i(vdd)"])
    return out


def vtc_metrics(vin, vout):
    """VM, peak gain and maximum-equal-criterion (largest square) noise margins.

    The unity-gain VIL/VIH definition does not work here: the L = 5 um devices
    already conduct ~10 nA at VGS = 0, so |dVout/dVin| > 1 at Vin = 0. The
    largest square inscribed between the VTC and its mirror (Hauser 1993) is the
    usual noise-margin figure for TFT logic in that case. For a gate paired
    with its own mirror the two lobes are reflections of each other, so it is
    one number (NM_H = NM_L).
    """
    g = np.gradient(vout, vin)
    f = lambda x: np.interp(x, vin, vout)                         # noqa: E731
    asc = np.argsort(vout)
    h = lambda x: np.interp(x, vout[asc], vin[asc], left=vin[-1], right=vin[0])  # noqa: E731
    vm = float(vin[np.argmin(np.abs(vout - vin))])

    def lobe(upper, lower, xs):
        best = 0.0
        for x in xs:
            y = lower(x)
            lo, hi = 0.0, vin[-1]
            if upper(x) <= y:
                continue
            for _ in range(40):                # largest s with y + s <= upper(x + s)
                s = (lo + hi) / 2
                lo, hi = (s, hi) if y + s <= upper(x + s) else (lo, s)
            best = max(best, lo)
        return best

    xs = np.linspace(vin[0], vin[-1], 400)
    nm = lobe(f, h, xs[xs <= vm])
    return dict(VOH=float(vout.max()), VOL=float(vout.min()), VM=vm,
                NM_mec=float(nm), gain=float(-g.min()))


# ------------------------------------------------------------------ DFF
def dff_bench(vdd, f, ncyc=7):
    T = 1 / f
    net = Netlist()
    net.vdc("vdd", vdd)
    net.vpulse("clk", 0, vdd, T, delay=0.5 * T)
    # D period 3T: its edges alternate between the clock-low and clock-high
    # halves, so the run also shows D changes while CLK is high are ignored
    net.vpulse("d", 0, vdd, 3 * T, delay=0.25 * T)
    net.dff("ff", "d", "clk", "q", "qb")
    net.nand("lq", ["q", "vdd"], "zq")
    net.nand("lqb", ["qb", "vdd"], "zqb")
    t0 = time.time()
    r = net.simulate(ncyc * T, nodeset=dff_nodeset("ff", 0, 0, "q", "qb", vdd))
    r["cpu_s"] = time.time() - t0
    r["ntft"] = net.ntft
    return r


def dff_measure(r, vdd, f):
    T = 1 / f
    t = r["time"]
    mid = vdd / 2
    ck = crossings(t, r["clk"], mid, +1)
    exp, got, cq_r, cq_f = [], [], [], []
    q = 0
    for te in ck:
        q = int(np.interp(te - 0.02 * T, t, r["d"]) > mid)
        exp.append(q)
        got.append(int(np.interp(te + 0.45 * T, t, r["q"]) > mid))
        for name, lst, d in (("q", cq_r if q else cq_f, +1 if q else -1),):
            c = crossings(t, r[name], mid, d)
            c = c[(c > te) & (c < te + T)]
            if len(c):
                lst.append(c[0] - te)
    qs = np.array([np.interp(te + 0.45 * T, t, r["q"]) for te in ck])
    qbs = np.array([np.interp(te + 0.45 * T, t, r["qb"]) for te in ck])
    ex = np.array(exp, bool)
    highs = np.concatenate([qs[ex], qbs[~ex]])
    lows = np.concatenate([qs[~ex], qbs[ex]])
    return dict(functional=bool(exp == got), expected=exp, got=got,
                clk_to_q_rise_us=float(np.max(cq_r) * 1e6) if cq_r else None,
                clk_to_q_fall_us=float(np.max(cq_f) * 1e6) if cq_f else None,
                VOH=float(highs.min()), VOL=float(lows.max()),
                P_bench_uW=float(avg_power(t, r["i(vdd)"], vdd, T, t[-1]) * 1e6))


# ------------------------------------------------------------------ counter
def counter_netlist(vdd, f, sizing=None):
    T = 1 / f
    net = Netlist(sizing)
    net.vdc("vdd", vdd)
    net.vpulse("clk", 0, vdd, T, delay=0.5 * T)
    ns = {}
    clk = "clk"
    for k in range(3):
        q, qb = f"q{k}", f"qb{k}"
        net.dff(f"ff{k}", qb, clk, q, qb)          # toggle: D = Qbar
        # power-on state 000: FF0's clock is low, FF1/FF2 are clocked by a high Qbar
        ns.update(dff_nodeset(f"ff{k}", 1, 0, q, qb, vdd, clk=int(k > 0)))
        net.nand(f"lq{k}", [q, "vdd"], f"zq{k}")   # an observer load on each Q
        ns.update(nand_nodeset(f"lq{k}", [0, 1], 1, vdd))
        clk = qb                                   # ripple: next stage clocked by Qbar
    return net, ns


def counter_bench(vdd, f, ncyc=10, sizing=None, **kw):
    T = 1 / f
    net, ns = counter_netlist(vdd, f, sizing)
    t0 = time.time()
    r = net.simulate(ncyc * T, nodeset=ns, **kw)
    r["cpu_s"] = time.time() - t0
    r["ntft"] = net.ntft
    return r


def counter_measure(r, vdd, f):
    T = 1 / f
    t = r["time"]
    mid = vdd / 2
    ck = crossings(t, r["clk"], mid, +1)
    # sample just before the next rising edge: count after edge k = k+1 (mod 8)
    ts = np.append(ck[1:], ck[-1] + T) - 0.03 * T
    ts = ts[ts < t[-1]]
    bits = np.array([np.interp(ts, t, r[f"q{k}"]) for k in range(3)])
    got = ((bits > mid).astype(int) * np.array([[1], [2], [4]])).sum(axis=0)
    exp = (np.arange(len(ts)) + 1) % 8
    hi = bits[bits > mid]
    lo = bits[bits <= mid]
    # ripple delay: rising CLK edge -> each Q crossing that it causes
    dly = {k: [] for k in range(3)}
    for te in ck:
        for k in range(3):
            c = np.sort(np.concatenate([crossings(t, r[f"q{k}"], mid, d) for d in (1, -1)]))
            c = c[(c > te) & (c < te + T)]
            if len(c):
                dly[k].append(c[0] - te)
    P = avg_power(t, r["i(vdd)"], vdd, T, t[-1])
    return dict(functional=bool(np.array_equal(got, exp)), count=got.tolist(), expected=exp.tolist(),
                VOH=float(hi.min()) if len(hi) else None, VOL=float(lo.max()) if len(lo) else None,
                clk_to_q0_us=float(np.max(dly[0]) * 1e6) if dly[0] else None,
                clk_to_q1_us=float(np.max(dly[1]) * 1e6) if dly[1] else None,
                clk_to_q2_us=float(np.max(dly[2]) * 1e6) if dly[2] else None,
                P_avg_uW=float(P * 1e6), E_per_clock_nJ=float(P / f * 1e9))


# ------------------------------------------------------------------ plots
def _wave(ax, t, y, label, color, us=True):
    ax.plot(t * (1e6 if us else 1), y, color=color, label=label)


def plot_nand(r2, r3, vtc2, vtc3, vdd, f, path):
    fig, axs = plt.subplots(4, 2, figsize=(12, 9), sharex="col",
                            gridspec_kw=dict(height_ratios=[1.3, 1, 1, 0.9]))
    for col, (r, n) in enumerate(((r2, 2), (r3, 3))):
        t = r["time"]
        ax = axs[0, col]
        for k, a in enumerate(r["ins"]):
            _wave(ax, t, r[a] + (vdd + 1) * (n - 1 - k), a.upper(), SERIES[k])
        ax.set_yticks([(vdd + 1) * (n - 1 - k) + vdd / 2 for k in range(n)],
                      [a.upper() for a in r["ins"]])
        ax.set_title(f"NAND{n} -- inputs (0/{vdd:g} V, {f / 1e3:g} kHz)")
        ax = axs[1, col]
        _wave(ax, t, r["y"], "Y", C["blue"])
        ax.set_ylabel("Y (V)")
        ax.set_title("output Y (fan-out 3)")
        ax.axhline(vdd, color=C["ink2"], lw=0.8, ls=":")
        ax = axs[2, col]
        _wave(ax, t, r["dut_x"], "X", C["orange"])
        ax.axhline(vdd, color=C["ink2"], lw=0.8, ls=":")
        ax.text(t[-1] * 1e6, vdd, " VDD", va="bottom", ha="right", color=C["ink2"], fontsize=8)
        ax.set_ylabel("X (V)")
        ax.set_title("internal node X (gate of pull-up M4) -- bootstrapped above VDD")
        ax = axs[3, col]
        _wave(ax, t, r["i(vdd)"] * 1e6, "IDD", C["aqua"])
        ax.set_ylabel("IDD (uA)")
        ax.set_title("supply current, DUT + 3 load gates")
        ax.set_xlabel("time (us)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (vtc, n) in zip(axs, ((vtc2, 2), (vtc3, 3))):
        for k, pos in enumerate(("top", "bottom")):
            vin, vout, _ = vtc[pos]
            m = vtc_metrics(vin, vout)
            ax.plot(vin, vout, color=SERIES[k],
                    label=f"swept input at {pos} of stack: VM {m['VM']:.2f} V, "
                          f"gain {m['gain']:.1f}, NM {m['NM_mec']:.2f} V")
        ax.plot([0, vdd], [0, vdd], color=C["ink2"], lw=0.8, ls=":")
        ax.set_xlabel("swept input (V), other inputs at VDD")
        ax.set_ylabel("Y (V)")
        ax.set_title(f"NAND{n} DC transfer, VDD = {vdd:g} V, FO3")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.replace("transient", "vtc"), dpi=150)
    plt.close(fig)


def plot_dff(r, m, vdd, f, path):
    t = r["time"]
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1.4, 0.9]))
    _wave(axs[0], t, r["clk"], "CLK", C["ink2"])
    axs[0].set_title(f"CLK ({f / 1e3:g} kHz)")
    _wave(axs[1], t, r["d"], "D", C["orange"])
    axs[1].set_title("D (edges alternate between CLK-low and CLK-high halves)")
    _wave(axs[2], t, r["q"], "Q", C["blue"])
    _wave(axs[2], t, r["qb"], "Q̄", C["aqua"])
    for te, e in zip(crossings(t, r["clk"], vdd / 2, +1), m["expected"]):
        axs[2].axvline(te * 1e6, color=C["grid"], lw=1, zorder=0)
        axs[2].text(te * 1e6, vdd * 1.07, f"D={e}", ha="center", fontsize=7, color=C["ink2"])
    axs[2].set_title("Q / Q̄ -- D sampled on each rising CLK edge")
    axs[2].legend(loc="center right")
    axs[2].set_ylim(-0.3, vdd * 1.18)
    _wave(axs[3], t, r["i(vdd)"] * 1e6, "IDD", C["aqua"])
    axs[3].set_title("supply current, DFF + 2 load gates")
    axs[3].set_ylabel("uA")
    for ax in axs[:3]:
        ax.set_ylabel("V")
    axs[-1].set_xlabel("time (us)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_counter(r, m, vdd, f, path):
    t = r["time"]
    fig, axs = plt.subplots(6, 1, figsize=(11, 10), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1.3, 0.9]))
    _wave(axs[0], t, r["clk"], "CLK", C["ink2"])
    axs[0].set_title(f"CLK ({f / 1e3:g} kHz, {vdd:g} V)")
    for k in range(3):
        _wave(axs[k + 1], t, r[f"q{k}"], f"Q{k}", SERIES[k])
        axs[k + 1].set_title(f"Q{k}" + (" (LSB)" if k == 0 else " (MSB)" if k == 2 else ""))
    for ax in axs[:4]:
        ax.set_ylabel("V")
        ax.set_ylim(-0.3, vdd * 1.12)
    bits = np.array([r[f"q{k}"] > vdd / 2 for k in range(3)]).astype(int)
    val = bits[0] + 2 * bits[1] + 4 * bits[2]
    axs[4].step(t * 1e6, val, where="post", color=C["blue"])
    axs[4].set_yticks(range(8))
    axs[4].set_ylabel("count")
    axs[4].set_title("decoded count Q2Q1Q0 (threshold VDD/2) -- " +
                     ("counts 0-7 correctly" if m["functional"] else "COUNT ERROR"))
    _wave(axs[5], t, r["i(vdd)"] * 1e6, "IDD", C["aqua"])
    axs[5].set_ylabel("uA")
    axs[5].set_title(f"supply current (avg {m['P_avg_uW']:.0f} uW, {m['E_per_clock_nJ']:.1f} nJ/clock)")
    axs[-1].set_xlabel("time (us)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_counter_zoom(r, vdd, f, path):
    """Ripple through all three stages at the 3 -> 4 transition."""
    T = 1 / f
    t = r["time"]
    ck = crossings(t, r["clk"], vdd / 2, +1)
    te = ck[3]                                  # 4th rising edge: 011 -> 100
    w = (t > te - 0.1 * T) & (t < te + 0.6 * T)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot((t[w] - te) * 1e6, r["clk"][w], color=C["ink2"], label="CLK")
    for k in range(3):
        ax.plot((t[w] - te) * 1e6, r[f"q{k}"][w], color=SERIES[k], label=f"Q{k}")
    ax.axhline(vdd / 2, color=C["grid"], lw=1)
    ax.set_xlabel("time from CLK rising edge (us)")
    ax.set_ylabel("V")
    ax.set_title("ripple at the 3 -> 4 (011 -> 100) transition: each stage clocks the next")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdd", type=float, default=5.0)
    ap.add_argument("--f", type=float, default=50e3)
    ap.add_argument("--tag", default="5V_50k")
    a = ap.parse_args()
    vdd, f, tag = a.vdd, a.f, a.tag
    os.makedirs(OUT, exist_ok=True)
    style()
    res = dict(vdd=vdd, f=f, sizing={k: (list(v) if isinstance(v, tuple) else v) for k, v in SIZING.items()})

    print("NAND2 / NAND3 ...", flush=True)
    r2, r3 = nand_bench(2, vdd, f), nand_bench(3, vdd, f)
    v2, v3 = nand_vtc(2, vdd), nand_vtc(3, vdd)
    res["nand2"] = dict(nand_measure(r2, vdd, f), **{f"vtc_{p}": vtc_metrics(*v2[p][:2]) for p in v2},
                        ntft=r2["ntft"])
    res["nand3"] = dict(nand_measure(r3, vdd, f), **{f"vtc_{p}": vtc_metrics(*v3[p][:2]) for p in v3},
                        ntft=r3["ntft"])
    plot_nand(r2, r3, v2, v3, vdd, f, os.path.join(OUT, f"nand_transient_{tag}.png"))

    print("DFF ...", flush=True)
    rd = dff_bench(vdd, f)
    res["dff"] = dict(dff_measure(rd, vdd, f), ntft=rd["ntft"], cpu_s=rd["cpu_s"])
    plot_dff(rd, res["dff"], vdd, f, os.path.join(OUT, f"dff_transient_{tag}.png"))

    print("counter ...", flush=True)
    rc = counter_bench(vdd, f)
    res["counter"] = dict(counter_measure(rc, vdd, f), ntft=rc["ntft"], cpu_s=rc["cpu_s"])
    plot_counter(rc, res["counter"], vdd, f, os.path.join(OUT, f"counter_transient_{tag}.png"))
    plot_counter_zoom(rc, vdd, f, os.path.join(OUT, f"counter_ripple_{tag}.png"))
    np.savez_compressed(os.path.join(OUT, f"counter_waveforms_{tag}.npz"),
                        **{k: rc[k] for k in ("time", "clk", "q0", "q1", "q2", "qb0", "qb1", "qb2", "i(vdd)")})

    with open(os.path.join(OUT, f"metrics_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
