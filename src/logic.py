"""Bootstrapped pseudo-CMOS logic in a-IGZO (n-type only), simulated in ngspice
with the transliterated verilogA/ntft_full.va (src/ngspice_tft.py).

Gate topology (n-type pseudo-CMOS, single supply, bootstrapped):

    VDD --+------------------+
          |                  |
         M2 (diode, g=VDD)  M4 (pull-up, g=X)
          |                  |
          X ----||Cb||------OUT
          |                  |
       M1a..M1n (series,  M3a..M3n (series,
        gates = inputs)    gates = inputs)
          |                  |
         GND                GND

Stage 1 (M2 load + M1 stack) computes X = NAND(inputs) and drives the gate of
the source-follower pull-up M4. Stage 2's pull-down stack M3 is driven by the
inputs directly, so the output pull-down is never fighting a fully-on M4.
When the output rises, Cb (in parallel with M4's own Cgs) lifts X above VDD
so M4 stays on and the output reaches the full VDD instead of VDD - Vth.
"""
import os
import subprocess
import tempfile

import numpy as np

from .ngspice_tft import library, subckt_name
from .tran import Circuit, pulse_fn, pwl_fn

# default sizing (W, L in m; Cb in F). Chosen in scripts/logic_sizing.py.
SIZING = dict(
    m1=(20e-6, 5e-6),     # stage-1 stack (per input)
    m2=(5e-6, 20e-6),     # stage-1 diode load (weakest buildable: sets static power)
    m3=(40e-6, 5e-6),     # output pull-down stack (per input)
    m4=(40e-6, 5e-6),     # output pull-up (source follower)
    cb=1.0e-12,           # bootstrap capacitor X-OUT
)


class Netlist:
    """Structural netlist: emits both a src.tran Circuit and an ngspice deck."""

    def __init__(self, sizing=None):
        self.sz = dict(SIZING, **(sizing or {}))
        self.tfts = []      # (name, d, g, s, (w, l))
        self.caps = []      # (name, a, b, C)
        self.dc = {}        # node -> V
        self.pulses = {}    # node -> (v0, v1, delay, tr, tf, pw, per)
        self.pwls = {}      # node -> [(t, V), ...]

    @property
    def ntft(self):
        return len(self.tfts)

    def tft(self, name, d, g, s, geom):
        self.tfts.append((name, d, g, s, tuple(geom)))

    def cap(self, name, a, b, c):
        self.caps.append((name, a, b, c))

    def vdc(self, node, v):
        self.dc[node] = v

    def vpulse(self, node, v0, v1, period, delay=0.0, tr=100e-9, duty=0.5):
        self.pulses[node] = (v0, v1, delay, tr, tr, period * duty - tr, period)

    def vpwl(self, node, points):
        self.pwls[node] = list(points)

    def nand(self, name, ins, out, vdd="vdd"):
        """Bootstrapped pseudo-CMOS NAND with len(ins) inputs."""
        sz = self.sz
        x = f"{name}_x"
        n = len(ins)
        # stage 1: series stack X -> GND, inputs ordered top..bottom
        top = x
        for k, a in enumerate(ins):
            bot = "0" if k == n - 1 else f"{name}_s1_{k}"
            self.tft(f"{name}_m1{k}", top, a, bot, sz["m1"])
            top = bot
        self.tft(f"{name}_m2", vdd, vdd, x, sz["m2"])
        # stage 2
        top = out
        for k, a in enumerate(ins):
            bot = "0" if k == n - 1 else f"{name}_s3_{k}"
            self.tft(f"{name}_m3{k}", top, a, bot, sz["m3"])
            top = bot
        self.tft(f"{name}_m4", vdd, x, out, sz["m4"])
        if sz["cb"] > 0:
            self.cap(f"{name}_b", x, out, sz["cb"])

    def dff(self, name, d, clk, q, qb, vdd="vdd"):
        """Positive-edge D flip-flop: 5 NAND2 + 1 NAND3 (7474 core, no PRE/CLR)."""
        n1, n2, n3, n4 = (f"{name}_n{k}" for k in range(1, 5))
        self.nand(f"{name}_g1", [n4, n2], n1, vdd)
        self.nand(f"{name}_g2", [n1, clk], n2, vdd)
        self.nand(f"{name}_g3", [n2, clk, n4], n3, vdd)
        self.nand(f"{name}_g4", [n3, d], n4, vdd)
        self.nand(f"{name}_g5", [n2, qb], q, vdd)
        self.nand(f"{name}_g6", [q, n3], qb, vdd)
        return dict(n1=n1, n2=n2, n3=n3, n4=n4)

    # ------------------------------------------------------------ backends
    def circuit(self):
        c = Circuit()
        for name, d, g, s, (w, l) in self.tfts:
            c.add_tft(name, d, g, s, w, l)
        for name, a, b, cc in self.caps:
            c.add_cap(name, a, b, cc)
        for node, v in self.dc.items():
            c.add_source(node, (lambda vv: (lambda t: vv))(v), [])
        self._pulse_bps = {}
        for node, p in self.pulses.items():
            fn, bps = pulse_fn(*p)
            c.add_source(node, fn, [])
            self._pulse_bps[node] = bps
        for node, pts in self.pwls.items():
            fn, bps = pwl_fn(pts)
            c.add_source(node, fn, bps)
        return c

    def simulate(self, tstop, nodeset=None, **kw):
        c = self.circuit()
        for node, bps in self._pulse_bps.items():
            fn, _ = c.sources[c.nodes[node]]
            c.sources[c.nodes[node]] = (fn, bps(tstop))
        return c.tran(tstop, v0=nodeset, **kw)

    def ngspice(self, title, tstop, tstep, saves, nodeset=None, form="cdv"):
        geoms = {t[4] for t in self.tfts}
        L = [f"* {title}", library(geoms, form)]
        for name, d, g, s, geom in self.tfts:
            L.append(f"x{name} {d} {g} {s} {subckt_name(*geom, form)}")
        for name, a, b, c in self.caps:
            L.append(f"c{name} {a} {b} {c:.6g}")
        for node, v in self.dc.items():
            L.append(f"v_{node} {node} 0 {v}")
        for node, (v0, v1, dl, tr, tf, pw, per) in self.pulses.items():
            L.append(f"v_{node} {node} 0 pulse({v0} {v1} {dl:.6g} {tr:.6g} {tf:.6g} {pw:.6g} {per:.6g})")
        for node, pts in self.pwls.items():
            L.append(f"v_{node} {node} 0 pwl(" + " ".join(f"{t:.6g} {v:.6g}" for t, v in pts) + ")")
        if nodeset:
            L.append(".nodeset " + " ".join(f"v({k})={v}" for k, v in nodeset.items()))
        L.append(".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7 "
                 "chgtol=1e-17 itl4=200 rshunt=1e13")
        L.append(".control\nset noaskquit")
        L.append(f"tran {tstep:.4g} {tstop:.4g}")
        L.append("wrdata {OUT} " + " ".join(saves))
        L.append(".endc\n.end\n")
        return "\n".join(L)


def nand_nodeset(name, ins, out_high, vdd):
    """Initial guess for one gate: X follows the output, stack nodes at 0."""
    v = vdd if out_high else 0.0
    ns = {f"{name}_x": v}
    for k in range(len(ins) - 1):
        ns[f"{name}_s1_{k}"] = 0.0
        ns[f"{name}_s3_{k}"] = 0.0
    return ns


def dff_nodeset(name, d, q, q_node, qb_node, vdd, clk=0):
    """Nodeset for a DFF holding Q=q with input D=d and the clock at `clk`."""
    if not clk:                      # transparent master: n1 follows D
        n1, n2, n3, n4 = d, 1, 1, 1 - d
    elif q:                          # clock high, 1 captured: n2 low locks it
        n1, n2, n3, n4 = 1, 0, 1, 1 - d
    else:                            # clock high, 0 captured: n3 low locks it
        n1, n2, n3, n4 = 0, 1, 0, 1
    vals = {f"{name}_n1": n1, f"{name}_n2": n2, f"{name}_n3": n3,
            f"{name}_n4": n4, q_node: q, qb_node: 1 - q}
    ns = {k: vdd * b for k, b in vals.items()}
    outs = dict(g1=n1, g2=n2, g3=n3, g4=n4, g5=q, g6=1 - q)
    nins = dict(g1=2, g2=2, g3=3, g4=2, g5=2, g6=2)
    for g, b in outs.items():
        ns.update(nand_nodeset(f"{name}_{g}", range(nins[g]), b, vdd))
    return ns


def run_ngspice(net_text, cols, workdir=None, timeout=36000):
    """Run ngspice in batch mode; return dict col -> array (plus 'time')."""
    wd = workdir or tempfile.mkdtemp(prefix="ngs_")
    os.makedirs(wd, exist_ok=True)
    out = os.path.join(wd, "out.txt")
    cir = os.path.join(wd, "deck.cir")
    with open(cir, "w") as f:
        f.write(net_text.replace("{OUT}", out))
    r = subprocess.run(["ngspice", "-b", cir], capture_output=True, text=True,
                       timeout=timeout)
    if not os.path.exists(out):
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    d = np.loadtxt(out)
    res = {"time": d[:, 0]}
    for k, c in enumerate(cols):
        res[c] = d[:, 2 * k + 1]
    return res
