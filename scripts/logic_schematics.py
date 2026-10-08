"""Draw the schematics: transistor-level NAND2/NAND3, gate-level D flip-flop,
block-level 3-bit ripple counter.

    python scripts/logic_schematics.py      # -> outputs/logic_test/*_schematic.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import schemdraw                         # noqa: E402
import schemdraw.elements as elm         # noqa: E402
import schemdraw.logic as logic          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic import SIZING              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "logic_test")


def wl(key):
    w, l = SIZING[key]
    return f"{w * 1e6:g}/{l * 1e6:g}"


def nand_tx(n, path):
    """Transistor-level bootstrapped pseudo-CMOS NAND with n inputs."""
    names = "ABC"[:n]
    fet = 1.5            # drain-source length of the schemdraw NFet
    with schemdraw.Drawing(show=False, file=path, dpi=160) as d:
        d.config(fontsize=11)

        def nfet(drain, gate_left):
            e = elm.NFet().right()
            if gate_left:
                e = e.reverse()
            return d.add(e.anchor("drain").at(drain))

        def tag(xy, text, align):
            d.add(elm.Label().at(xy).label(text, halign=align))

        # ---------------- stage 1: x = 0 ----------------
        d.add(elm.Vdd().at((0, 0)).label("VDD"))
        m2 = nfet((0, 0), gate_left=True)                  # diode-connected load
        d.add(elm.Line().at(m2.gate).left(0.5))
        d.add(elm.Line().up().toy(0.3))
        d.add(elm.Line().right().tox(0))
        tag((0.35, -0.75), f"M2  {wl('m2')}", "left")
        x = m2.source
        d.add(elm.Dot().at(x))
        tag((-0.25, x[1] - 0.25), "X", "right")
        d.add(elm.Line().at(x).down(0.5))
        top = d.here
        for k, a in enumerate(names):
            m = nfet(top, gate_left=True)
            d.add(elm.Line().at(m.gate).left(0.6))
            d.add(elm.Dot(open=True).label(a, loc="left"))
            tag((0.35, top[1] - 0.75), f"M1{a.lower()}  {wl('m1')}", "left")
            top = m.source
        d.add(elm.Ground().at(top))

        # ---------------- stage 2: x = X2 ----------------
        X2 = 5.5
        d.add(elm.Vdd().at((X2, 0)).label("VDD"))
        m4 = nfet((X2, 0), gate_left=True)                 # source-follower pull-up
        tag((X2 + 0.35, -0.75), f"M4  {wl('m4')}", "left")
        out = m4.source
        d.add(elm.Dot().at(out))
        # X -> gate of M4, Cb from X to OUT
        xj = (2.6, x[1])
        d.add(elm.Line().at(x).to(xj))
        d.add(elm.Dot().at(xj))
        d.add(elm.Line().at(xj).up().toy(m4.gate))
        d.add(elm.Line().right().tox(m4.gate))
        d.add(elm.Capacitor().at(xj).right().tox(X2))
        tag(((xj[0] + X2) / 2, x[1] - 0.75), f"Cb {SIZING['cb'] * 1e12:g} pF", "center")
        d.add(elm.Line().at(out).right(1.8))
        d.add(elm.Dot(open=True).label("Y", loc="right"))
        d.add(elm.Line().at(out).down(0.5))
        top = d.here
        for k, a in enumerate(names):
            m = nfet(top, gate_left=False)
            d.add(elm.Line().at(m.gate).right(0.6))
            d.add(elm.Dot(open=True).label(a, loc="right"))
            tag((X2 - 0.35, top[1] - 0.75), f"M3{a.lower()}  {wl('m3')}", "right")
            top = m.source
        d.add(elm.Ground().at(top))
        d.add(elm.Label().at((X2 / 2, top[1] - 1.6)).label(
            f"W/L in um;  stage 1: X = NAND({', '.join(names)}),  stage 2 pull-up M4 bootstrapped by Cb",
            fontsize=9))


def dff(path):
    with schemdraw.Drawing(show=False, file=path, dpi=160) as d:
        d.config(fontsize=11, unit=1.5)
        g1 = d.add(logic.Nand().at((0, 4)).anchor("in1").label("G1", loc="top"))
        g2 = d.add(logic.Nand().at((0, 1.5)).anchor("in1").label("G2", loc="bottom"))
        g3 = d.add(logic.Nand(inputs=3).at((0, -1.5)).anchor("in1").label("G3", loc="top"))
        g4 = d.add(logic.Nand().at((0, -4.5)).anchor("in1").label("G4", loc="bottom"))
        g5 = d.add(logic.Nand().at((7, 2.3)).anchor("in1").label("G5", loc="top"))
        g6 = d.add(logic.Nand().at((7, -1.5)).anchor("in1").label("G6", loc="bottom"))

        def wire(a, b, shape="-|", **k):
            d.add(logic.Wire(shape, **k).at(a).to(b))

        # n1 = G1 out -> G2.in1
        d.add(elm.Line().at(g1.out).right(0.6)); p = d.here; d.add(elm.Dot()); d.add(elm.Label().at((p[0] + 0.3, p[1] + 0.3)).label("n1"))
        d.add(elm.Line().at(p).down(1.2)); d.add(elm.Line().left().tox(g2.in1[0] - 0.8))
        d.add(elm.Line().down().toy(g2.in1)); d.add(elm.Line().right().tox(g2.in1))
        # n2 = G2 out -> G1.in2, G3.in1, G5.in1
        d.add(elm.Line().at(g2.out).right(1.2)); n2 = d.here; d.add(elm.Dot())
        d.add(elm.Label().at((n2[0] + 0.3, n2[1] + 0.3)).label("n2"))
        d.add(elm.Line().at(n2).up(1.0)); d.add(elm.Line().left().tox(g1.in2[0] - 0.4))
        d.add(elm.Line().up().toy(g1.in2)); d.add(elm.Line().right().tox(g1.in2))
        d.add(elm.Line().at(n2).right(2.0)); d.add(elm.Line().up().toy(g5.in1)); d.add(elm.Line().right().tox(g5.in1))
        d.add(elm.Line().at(n2).down(1.6)); d.add(elm.Line().left().tox(g3.in1[0] - 1.2))
        d.add(elm.Line().down().toy(g3.in1)); d.add(elm.Line().right().tox(g3.in1))
        # n3 = G3 out -> G4.in1, G6.in2
        d.add(elm.Line().at(g3.out).right(1.6)); n3 = d.here; d.add(elm.Dot())
        d.add(elm.Label().at((n3[0] + 0.3, n3[1] + 0.3)).label("n3"))
        d.add(elm.Line().at(n3).down(1.4)); d.add(elm.Line().left().tox(g4.in1[0] - 0.4))
        d.add(elm.Line().down().toy(g4.in1)); d.add(elm.Line().right().tox(g4.in1))
        d.add(elm.Line().at(n3).right(2.4)); d.add(elm.Line().down().toy(g6.in2)); d.add(elm.Line().right().tox(g6.in2))
        # n4 = G4 out -> G3.in3, G1.in1
        d.add(elm.Line().at(g4.out).right(2.0)); n4 = d.here; d.add(elm.Dot())
        d.add(elm.Label().at((n4[0] + 0.3, n4[1] + 0.3)).label("n4"))
        d.add(elm.Line().at(n4).down(0.8)); d.add(elm.Line().left().tox(g1.in1[0] - 2.0))
        d.add(elm.Line().up().toy(g1.in1)); d.add(elm.Line().right().tox(g1.in1))
        d.add(elm.Line().at((g1.in1[0] - 2.0, g3.in3[1])).right().tox(g3.in3))
        d.add(elm.Dot().at((g1.in1[0] - 2.0, g3.in3[1])))
        # CLK -> G2.in2, G3.in2
        ck = (g2.in2[0] - 3.0, g2.in2[1])
        d.add(elm.Dot(open=True).at(ck).label("CLK", loc="left"))
        d.add(elm.Line().at(ck).right().tox(g2.in2[0] - 0.8)); cj = d.here; d.add(elm.Dot())
        d.add(elm.Line().right().tox(g2.in2))
        d.add(elm.Line().at(cj).down().toy(g3.in2)); d.add(elm.Line().right().tox(g3.in2))
        # D -> G4.in2
        dd = (g4.in2[0] - 3.0, g4.in2[1])
        d.add(elm.Dot(open=True).at(dd).label("D", loc="left"))
        d.add(elm.Line().at(dd).right().tox(g4.in2))
        # output latch
        d.add(elm.Line().at(g5.out).right(1.0)); q = d.here; d.add(elm.Dot())
        d.add(elm.Line().right(1.0)); d.add(elm.Dot(open=True).label("Q", loc="right"))
        d.add(elm.Line().at(q).down(1.2)); d.add(elm.Line().left().tox(g6.in1[0] - 0.4))
        d.add(elm.Line().down().toy(g6.in1)); d.add(elm.Line().right().tox(g6.in1))
        d.add(elm.Line().at(g6.out).right(1.0)); qb = d.here; d.add(elm.Dot())
        d.add(elm.Line().right(1.0)); d.add(elm.Dot(open=True).label("Q̄", loc="right"))
        d.add(elm.Line().at(qb).up(1.2)); d.add(elm.Line().left().tox(g5.in2[0] - 0.8))
        d.add(elm.Line().up().toy(g5.in2)); d.add(elm.Line().right().tox(g5.in2))


def ff_block(d, xy, name):
    """Rectangle D flip-flop symbol, returns pin coordinates."""
    x, y = xy
    w, h = 2.4, 3.0
    d.add(elm.Line().at((x, y)).right(w))
    d.add(elm.Line().down(h))
    d.add(elm.Line().left(w))
    d.add(elm.Line().up(h))
    d.add(elm.Label().at((x + w / 2, y + 0.35)).label(name))
    pins = dict(D=(x, y - 0.75), CLK=(x, y - 2.25), Q=(x + w, y - 0.75), QB=(x + w, y - 2.25))
    d.add(elm.Label().at((x + 0.3, y - 0.75)).label("D", halign="left"))
    d.add(elm.Label().at((x + 0.45, y - 2.25)).label("CLK", halign="left"))
    d.add(elm.Line().at((x, y - 2.0)).to((x + 0.3, y - 2.25)))
    d.add(elm.Line().at((x + 0.3, y - 2.25)).to((x, y - 2.5)))
    d.add(elm.Label().at((x + w - 0.3, y - 0.75)).label("Q", halign="right"))
    d.add(elm.Label().at((x + w - 0.3, y - 2.25)).label("Q̄", halign="right"))
    return pins


def counter(path):
    with schemdraw.Drawing(show=False, file=path, dpi=160) as d:
        d.config(fontsize=11, unit=1.5)
        prev_clk = None
        xs = [0, 5.5, 11]
        for k, x0 in enumerate(xs):
            p = ff_block(d, (x0, 0), f"FF{k}")
            # D <- Qbar feedback
            d.add(elm.Line().at(p["QB"]).right(0.8)); j = d.here; d.add(elm.Dot())
            d.add(elm.Line().down(1.2)); d.add(elm.Line().left().tox(p["D"][0] - 0.8))
            d.add(elm.Line().up().toy(p["D"])); d.add(elm.Line().right().tox(p["D"]))
            # Q output tap
            d.add(elm.Line().at(p["Q"]).right(0.8)); d.add(elm.Line().up(1.2))
            d.add(elm.Dot(open=True).label(f"Q{k}", loc="top"))
            if k == 0:
                d.add(elm.Line().at(p["CLK"]).left(1.6))
                d.add(elm.Dot(open=True).label("CLK", loc="left"))
            else:
                d.add(elm.Line().at(prev_clk).right().tox(p["CLK"]))
            prev_clk = j
        d.add(elm.Label().at((7.0, -5.2)).label(
            "asynchronous (ripple) up-counter: each stage toggles (D = Q̄) and clocks the next from its Q̄", fontsize=10))


def main():
    os.makedirs(OUT, exist_ok=True)
    nand_tx(2, os.path.join(OUT, "nand2_schematic.png"))
    nand_tx(3, os.path.join(OUT, "nand3_schematic.png"))
    dff(os.path.join(OUT, "dff_schematic.png"))
    counter(os.path.join(OUT, "counter_schematic.png"))


if __name__ == "__main__":
    main()
