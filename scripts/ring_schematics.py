"""Circuit diagrams: the bootstrapped pseudo-CMOS inverter, and the ring built
from it, plus the pad/buffer test structure.

Hand-laid-out in matplotlib rather than auto-routed, so wires go where a reader
expects and the only crossing in the whole drawing is marked with a hop.

    python scripts/ring_schematics.py   # -> outputs/pseudo_cmos/*_schematic.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.patches import Polygon                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "pseudo_cmos")
BLUE, ORANGE, AQUA, YELL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, WIRE = "#fcfcfb", "#0b0b0b", "#52514e", "#1c1c1b"
LW = 1.9


def line(ax, pts, c=WIRE, lw=LW, z=2, **kw):
    p = np.asarray(pts, float)
    ax.plot(p[:, 0], p[:, 1], color=c, lw=lw, solid_capstyle="round", zorder=z,
            **kw)


def dot(ax, xy, c=WIRE, r=5.5, z=4):
    ax.plot([xy[0]], [xy[1]], "o", color=c, ms=r, zorder=z)


def txt(ax, xy, s, c=INK, fs=11, ha="center", va="center", w="normal", z=5):
    ax.text(xy[0], xy[1], s, color=c, fontsize=fs, ha=ha, va=va,
            fontweight=w, zorder=z)


def hop(ax, xy, r=0.14, c=WIRE):
    """Wire crossing with no connection."""
    a = np.linspace(0, np.pi, 40)
    ax.plot(xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), color=c, lw=LW,
            zorder=3)


def nfet(ax, x, y, name, wl, c=WIRE, lc=INK, h=0.58, gate_lead=0.78,
         lab_dx=0.95):
    """n-channel TFT symbol. Returns (drain, gate, source) connection points."""
    gx, cx = x - 0.24, x
    line(ax, [(gx, y - h), (gx, y + h)], c=c)                    # gate plate
    for y0, y1 in ((y - h, y - h * 0.3), (y - h * 0.22, y + h * 0.22),
                   (y + h * 0.3, y + h)):                        # channel
        line(ax, [(cx, y0), (cx, y1)], c=c)
    line(ax, [(cx, y + h * 0.65), (cx + 0.55, y + h * 0.65),
              (cx + 0.55, y + h + 0.1)], c=c)                    # drain lead
    line(ax, [(cx, y - h * 0.65), (cx + 0.55, y - h * 0.65),
              (cx + 0.55, y - h - 0.1)], c=c)                    # source lead
    ax.annotate("", xy=(cx + 0.12, y - h * 0.65),
                xytext=(cx + 0.42, y - h * 0.65),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=0,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=3)
    line(ax, [(gx - gate_lead, y), (gx, y)], c=c)                # gate lead
    txt(ax, (cx + 0.55 + lab_dx, y), f"{name}\n{wl}", c=lc, fs=10,
        ha="left", w="bold")
    return (cx + 0.55, y + h + 0.1), (gx - gate_lead, y), (cx + 0.55, y - h - 0.1)


def inverter(path, s):
    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    YV, YG = 6.5, 0.25
    XL, XR = 2.9, 6.6

    # supply rails
    line(ax, [(1.1, YV), (8.1, YV)], c=INK2)
    txt(ax, (0.95, YV), "VDD", c=INK2, fs=12, ha="right", w="bold")
    line(ax, [(1.1, YG), (8.1, YG)], c=INK2)
    txt(ax, (0.95, YG), "GND", c=INK2, fs=12, ha="right", w="bold")
    for x in (1.35, 7.85):
        for yy in (YV, YG):
            pass

    # devices
    d1, g1, s1 = nfet(ax, XL, 5.0, "M1", f"{s['w1']:.0f}/{s['l']:.0f}",
                      c=BLUE, lc=BLUE, lab_dx=0.3)
    d2, g2, s2 = nfet(ax, XL, 1.8, "M2", f"{s['w2']:.0f}/{s['l']:.0f}",
                      c=BLUE, lc=BLUE, lab_dx=0.3)
    d3, g3, s3 = nfet(ax, XR, 5.0, "M3", f"{s['w3']:.0f}/{s['l']:.0f}",
                      c=ORANGE, lc=ORANGE)
    d4, g4, s4 = nfet(ax, XR, 1.8, "M4", f"{s['w4']:.0f}/{s['l']:.0f}",
                      c=ORANGE, lc=ORANGE)

    XN, ON = (d2[0], 3.4), (d4[0], 3.4)
    line(ax, [d1, (d1[0], YV)])                       # M1 drain -> VDD
    line(ax, [s1, XN])                                # M1 source -> X
    line(ax, [d2, XN])                                # M2 drain  -> X
    line(ax, [s2, (s2[0], YG)])                       # M2 source -> GND
    line(ax, [d3, (d3[0], YV)])
    line(ax, [s3, ON])
    line(ax, [d4, ON])
    line(ax, [s4, (s4[0], YG)])
    dot(ax, XN)
    dot(ax, ON)
    txt(ax, (XN[0] - 0.22, XN[1] + 0.26), "X", c=BLUE, fs=13, ha="right",
        w="bold")


    # M1 gate tied to VDD -> it is a diode, and it blocks when X rises past VDD
    line(ax, [g1, (g1[0] - 0.5, g1[1]), (g1[0] - 0.5, YV)], c=BLUE)
    dot(ax, (g1[0] - 0.5, YV), c=BLUE)
    txt(ax, (g1[0] - 0.65, (g1[1] + YV) / 2), "gate tied\nto VDD:\na diode",
        c=BLUE, fs=9, ha="right")

    # X drives M3's gate
    XB = 5.05
    line(ax, [XN, (XB, XN[1]), (XB, g3[1]), g3], c=BLUE)

    # bootstrap capacitor, X <-> OUT
    YC = 4.05
    line(ax, [(XB, YC), (XB + 0.62, YC)], c=AQUA)
    line(ax, [(XB + 0.62, YC - 0.3), (XB + 0.62, YC + 0.3)], c=AQUA, lw=2.6)
    line(ax, [(XB + 0.9, YC - 0.3), (XB + 0.9, YC + 0.3)], c=AQUA, lw=2.6)
    line(ax, [(XB + 0.9, YC), (ON[0], YC)], c=AQUA)
    dot(ax, (XB, YC), c=AQUA)
    dot(ax, (ON[0], YC), c=AQUA)
    txt(ax, (XB - 0.18, YC), f"Cboot\n{s['cb']:.0f} pF", c=AQUA, fs=10.5,
        ha="right", w="bold")

    # IN bus to both driver gates (one crossing, hopped)
    YB = 0.95
    line(ax, [g2, (g2[0] - 0.45, g2[1]), (g2[0] - 0.45, YB), (g4[0], YB), g4])
    dot(ax, (g2[0] - 0.45, g2[1]))
    line(ax, [(g2[0] - 0.45, g2[1]), (0.95, g2[1])])
    txt(ax, (0.8, g2[1]), "IN", fs=12, ha="right", w="bold")
    hop(ax, (s2[0], YB))

    line(ax, [ON, (8.6, ON[1])], c=ORANGE)
    txt(ax, (8.75, ON[1]), "OUT", c=ORANGE, fs=12, ha="left", w="bold")

    txt(ax, (4.8, 7.55), "Bootstrapped pseudo-CMOS inverter", fs=15, w="bold")
    txt(ax, (4.8, 7.05),
        "every device is n-type - a-IGZO has no p-type, so this is what "
        "replaces a CMOS inverter.   W/L in um, L = 5 um throughout.",
        c=INK2, fs=10)
    txt(ax, (4.8, -0.65),
        "stage 1  (M1 load + M2 driver)  makes  X = NOT(IN)        "
        "stage 2  (M3 pull-up, gate on X + M4 pull-down, gate on IN)  drives OUT",
        c=INK2, fs=10)
    txt(ax, (4.8, -1.1),
        "M3's gate is driven by X, so the pull-up is switched OFF while M4 pulls "
        "down - that is what decouples VOL from the W ratio.", c=INK2, fs=10)
    txt(ax, (4.8, -1.55),
        "Cboot couples the rising OUT back into M3's own gate. M1 is "
        "diode-connected, so it blocks once X passes VDD and the boost survives.",
        c=AQUA, fs=10)
    ax.set_xlim(-0.6, 10.0)
    ax.set_ylim(-1.95, 7.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def inv_tri(ax, x, y, w=1.05, h=0.78, c=WIRE, fill="#ffffff", lab=None,
            labc=INK2):
    ax.add_patch(Polygon([[x, y - h / 2], [x, y + h / 2], [x + w, y]],
                         closed=True, fc=fill, ec=c, lw=LW, zorder=3))
    cr = 0.11
    a = np.linspace(0, 2 * np.pi, 40)
    ax.plot(x + w + cr + cr * np.cos(a), y + cr * np.sin(a), color=c, lw=LW,
            zorder=3)
    if lab:
        txt(ax, (x + w * 0.42, y - h / 2 - 0.34), lab, c=labc, fs=9.5)
    return (x, y), (x + w + 2 * cr, y)


def ring(path, nshow=5):
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y, x0, pitch = 2.0, 1.4, 1.85
    ports = []
    labs = ["1", "2", "3", "...", "N"]
    cols = [BLUE, BLUE, BLUE, INK2, BLUE]
    for i in range(nshow):
        a, b = inv_tri(ax, x0 + i * pitch, y, lab=labs[i], labc=cols[i])
        ports.append((a, b))
        if i:
            line(ax, [ports[i - 1][1], a])
    line(ax, [ports[-1][1], (ports[-1][1][0] + 0.9, y)])
    tap = (ports[-1][1][0] + 0.9, y)
    dot(ax, tap)
    line(ax, [tap, (tap[0] + 1.25, y)], c=ORANGE)
    txt(ax, (tap[0] + 1.42, y), "OUT", c=ORANGE, fs=12.5, ha="left", w="bold")
    yb = y - 1.25
    line(ax, [tap, (tap[0], yb), (x0 - 0.95, yb), (x0 - 0.95, y), ports[0][0]],
         c=BLUE)
    txt(ax, ((x0 + tap[0]) / 2, yb - 0.32),
        "inverting feedback - N must be ODD (an even ring just latches)",
        c=BLUE, fs=10.5)
    txt(ax, ((x0 + tap[0]) / 2, 3.72), "Ring oscillator: N inverters in a loop",
        fs=15, w="bold")
    txt(ax, ((x0 + tap[0]) / 2, 3.22),
        "each triangle is the 4-transistor + Cboot inverter above; every stage "
        "shares the same VDD and GND", c=INK2, fs=10)
    txt(ax, ((x0 + tap[0]) / 2, -0.35),
        r"$f = 1/(2\,N\,t_{pd})$,   $t_{pd}$ = 520 ns/stage at VDD = 3 V"
        "        N = 3 : 397 kHz        N = 7 : 137 kHz        "
        "N = 11 : 87 kHz        N = 23 : 42 kHz", fs=11)
    ax.set_xlim(-0.4, 13.0)
    ax.set_ylim(-0.75, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def testbench(path):
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = 2.2
    ax.add_patch(plt.Rectangle((0.6, y - 1.0), 3.5, 2.0, fc="#eaf1fb",
                               ec=BLUE, lw=LW, zorder=2))
    txt(ax, (2.35, y + 0.3), "RING", fs=14, w="bold")
    txt(ax, (2.35, y - 0.32), "N stages", c=INK2, fs=10.5)
    line(ax, [(4.1, y), (4.9, y)])
    px = 4.9
    for lab in ("1x", "4x", "16x"):
        a, b = inv_tri(ax, px, y, lab=lab, labc=ORANGE, c=ORANGE)
        px = b[0] + 0.45
        line(ax, [b, (px, y)], c=ORANGE)
    pad = (px + 0.75, y)
    line(ax, [(px, y), pad], c=ORANGE)
    ax.plot([pad[0]], [pad[1]], "s", color=ORANGE, ms=14, zorder=4)
    txt(ax, (pad[0], pad[1] + 0.52), "OUT", c=ORANGE, fs=12, w="bold")
    line(ax, [pad, (pad[0], y - 0.75)], c=INK2)
    for dy in (0, 0.26):
        line(ax, [(pad[0] - 0.3, y - 0.75 - dy), (pad[0] + 0.3, y - 0.75 - dy)],
             c=INK2, lw=2.6)
    line(ax, [(pad[0], y - 1.3), (pad[0], y - 1.6)], c=INK2)
    for i, w in enumerate((0.42, 0.27, 0.12)):
        line(ax, [(pad[0] - w, y - 1.6 - i * 0.15),
                  (pad[0] + w, y - 1.6 - i * 0.15)], c=INK2)
    txt(ax, (pad[0] + 0.45, y - 1.0), "$C_{pad}$ + probe\n0.5 - 50 pF", c=INK2,
        fs=9.5, ha="left")

    def pad_pin(x, yy, up, label, c):
        line(ax, [(x, yy), (x, yy + (0.9 if up else -0.9))], c=c)
        ax.plot([x], [yy + (0.9 if up else -0.9)], "s", color=c, ms=13, zorder=4)
        txt(ax, (x, yy + (1.42 if up else -1.42)), label, c=INK, fs=11.5,
            w="bold")

    pad_pin(2.35, y + 1.0, True, "VDD_core", BLUE)
    pad_pin(2.35, y - 1.0, False, "GND_core", BLUE)
    bx = 6.9
    pad_pin(bx, y + 0.42, True, "VDD_buf", ORANGE)
    pad_pin(bx, y - 0.42, False, "GND_buf", ORANGE)
    txt(ax, (5.6, 5.85), "What to tape out: five pads, not three", fs=15,
        w="bold")
    txt(ax, (5.6, 5.38),
        "VDD / GND / OUT is the true electrical minimum - the ring self-starts, "
        "needs no input, no bias, no reset.", c=INK2, fs=10)
    txt(ax, (5.6, -0.55),
        "Straight to a pad the ring loses 3% of its frequency at 1 pF and 35% "
        "at 50 pF, and the pad only sees 61% of VDD. Behind this buffer the "
        "frequency is flat to 0.01%.", c=INK2, fs=10)
    txt(ax, (5.6, -1.0),
        "The buffer draws about twice the ring's own current, so sharing one "
        "supply would make the core power - the number worth measuring - "
        "impossible to separate.", c=ORANGE, fs=10)
    ax.set_xlim(-0.2, 11.4)
    ax.set_ylim(-1.35, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    r = pd.read_csv(os.path.join(OUT, "recommended_gate.csv")).iloc[0]
    s = dict(w1=r.w1, w2=r.w2, w3=r.w3, w4=r.w4, l=5, cb=r.cboot * 1e12)
    inverter(os.path.join(OUT, "inverter_schematic.png"), s)
    ring(os.path.join(OUT, "ring_schematic.png"))
    testbench(os.path.join(OUT, "ring_testbench_schematic.png"))
    print("wrote inverter_schematic.png, ring_schematic.png, "
          "ring_testbench_schematic.png")
