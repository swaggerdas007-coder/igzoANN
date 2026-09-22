"""n-only a-IGZO TFT differential pair, built on the ANN Verilog-A model.

a-IGZO has no p-type device, so the load cannot be a PMOS current mirror.
Two loads that a real IGZO process can build are simulated:

  "res"   resistive load (thin-film resistor). Av = gm1 * (R || ro) -- the
          gain leans on gm, which this model gets right.
  "diode" NMOS load, gate tied to its drain at VDD, source at the output.
          In n-only tech a top-side NMOS is always a source follower, so the
          load impedance is 1/(gm_L+gds_L) and Av ~ gm1/gm_L -- low gain, but
          bias-insensitive and fast.

Nodes: 1 out1, 2 out2, 3 tail, 4 vin1, 5 vin2, 6 vdd, 7 vbias(tail gate)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.circuit import Circuit, model, _f    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")
os.makedirs(OUT, exist_ok=True)

OUT1, OUT2, TAIL, IN1, IN2, VDDN, VB = 1, 2, 3, 4, 5, 6, 7


def build(load, w1, l1, w3, l3, vdd, vcm, vb, rl=None, wl=None, ll=None, cload=0.0):
    c = Circuit(7)
    c.fixed = {IN1: vcm, IN2: vcm, VDDN: vdd, VB: vb}
    c.add_tft("M1", OUT1, IN1, TAIL, w1, l1)
    c.add_tft("M2", OUT2, IN2, TAIL, w1, l1)
    c.add_tft("M3", TAIL, VB, 0, w3, l3)
    if load == "res":
        c.add_res(VDDN, OUT1, rl)
        c.add_res(VDDN, OUT2, rl)
    else:
        c.add_tft("ML1", VDDN, VDDN, OUT1, wl, ll)
        c.add_tft("ML2", VDDN, VDDN, OUT2, wl, ll)
    if cload:
        c.add_cap(OUT1, 0, cload)
        c.add_cap(OUT2, 0, cload)
    return c


def analyse(c, vdd, cload, fmax=1e9):
    v, ok = c.solve_dc(guess={OUT1: vdd * 0.6, OUT2: vdd * 0.6, TAIL: 0.5})
    if not ok:
        return None
    ops = {t.name: t.op(v) for t in c.tfts}
    itail = ops["M3"]["id"]
    if itail < 1e-9 or itail > 1e-3:
        return None
    # every device must sit inside the model's trained box, unclamped
    for t in c.tfts:
        vgs, vds = t.bias(v)
        if not (-5 <= vgs <= 5) or not (0.05 <= vds <= 5):
            return None
        if ops[t.name]["clamped_vg"] or ops[t.name]["clamped_vd"]:
            return None
    # M1/M2 must be saturated: use the model's own gds, not a VT formula
    if ops["M1"]["gds"] <= 0 or ops["M1"]["gm"] / ops["M1"]["gds"] < 2:
        return None

    f = np.logspace(0, np.log10(fmax), 600)
    vac = c.ac(v, f, {IN1: 0.5, IN2: -0.5, VDDN: 0.0, VB: 0.0})
    add = vac[:, OUT1] - vac[:, OUT2]          # differential output / vid
    a0 = abs(add[0])
    if a0 < 1e-3:
        return None
    mag = np.abs(add) / a0
    below = np.where(mag < 1 / np.sqrt(2))[0]
    if len(below):
        k = below[0]
        f3 = float(np.interp(1 / np.sqrt(2), [mag[k], mag[k - 1]], [f[k], f[k - 1]]))
        feedthrough = False
    else:
        # gate-overlap Cgd feeds the input straight to the output, holding the
        # magnitude up past the pole -- there is no usable -3dB corner
        f3, feedthrough = np.inf, True

    # common mode: drive both inputs together. A perfectly matched pair has
    # exactly zero differential CM gain, so quote the single-ended CMRR, which
    # is the one a real (mismatched) layout approaches.
    vcm_ac = c.ac(v, f[:1], {IN1: 1.0, IN2: 1.0, VDDN: 0.0, VB: 0.0})
    acm_se = abs(vcm_ac[0, OUT1])
    fz = ops["M1"]["gm"] / (2 * np.pi * ops["M1"]["cgd"])
    return dict(ok=True, feedthrough_limited=feedthrough, f_zero_cgd=fz,
                vout=v[OUT1], vtail=v[TAIL], itail=itail,
                gm1=ops["M1"]["gm"], gds1=ops["M1"]["gds"],
                av=a0, av_db=20 * np.log10(a0), f3db=f3,
                gbw=a0 * f3, power=itail * vdd,
                cgs1=ops["M1"]["cgs"], cgd1=ops["M1"]["cgd"],
                cmrr_db=20 * np.log10((a0 / 2) / max(acm_se, 1e-18)),
                cm_gain_se_db=20 * np.log10(max(acm_se, 1e-18)),
                slew_v_per_us=itail / max(cload, 1e-15) * 1e-6,
                gm_over_id=ops["M1"]["gm"] / ops["M1"]["id"],
                ro1=1 / ops["M1"]["gds"], av_intrinsic=ops["M1"]["gm"] / ops["M1"]["gds"])


def solve_rl(w1, l1, w3, l3, vdd, vcm, vb, vdrop, cload, iters=6):
    """Pick R_L so the output common mode sits vdrop volts below VDD."""
    rl = 100e3
    for _ in range(iters):
        c = build("res", w1, l1, w3, l3, vdd, vcm, vb, rl=rl, cload=cload)
        v, ok = c.solve_dc(guess={OUT1: vdd - vdrop, OUT2: vdd - vdrop, TAIL: 0.5})
        if not ok:
            return None, None
        i_half = float(np.ravel(model().id_(*c.tfts[0].bias(v), w1, l1))[0])
        if i_half < 1e-10:
            return None, None
        rl_new = vdrop / i_half
        if abs(rl_new - rl) / rl < 1e-4:
            rl = rl_new
            break
        rl = 0.5 * rl + 0.5 * rl_new
    c = build("res", w1, l1, w3, l3, vdd, vcm, vb, rl=rl, cload=cload)
    return c, rl
