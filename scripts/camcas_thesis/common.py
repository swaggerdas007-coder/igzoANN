"""Shared paths, data loading and the thesis's Von/Ioff onset rule for the
thesis-faithful CAMCAS pipeline (step0 ... step5 in this folder)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLEAN = os.path.join(REPO, "data_cleaned")
OUT = os.path.join(REPO, "outputs_camcas_thesis")
PLOTS = os.path.join(OUT, "plots")
VA_PATH = os.path.join(REPO, "verilogA", "tft_camcas_thesis.va")

VH = 5.0            # highest VGS of the transfer sweeps (thesis: VH = max VGS)
VDS_LIN = 0.1
VDS_SAT = 5.0

# 16 geometries used to extract the scaling coefficients (full 4x4 grid) and
# the 3 held out as unseen test geometries.
TRAIN_W = (20, 40, 80, 160)
TRAIN_L = (5, 10, 15, 20)
TRAIN = [(w, l) for w in TRAIN_W for l in TRAIN_L]
HOLDOUT = [(5, 5), (10, 5), (10, 10)]
ALL_GEOMS = sorted(TRAIN + HOLDOUT)


def tag(w, l):
    return f"W{w}_L{l}"


def load_sweep(w, l, sweep, device=None):
    df = pd.read_csv(os.path.join(CLEAN, f"{tag(w, l)}_{sweep}_clean.csv"))
    if device is not None:
        df = df[df.device == device]
    sort = ["VG"] if sweep != "output" else ["VG", "VD"]
    return df.sort_values(sort).reset_index(drop=True)


ON_LEVEL = 1e-10     # A: clearly above the ~3e-12 A instrument floor
ON_TOL = 0.05        # relative wiggle tolerated once on (instrument noise)


def onset(vg, i):
    """Thesis Sec. 3.2 (Von, Ioff): examine dIDS/dVGS and take the first
    point from which it stays positive -- where the transfer curve begins
    its rise and keeps rising to VH -- as the (Von, Ioff) point.
    Implemented by walking down from VH while the curve keeps falling. Once
    the device is on (|ID| > ON_LEVEL) a dip smaller than ON_TOL is
    instrument noise, not a sign change of the derivative; in the off-state
    the rule is applied strictly.
    Returns (Von, Ioff = |IDS(Von)|, index)."""
    vg = np.asarray(vg, float)
    a = np.abs(np.asarray(i, float))
    k = len(a) - 1
    while k > 0:
        tol = ON_TOL if a[k] > ON_LEVEL else 0.0
        if a[k - 1] < a[k] * (1.0 + tol):
            k -= 1
        else:
            break
    return float(vg[k]), float(a[k]), k


def max_rel_drop(y):
    """Largest fractional decrease between consecutive points of a curve
    that should be monotonically increasing."""
    y = np.asarray(y, float)
    run_max = np.maximum.accumulate(y)
    return float(np.max((run_max - y) / np.maximum(run_max, 1e-30)))


PARAMS = ["Von_lin", "Von_sat", "alpha_lin", "alpha_sat", "k_lin", "k_sat",
          "G0_lin", "G0_sat", "Ioff_lin", "Ioff_sat"]


def camcas_ids(vgs, vds, W, L, p):
    """Drain current of Appendix F (Listing F.1, lines 211-235), vectorized.
    W, L in um; p holds the ten regime parameters plus DeltaL (um), RSD
    (Ohm) and m -- scalars or arrays broadcastable against vgs."""
    vgs, vds = np.broadcast_arrays(np.asarray(vgs, float), np.asarray(vds, float))
    g = {k: np.broadcast_to(np.asarray(p[k], float), vgs.shape)
         for k in PARAMS + ["DeltaL", "RSD", "m"]}
    von_eff = np.maximum(g["Von_lin"], g["Von_sat"])
    lp = np.asarray(L, float) - g["DeltaL"]
    wl = np.asarray(W, float) / lp
    on = vgs > von_eff
    x = np.where(on, vgs - von_eff, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        G = g["G0_lin"] * wl * np.exp(g["k_lin"] * x ** g["alpha_lin"])
        i_lin = (G * vds + g["Ioff_lin"]) / (1.0 + g["RSD"] * G)
        i_sat = g["G0_sat"] * wl * np.exp(g["k_sat"] * x ** g["alpha_sat"]) * x + g["Ioff_sat"]
        i_on = (i_lin ** (-g["m"]) + i_sat ** (-g["m"])) ** (-1.0 / g["m"])
    i_off = np.where(vds < vgs - von_eff, g["Ioff_lin"], g["Ioff_sat"])
    return np.where(on, i_on, i_off)


def poly_LW(c, W, L):
    """Thesis Eqs. 3.5/3.6 as coded in Appendix F:
    X = (aW*W + a0)*L^2 + (bW*W + b0)*L + (cW*W + c0)."""
    return (c["aW"] * W + c["a0"]) * L ** 2 + (c["bW"] * W + c["b0"]) * L + (c["cW"] * W + c["c0"])


def lin_W(c, W):
    """Thesis Eqs. 3.17/3.18: X = s*W + i."""
    return c["s"] * W + c["i"]


def scaled_params(coef, W, L):
    """All model parameters at (W, L) um from a step-3 coefficient set, with
    exactly the Appendix F evaluation (RSD_func in kOhm -> Ohm)."""
    p = {k: poly_LW(coef["poly"][k], W, L) for k in PARAMS}
    p["DeltaL"] = lin_W(coef["DeltaL"], W)
    p["RSD"] = max(lin_W(coef["RSD_kohm"], W), 0.0) * 1e3
    p["m"] = coef["m"]
    return p
