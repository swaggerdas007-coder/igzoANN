"""Step 3 -- geometry-scaling relations with the thesis's functional forms
(Sec. 3.4, Eqs. 3.5-3.18, Appendix F), coefficients fitted to our data.

Two-step procedure exactly as in the thesis:
  1. for each width, fit every regime parameter as a quadratic in L,
     X = a1*L^2 + b1*L + c1 (thesis Eq. 3.5, Table 3.13) -- we have 4
     lengths per width, so this is a least-squares quadratic instead of an
     exact 3-point one;
  2. fit each of a1, b1, c1 as a straight line in W (thesis Eq. 3.6) -- 4
     widths instead of 2, again least squares.
Result per parameter: X(L,W) = (aW*W+a0)*L^2 + (bW*W+b0)*L + (cW*W+c0), the
six-coefficient form of von_lin_func ... Ioff_sat_func in Appendix F.
DeltaL(W) and RSD(W) are straight lines in W (Eqs. 3.17/3.18) through the
per-width TLM values. m is one global constant (Appendix F: parameter m),
the median of the per-device values.

On a full W x L grid, the two-step least squares is identical to a single
least-squares fit of the six coefficients (the design matrix is a
Kronecker product); the script asserts this. The coefficients are computed
with the single fit, which is the same estimator but stays well defined
when a grid point is missing (a width left with < 3 lengths has no
per-width quadratic). The per-width quadratics are still written out as
the analogue of the thesis's Table 3.13.

Only the 16 training geometries are used (holdout W5_L5, W10_L5, W10_L10
never enter). `fit(train, exclude=...)` lets step 6 study outlier devices.

Outputs: step3_coefficients_<variant>.json, step3_per_width_quadratics_<variant>.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from common import OUT, PARAMS, TRAIN, poly_LW


def two_step(t, p):
    per_w = []
    for w, g in t.groupby("W"):
        if len(g) < 3:
            continue
        a1, b1, c1 = np.polyfit(g.L, g[p], 2)
        per_w.append(dict(parameter=p, W=w, n_L=len(g), a1=a1, b1=b1, c1=c1))
    pw = pd.DataFrame(per_w)
    c = {}
    for k in ("a", "b", "c"):
        s, i = np.polyfit(pw.W, pw[f"{k}1"], 1)
        c[f"{k}W"], c[f"{k}0"] = float(s), float(i)
    return c, pw


def direct_lsq(t, p):
    X = np.c_[t.W * t.L ** 2, t.L ** 2, t.W * t.L, t.L, t.W, np.ones(len(t))]
    beta = np.linalg.lstsq(X, t[p], rcond=None)[0]
    return dict(zip(["aW", "a0", "bW", "b0", "cW", "c0"], map(float, beta)))


def load_table():
    return pd.read_csv(os.path.join(OUT, "step1_parameter_table.csv"))


def fit(table, exclude=(), params=None):
    """Fit the scaling coefficients on the training geometries minus
    `exclude`. `params` may override per-device values of some parameters
    (used by the sequential variant of step 6)."""
    t = table[[(w, l) in TRAIN and (w, l) not in exclude for w, l in zip(table.W, table.L)]].copy()
    if params is not None:
        for k, v in params.items():
            t[k] = v.loc[t.index]
    coef = {"poly": {}, "fit_devices": [[int(w), int(l)] for w, l in zip(t.W, t.L)]}
    quads = []
    full_grid = len(t) == 16
    for p in PARAMS:
        if not np.isfinite(t[p]).all():
            bad = t.loc[~np.isfinite(t[p]), ["W", "L"]].values.tolist()
            raise ValueError(f"{p} undefined for {bad}")
        c = direct_lsq(t, p)
        two, pw = two_step(t, p)
        if full_grid:
            assert all(np.isclose(two[k], c[k], rtol=1e-6, atol=1e-12 * max(1, abs(c[k])))
                       for k in c), p
        coef["poly"][p] = c
        quads.append(pw)
    tw = t.groupby("W")[["DeltaL", "RSD_kohm"]].first()
    for p in ("DeltaL", "RSD_kohm"):
        s, i = np.polyfit(tw.index, tw[p], 1)
        coef[p] = {"s": float(s), "i": float(i)}
    coef["m"] = float(np.median(t.m))
    return coef, pd.concat(quads, ignore_index=True)


def residual_report(table, coef):
    rows = []
    for r in table.itertuples():
        row = {"W": r.W, "L": r.L}
        for p in PARAMS:
            row[p] = poly_LW(coef["poly"][p], r.W, r.L) - getattr(r, p)
        rows.append(row)
    return pd.DataFrame(rows)


def save(coef, quads, variant):
    with open(os.path.join(OUT, f"step3_coefficients_{variant}.json"), "w") as f:
        json.dump(coef, f, indent=2)
    quads.to_csv(os.path.join(OUT, f"step3_per_width_quadratics_{variant}.csv"), index=False,
                 float_format="%.6g")


def main(variant="thesis", exclude=()):
    table = load_table()
    coef, quads = fit(table, exclude)
    coef["variant"] = variant
    save(coef, quads, variant)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(coef["poly"]).T.to_string(float_format=lambda v: f"{v:.4g}"))
    print("DeltaL(W) =", coef["DeltaL"], " RSD(W) [kOhm] =", coef["RSD_kohm"], " m =", coef["m"])
    print(residual_report(table, coef).to_string(index=False, float_format=lambda v: f"{v:+.3g}"))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["thesis"]))
