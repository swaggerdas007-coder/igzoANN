"""Global refinement of the scaling coefficients (used by step 6): the
functional forms are untouched (six-coefficient quadratic-in-L x
linear-in-W for Von, alpha, kappa, G0 of both regimes); only their
coefficients are adjusted so that the measured currents of the fit devices
are reproduced, starting from the thesis-chain coefficients.

Kept fixed at their extracted/scaled values: Ioff_lin/sat(L, W) (off-state
floor), DeltaL(W), RSD(W) (TLM) and m (Eq. 2.35).

Objective (least squares, soft-L1):
  * log10|ID| error at every on-state point (|ID| > 1e-10 A) of the fit
    devices' linear-transfer, saturation-transfer and output sweeps, each
    (device, sweep) carrying equal total weight;
  * REG * (p(W_k, L_k) - p_extracted_k) / sd(p) for every fit device k and
    parameter p -- keeps every parameter close to what the thesis procedure
    extracted for that device (G0 in log10), so the alpha-kappa-G0
    trade-off cannot drift to unphysical combinations;
  * validity over the whole design range W = 20..160, L = 5..20 (dense
    grid): G0 > 0, kappa < 0, alpha < 0 -- the conditions under which
    G0*exp(kappa*x^alpha) is a positive, increasing conductance.
"""
import numpy as np
from scipy.optimize import least_squares

from common import camcas_ids, lin_W, poly_LW

FREE = ["Von_lin", "Von_sat", "alpha_lin", "alpha_sat", "k_lin", "k_sat", "G0_lin", "G0_sat"]
KEYS = ["aW", "a0", "bW", "b0", "cW", "c0"]
REG = 0.1
GRID_W, GRID_L = np.meshgrid(np.linspace(20, 160, 15), np.linspace(5, 20, 16))


def _basis(W, L):
    return np.stack([W * L ** 2, L ** 2, W * L, L, W, np.ones_like(W)], axis=-1)


def refine(coef, points, anchors):
    """coef: step-3 style coefficient dict (start); points: DataFrame of
    fit-device on-state points (W, L, sweep, VG, VD, ID); anchors: DataFrame
    (W, L, <param>...) of extracted per-device values to stay near."""
    W, L = points.W.to_numpy(float), points.L.to_numpy(float)
    vg, vd = points.VG.to_numpy(float), points.VD.to_numpy(float)
    y = np.log10(np.abs(points.ID.to_numpy(float)))
    n = points.groupby(["W", "L", "sweep"]).VG.transform("size").to_numpy()
    wts = 1.0 / np.sqrt(n)
    B = _basis(W, L)
    fixed = {"Ioff_lin": poly_LW(coef["poly"]["Ioff_lin"], W, L),
             "Ioff_sat": poly_LW(coef["poly"]["Ioff_sat"], W, L),
             "DeltaL": lin_W(coef["DeltaL"], W),
             "RSD": np.maximum(lin_W(coef["RSD_kohm"], W), 0.0) * 1e3, "m": coef["m"]}
    aW, aL = anchors.W.to_numpy(float), anchors.L.to_numpy(float)
    Ba = _basis(aW, aL)
    Bg = _basis(GRID_W.ravel(), GRID_L.ravel())
    sd = {p: (np.std(np.log10(anchors[p])) if p.startswith("G0") else np.std(anchors[p]))
          for p in FREE}
    target = {p: (np.log10(anchors[p].to_numpy()) if p.startswith("G0") else anchors[p].to_numpy())
              for p in FREE}
    # column scaling so every coefficient moves on a comparable scale
    colscale = 1.0 / np.abs(Bg).max(axis=0)
    x0 = np.concatenate([np.array([coef["poly"][p][k] for k in KEYS]) / colscale for p in FREE])

    def unpack(x):
        return {p: x[6 * i:6 * i + 6] * colscale for i, p in enumerate(FREE)}

    def resid(x):
        c = unpack(x)
        prm = dict(fixed, **{p: B @ c[p] for p in FREE})
        with np.errstate(all="ignore"):
            pred = np.log10(camcas_ids(vg, vd, W, L, prm))
        r_data = np.nan_to_num(wts * (pred - y), nan=5.0, posinf=5.0, neginf=5.0)
        r_reg = []
        for p in FREE:
            v = Ba @ c[p]
            if p.startswith("G0"):
                v = np.log10(np.maximum(v, 1e-12))
            r_reg.append(REG * (v - target[p]) / sd[p])
        r_val = []
        for p in FREE:
            v = Bg @ c[p]
            if p.startswith("G0"):
                r_val.append(10.0 * np.maximum(0.0, 1e-7 - v) / 1e-7)
            elif p.startswith(("k_", "alpha")):
                r_val.append(10.0 * np.maximum(0.0, v + (0.5 if p.startswith("k") else 0.02)))
        return np.concatenate([r_data] + r_reg + r_val)

    res = least_squares(resid, x0, loss="soft_l1", f_scale=0.3, max_nfev=3000, x_scale="jac")
    out = {"poly": {k: dict(v) for k, v in coef["poly"].items()}}
    for p, v in unpack(res.x).items():
        out["poly"][p] = dict(zip(KEYS, map(float, v)))
    for k in coef:
        if k != "poly":
            out[k] = coef[k]
    out["refine_cost"] = float(res.cost)
    return out
