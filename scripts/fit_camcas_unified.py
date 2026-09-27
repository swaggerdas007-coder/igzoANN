"""Fit ONE unified CAMCAS parameter set (scripts/camcas_unified.py) jointly to
the "golden" subset of data_cleaned_2 devices -- the ones that agree with the
population consensus -- rather than per device or as (L, W) polynomial
surfaces.

Golden-device selection (all computed on W/L-normalized curves, so geometry
itself doesn't count against a device):
  * turn-on voltage within +-0.75 V of the population median, and
  * normalized on-current at VG = 5 V within 2x of the median, in BOTH the
    linear (VD = 0.1 V) and saturation (VD = 5 V) transfer sweeps.
Devices failing this are atypical die sites (late turn-on / low mobility),
not a geometry trend -- fitting them would pull a single parameter set away
from the process's typical behavior.

The fit is end-to-end: the full unified model (both branches + harmonic-mean
blend) is evaluated at each measured bias point of the linear transfer,
saturation transfer AND output-curve sweeps, and the residual is taken in
log10|ID| (the same target space as this repo's ANN). Each (device, sweep)
group carries equal total weight so the 561-point output families don't
drown out the 71-point transfer curves. A soft-L1 loss keeps off-state noise
spikes from dominating.

Also runs leave-one-device-out cross-validation over the golden set -- refit
without device k, predict device k -- which is the real test of a compact
model: accuracy at a geometry it never saw.

Usage:
    python scripts/fit_camcas_unified.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camcas_unified import VH, UnifiedParams, ids  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_cleaned_2", "merged_ann_dataset.csv")
OUT_DIR = os.path.join(REPO, "outputs_camcas")

VT_TOL = 0.75          # V
CURRENT_TOL_DEC = np.log10(2.0)

BOUNDS = {
    "Von_lin": (-6.0, 3.0), "Von_sat": (-6.0, 3.0),
    "lnGH_lin": (-40.0, 0.0), "lnGH_sat": (-40.0, 0.0),
    "s_lin": (0.05, 20.0), "s_sat": (0.05, 20.0),
    "alpha_lin": (-8.0, -0.005), "alpha_sat": (-8.0, -0.005),
    "Rsw": (0.0, 1e7),
    "m": (1.0, 8.0),
    "lnIoff": (np.log(1e-14), np.log(1e-10)),
}


def load():
    df = pd.read_csv(DATA)
    df["geom"] = list(zip(df.W, df.L))
    return df


def turn_on(vg, i, W, L, thr):
    """VG where |ID| first rises through thr * W/L, searched from the curve
    minimum on (same constant-current definition as data_cleaned_2)."""
    order = np.argsort(vg)
    vg, i = vg[order], np.abs(i[order])
    k0 = int(np.argmin(i))
    above = np.nonzero(i[k0:] > thr * W / L)[0]
    return vg[k0 + above[0]] if len(above) else np.nan


def device_features(df):
    rows = []
    for (W, L), g in df.groupby("geom"):
        lin = g[g.sweep == "linear"]
        sat = g[g.sweep == "saturation"]
        at5 = lambda s: s.loc[(s.VG - VH).abs().idxmin(), "abs_ID"] * L / W
        rows.append({
            "W": W, "L": L,
            "vt_lin": turn_on(lin.VG.values, lin.ID.values, W, L, 1e-10),
            "vt_sat": turn_on(sat.VG.values, sat.ID.values, W, L, 1e-9),
            "log_lin5_norm": np.log10(at5(lin)),
            "log_sat5_norm": np.log10(at5(sat)),
        })
    return pd.DataFrame(rows)


def select_golden(feat):
    med = feat[["vt_lin", "vt_sat", "log_lin5_norm", "log_sat5_norm"]].median()
    ok = ((feat.vt_lin - med.vt_lin).abs() <= VT_TOL) \
        & ((feat.vt_sat - med.vt_sat).abs() <= VT_TOL) \
        & ((feat.log_lin5_norm - med.log_lin5_norm).abs() <= CURRENT_TOL_DEC) \
        & ((feat.log_sat5_norm - med.log_sat5_norm).abs() <= CURRENT_TOL_DEC)
    feat = feat.copy()
    feat["golden"] = ok
    return feat


def tlm_delta_l(df, golden):
    """Width-normalized TLM over all golden devices at VG = VH: Rtot*W vs L
    is linear, and its L-axis intercept is -DeltaL. The intercept drifts with
    VG (the access region is gate-controlled, so no single common
    intersection exists), so take it at full gate drive, where the access
    region is closest to fully on -- the thesis's own convention of probing
    the TLM deep in the on-state."""
    lin = df[(df.sweep == "linear") & df.geom.isin(golden) & np.isclose(df.VG, VH)]
    r_w = lin.VD.values / lin.ID.values * lin.W.values
    slope, intercept = np.polyfit(lin.L.values, r_w, 1)
    return float(-intercept / slope)


def group_weights(df):
    n = df.groupby(["W", "L", "sweep"])["VG"].transform("size").to_numpy()
    return 1.0 / np.sqrt(n)


def initial_guesses():
    base = dict(Von_lin=0.0, Von_sat=0.3, lnGH_lin=np.log(1e-5), lnGH_sat=np.log(2e-6),
                s_lin=2.0, s_sat=2.0, alpha_lin=-0.5, alpha_sat=-0.5,
                Rsw=2e5, m=2.2, lnIoff=np.log(3e-12))
    for von in (-0.5, 0.0, 0.5):
        for alpha in (-0.2, -0.8):
            for s in (1.0, 4.0):
                yield dict(base, Von_lin=von, Von_sat=von + 0.3, alpha_lin=alpha,
                           alpha_sat=alpha, s_lin=s, s_sat=s)


def fit(df, delta_l):
    free = list(BOUNDS)
    lo = np.array([BOUNDS[n][0] for n in free])
    hi = np.array([BOUNDS[n][1] for n in free])
    w = group_weights(df)
    vg, vd, W, L, y = (df[c].to_numpy() for c in ("VG", "VD", "W", "L", "log_ID"))

    def params(v):
        return UnifiedParams(DeltaL=delta_l, **dict(zip(free, v)))

    def resid(v):
        return w * (np.log10(ids(vg, vd, W, L, params(v))) - y)

    best = None
    for guess in initial_guesses():
        v0 = np.clip([guess[n] for n in free], lo + 1e-9, hi - 1e-9)
        r = least_squares(resid, v0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3,
                          x_scale="jac", max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    return params(best.x)


def log_metrics(pred, log_true):
    err = np.log10(pred) - log_true
    return {
        "mae_dec": float(np.mean(np.abs(err))),
        "rmse_dec": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((log_true - log_true.mean()) ** 2)),
    }


def main():
    df = load()
    feat = select_golden(device_features(df))
    golden = [(r.W, r.L) for r in feat.itertuples() if r.golden]
    print(feat.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nGolden set ({len(golden)}): {golden}")

    train = df[df.geom.isin(golden)].reset_index(drop=True)
    delta_l = tlm_delta_l(df, golden)
    print(f"TLM DeltaL (golden set, VG={VH} V) = {delta_l:.3f} um")
    params = fit(train, delta_l)
    print("\nUnified parameters:")
    for k, v in vars(params).items():
        print(f"  {k:10s} = {v:.6g}")
    print("Thesis-form equivalents:", params.thesis_form())

    pred = ids(train.VG, train.VD, train.W, train.L, params)
    print("Golden-set fit (log10 ID):", log_metrics(pred, train.log_ID.values))

    print("\nLeave-one-device-out CV over the golden set:")
    loo_rows = []
    for geom in golden:
        sub = train[train.geom != geom]
        held = train[train.geom == geom]
        p_k = fit(sub, tlm_delta_l(sub, [g for g in golden if g != geom]))
        m = log_metrics(ids(held.VG, held.VD, held.W, held.L, p_k), held.log_ID.values)
        m_in = log_metrics(ids(held.VG, held.VD, held.W, held.L, params), held.log_ID.values)
        loo_rows.append({"W": geom[0], "L": geom[1], "heldout_mae_dec": m["mae_dec"],
                         "insample_mae_dec": m_in["mae_dec"]})
        print(f"  W={geom[0]:>3} L={geom[1]:>2}  held-out MAE={m['mae_dec']:.3f} dec"
              f"  (in-sample {m_in['mae_dec']:.3f})")
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(os.path.join(OUT_DIR, "unified_loo_cv.csv"), index=False)
    print(f"  mean held-out MAE = {loo.heldout_mae_dec.mean():.3f} dec, "
          f"in-sample = {loo.insample_mae_dec.mean():.3f} dec")

    feat.to_csv(os.path.join(OUT_DIR, "unified_device_selection.csv"), index=False)
    with open(os.path.join(OUT_DIR, "camcas_unified_params.json"), "w") as f:
        json.dump({
            "model": "unified CAMCAS (scripts/camcas_unified.py)",
            "VH": VH,
            "golden_devices": [list(g) for g in golden],
            "params": vars(params),
            "thesis_form": params.thesis_form(),
            "loo_cv_mean_heldout_mae_dec": float(loo.heldout_mae_dec.mean()),
        }, f, indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'camcas_unified_params.json')}")


if __name__ == "__main__":
    main()
