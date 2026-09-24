"""Fit a CAMCAS-style semi-empirical compact model to our measured IGZO TFT
data, following the "Cambridge Compact Analytical Semiconductor" unified
drain-current model described in Carolina de Almeida's thesis "Development
of IGZO thin-film transistors (TFTs) compact models" (2025), Sections 2.3.4
and Chapter 3, and implemented there in Verilog-A in Appendix F.

CAMCAS is a single analytical expression that covers both the linear and
saturation regimes at once, referenced to a "turn-on voltage" Von rather than
a classical threshold voltage:

    IDS = G0 * (W/L') * exp(kappa * (VGS - Von)^alpha) * VDS' + Ioff        (thesis Eq. 2.27)

with separate (Von, alpha, kappa, G0, Ioff) fitted independently in the
linear (VDS' = VDS - 2*Rc*IDS) and saturation (VDS' = VGS - Von) regimes, the
two branches combined with a harmonic-mean smoothing function (Eq. 2.34), and
each of the 10 regime parameters then re-fit as a scalable bivariate
polynomial function of channel length L and width W so the model covers the
whole (L, W) design space instead of only the discrete geometries that were
actually measured.

This script re-derives that whole pipeline against OUR dataset
(data_cleaned_2/, one QC-passed, positive-VT device per geometry) instead of
the thesis's own measurements:

  1. Per-geometry (W, L) extraction of (Von, Ioff) from the linear and
     saturation transfer curves (derivative-onset method, thesis Sec. 3.2).
  2. Per-width extraction of (DeltaL, RSD) via the TLM (transfer-length
     method) common-intersection-point technique (thesis Sec. 3.2 / Fig 3.1),
     using the linear-regime curves across all measured L for that W.
  3. Per-geometry extraction of (alpha, kappa, G0) in both regimes via the
     log-log U-function linearization (thesis Eqs. 3.3/3.4 and 2.32/2.33).
  4. A single least-squares bivariate quadratic-in-L, linear-in-W fit
     (thesis Eqs. 3.5/3.6, same functional FORM as the six coefficients baked
     into Appendix F's von_lin_func etc.) of each of the 10 regime parameters
     over ALL measured geometries at once.

We have 19 (W, L) geometries (W in {5,10,20,40,80,160} um, L in {5,10,15,20}
um) versus the thesis's own Dataset 3 (only 2 widths x 3 lengths = 6 points),
so step 4 here is an ordinary-least-squares fit over up to 19 points per
parameter instead of the thesis's brittle "quadratic slice per W, then a
2-point linear fit across W" construction -- same polynomial form, sturdier
fit.

NOTE on RSD units: the current-current-loop term in the drain current
(Ids_lin = (G*Vds + Ioff)/(1 + RSD*G)) requires RSD*G dimensionless, i.e. RSD
in Ohms. The thesis's own Appendix F divides its fitted RSD by 100 before use,
which does not correspond to a clean kOhm->Ohm conversion (that would be
*1000); we suspect it is an ad hoc correction, possibly compensating for a
scientific-notation transcription slip in the PDF (Appendix F's overlap
capacitance expressions have similar dimensionally-odd literals, e.g.
`Lov = L*10e-5`, `RSD_func = -0.0076*Wm + 1.51` labeled kOhm then divided by
100). Rather than guess at that hack, we extract RSD directly in Ohms from
the TLM intersection (Rtot = VDS/IDS is already in Ohms) and use it as-is.

Outputs (outputs_camcas/):
  - extracted_params.csv     per-geometry raw extraction (for inspection)
  - dl_rsd_per_width.csv     per-width (DeltaL, RSD) TLM extraction
  - camcas_coefficients.json the scalable model: 10 bivariate polynomials
                              + DeltaL(W), RSD(W) linear fits + assumptions

Usage:
    python scripts/fit_camcas_model.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from sklearn.linear_model import RidgeCV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data_cleaned_2")
OUT_DIR = os.path.join(REPO, "outputs_camcas")

VH = 5.0          # reference gate voltage for G0 extraction: max VG in our sweeps
VDS_LIN = 0.1     # linear-regime bias
VDS_SAT = 5.0     # saturation-regime bias

# Parameters modeled as a bivariate quadratic-in-L, linear-in-W polynomial,
# X(L, W) = (a*W + b)*L^2 + (c*W + d)*L + (e*W + f), matching the FORM of
# thesis Eqs. 3.7-3.16 / Appendix F's von_lin_func etc., but fit here by
# ordinary least squares over all 19 measured geometries at once.
SCALABLE_PARAMS = [
    "Von_lin", "Von_sat",
    "alpha_lin", "alpha_sat",
    "kappa_lin", "kappa_sat",
    "G0_lin", "G0_sat",
    "Ioff_lin", "Ioff_sat",
]


def list_geometries():
    geoms = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_linear_best.csv"))):
        base = os.path.basename(path)
        wl = base[:-len("_linear_best.csv")]
        w_str, l_str = wl.split("_L")
        W = float(w_str[1:])
        L = float(l_str)
        geoms.append((W, L))
    return geoms


def load_curve(W, L, kind):
    path = os.path.join(DATA_DIR, f"W{int(W)}_L{int(L)}_{kind}_best.csv")
    df = pd.read_csv(path)
    df = df.sort_values("VG").reset_index(drop=True)
    return df


def extract_von_ioff(vg, ids):
    """Derivative-onset extraction of (Von, Ioff): search for the point where
    dIDS/dVGS first turns, and stays, positive after the off-state floor
    (thesis Sec. 3.2). Search starts at the curve's minimum so a residual
    off-state wiggle at the most negative VG can't be mistaken for turn-on."""
    vg = np.asarray(vg, dtype=float)
    ids = np.asarray(ids, dtype=float)
    off_idx = int(np.argmin(ids))
    vg_c, id_c = vg[off_idx:], ids[off_idx:]

    dV = np.diff(vg_c)
    dI = np.diff(id_c)
    deriv = dI / dV
    k = 3
    smooth = np.convolve(deriv, np.ones(k) / k, mode="valid")

    onset = 0
    for i in range(len(smooth) - 2):
        if smooth[i] > 0 and smooth[i + 1] > 0 and smooth[i + 2] > 0:
            onset = i
            break
    # `smooth[i]` sits at raw index i + (k // 2) relative to deriv/id_c
    idx = min(onset + k // 2, len(id_c) - 1)
    return float(vg_c[idx]), float(id_c[idx])


DELTA_L_BOUNDS = (0.0, 4.0)     # um -- a modest fraction of our smallest L (5 um)
RSD_BOUNDS = (10.0, 1.0e5)      # Ohms


def tlm_intersection(slopes, intercepts, min_L):
    """Bounded least-squares common intersection point of lines
    R = m_i*L + c_i. Returns (L*, R*) -- thesis's (DeltaL, RSD), Sec. 3.2 /
    Fig 3.1.

    Our per-width Rtot(L) trend at fixed VGS is much shallower/noisier than
    the thesis's own (the intrinsic channel resistance is small next to the
    parasitic/contact resistance here), so the unconstrained intersection is
    ill-conditioned and can land far outside any physical range (channel
    length "reductions" bigger than the channel itself). We therefore solve
    the same least-squares objective but bounded to a physically plausible
    box -- DeltaL a modest fraction of the smallest measured L, RSD a
    positive, sub-100 kOhm parasitic resistance -- rather than trusting an
    unconstrained 2x2 solve on a nearly-singular problem. Where the data
    does carry a clean signal this recovers the same answer as the
    unconstrained solve; where it doesn't, it degrades gracefully to a
    boundary value instead of a nonphysical one.
    """
    m = np.asarray(slopes, dtype=float)
    c = np.asarray(intercepts, dtype=float)

    def resid(p):
        L_, R_ = p
        return R_ - m * L_ - c

    lo = [DELTA_L_BOUNDS[0], RSD_BOUNDS[0]]
    hi = [min(DELTA_L_BOUNDS[1], 0.8 * min_L), RSD_BOUNDS[1]]
    p0 = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]
    result = least_squares(resid, p0, bounds=(lo, hi))
    return float(result.x[0]), float(result.x[1])


def extract_dl_rsd(W, geoms_for_w):
    """TLM extraction of (DeltaL, RSD) for one channel width, using the
    linear-regime curves across every measured L at that width."""
    curves = {L: load_curve(W, L, "linear") for L in geoms_for_w}
    # Probe well into the on-state (VGS in [3,5]) and require a current well
    # above the noise floor at every L, so Rtot = VDS/IDS isn't dominated by
    # near-off-state noise (which is what produced kOhm-scale garbage in an
    # earlier version of this extraction).
    vgs_grid = np.linspace(3.0, 5.0, 5)
    slopes, intercepts = [], []
    for vgs in vgs_grid:
        Ls, Rs = [], []
        for L, df in curves.items():
            idx = (df["VG"] - vgs).abs().idxmin()
            ids = df.loc[idx, "ID"]
            if ids <= 1e-8:
                continue
            Ls.append(L)
            Rs.append(VDS_LIN / ids)
        if len(Ls) < 2:
            continue
        a, b = np.polyfit(Ls, Rs, 1)
        slopes.append(a)
        intercepts.append(b)
    if len(slopes) < 2:
        return None
    return tlm_intersection(slopes, intercepts, min_L=min(geoms_for_w))


def fit_alpha_kappa_G0(vg, ids, von, ioff, W, Lp, vds_prime_of):
    """Extraction of (alpha, kappa, G0) for one regime of one geometry.

    The thesis linearizes Eq. 2.27 into the log-log U-function of Eqs.
    3.3/3.4, which needs a numerical d/dVGS of an already-noisy measured
    current -- fine on the thesis's own smooth Dataset 3 but far too noise-
    amplifying on our data (an early version of this script using that
    method produced the same kind of "non-physical" alpha/kappa/G0 values
    the thesis itself reports for its noisier Dataset 2). Instead we fit
    (alpha, kappa, G0) directly by nonlinear least squares against Eq. 2.27
    itself,

        I' = G0 * (W/Lp) * exp(kappa * (VGS - Von)^alpha) * VDS'

    with VDS' computed from the *measured* IDS (thesis's own linear-regime
    formulation, Eq. 2.28, does the same rather than solving self-
    consistently). This is mathematically the same model, just fit without
    an intermediate derivative.

    I' spans up to ~5 decades over the on-state samples of a single curve, so
    the fit is done in log space (ln I' vs. the model's ln), exactly as the
    rest of this repo's ANN regresses log10(|ID|) rather than raw ID for the
    same reason (src/dataset.py) -- an ordinary-scale least squares fit is
    dominated entirely by the highest-current points and leaves the
    optimizer with essentially no gradient in (alpha, kappa) at all (this
    was confirmed to get every geometry stuck at its initial guess). A small
    multi-start grid over the (alpha, kappa) initial guess guards against
    local optima in this non-convex fit.
    """
    vg = np.asarray(vg, dtype=float)
    ids = np.asarray(ids, dtype=float)
    mask = vg > von + 0.05
    vg, ids = vg[mask], ids[mask]
    iprime = ids - ioff
    vds_prime = np.array([vds_prime_of(v, i) for v, i in zip(vg, ids)])

    valid = (iprime > 0) & (vds_prime > 0)
    vg, iprime, vds_prime = vg[valid], iprime[valid], vds_prime[valid]
    if len(vg) < 6:
        return None

    x = vg - von
    wl = W / Lp
    ln_target = np.log(iprime)
    ln_wl_vds = np.log(wl) + np.log(vds_prime)

    def ln_model(_, ln_g0, kappa, alpha):
        expo = np.clip(kappa * np.power(x, alpha), -700.0, 700.0)
        return ln_g0 + ln_wl_vds + expo

    bounds = ([-60.0, -300.0, -3.0], [20.0, 300.0, 3.0])
    best = None
    for alpha0 in (-0.1, -0.3, -0.5, -1.0):
        for kappa0 in (-5.0, -10.0, -20.0, -50.0):
            expo0 = np.clip(kappa0 * np.power(x, alpha0), -700.0, 700.0)
            ln_g0_0 = np.median(ln_target - ln_wl_vds - expo0)
            if not np.isfinite(ln_g0_0):
                continue
            ln_g0_0 = float(np.clip(ln_g0_0, bounds[0][0] + 1e-6, bounds[1][0] - 1e-6))
            p0 = [ln_g0_0, kappa0, alpha0]
            try:
                popt, _ = curve_fit(ln_model, x, ln_target, p0=p0, bounds=bounds, maxfev=20000)
            except RuntimeError:
                continue
            sse = np.sum((ln_model(None, *popt) - ln_target) ** 2)
            if best is None or sse < best[0]:
                best = (sse, popt)
    if best is None:
        return None

    ln_g0, kappa, alpha = best[1]
    return float(alpha), float(kappa), float(np.exp(ln_g0))


# G0 and Ioff span several decades across geometries even after outlier
# rejection (same lesson as the per-geometry alpha/kappa/G0 fit above): an
# ordinary-scale polynomial surface fit is dominated by the largest values
# and extrapolates to nonsense (even negative "currents") elsewhere. Fit
# these two in log space instead; Von/alpha/kappa are all O(1)-O(100) and
# fit fine directly.
LOG_PARAMS = {"G0_lin", "G0_sat", "Ioff_lin", "Ioff_sat"}


def fit_bivariate(L, W, X, log=False):
    """Fit of X = (a*W+b)*L^2 + (c*W+d)*L + (e*W+f) i.e. columns
    [L^2*W, L^2, L*W, L, W, 1] -> reordered to match thesis' (a1,b1,c1)
    per-slice form, but as one direct 6-coefficient regression. If log=True,
    the fit (and the coefficients) are in ln(X) space.

    The [L^2*W, L^2, L*W, L, W] design matrix is badly conditioned here
    (condition number ~1e5-1e6: L^2*W spans 5-64000 while W spans 5-160,
    and our 12-19 geometries don't fill a regular (L,W) grid), so an
    ordinary least-squares solve amplifies the real scatter in the per-
    geometry extracted parameters into wild, non-physical swings between
    the actual measured geometries -- confirmed by comparing the fitted
    surface back against its own training points. We therefore standardize
    the features and ridge-regularize (regularization strength picked by
    leave-one-out CV via RidgeCV), which damps exactly those poorly-
    determined, noise-amplifying directions while leaving the well-
    determined ones essentially untouched.
    """
    L = np.asarray(L, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.log(X) if log else X
    A = np.stack([W * L ** 2, L ** 2, W * L, L, W], axis=1)

    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std[std == 0] = 1.0
    A_scaled = (A - mean) / std

    alphas = np.logspace(-3, 3, 25)
    ridge = RidgeCV(alphas=alphas, fit_intercept=True)
    ridge.fit(A_scaled, y)

    coef_raw = ridge.coef_ / std
    intercept_raw = ridge.intercept_ - float(np.dot(ridge.coef_, mean / std))
    a, b, c, d, e = coef_raw
    f = intercept_raw
    return {"a": float(a), "b": float(b), "c": float(c),
            "d": float(d), "e": float(e), "f": float(f), "log": log}


def eval_bivariate(coef, L, W):
    val = ((coef["a"] * W + coef["b"]) * L ** 2
           + (coef["c"] * W + coef["d"]) * L
           + (coef["e"] * W + coef["f"]))
    return np.exp(val) if coef.get("log") else val


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    geoms = list_geometries()
    print(f"Found {len(geoms)} geometries: {geoms}")

    # ---- Step 1: Von/Ioff per geometry, both regimes ----
    rows = []
    curves_lin, curves_sat = {}, {}
    for (W, L) in geoms:
        df_lin = load_curve(W, L, "linear")
        df_sat = load_curve(W, L, "saturation")
        curves_lin[(W, L)] = df_lin
        curves_sat[(W, L)] = df_sat
        von_lin, ioff_lin = extract_von_ioff(df_lin["VG"], df_lin["ID"])
        von_sat, ioff_sat = extract_von_ioff(df_sat["VG"], df_sat["ID"])
        rows.append({"W": W, "L": L, "Von_lin": von_lin, "Ioff_lin": ioff_lin,
                      "Von_sat": von_sat, "Ioff_sat": ioff_sat})
    von_df = pd.DataFrame(rows)

    # ---- Step 2: DeltaL, RSD per width (TLM), then fit vs W ----
    by_w = {}
    for (W, L) in geoms:
        by_w.setdefault(W, []).append(L)
    dl_rsd_rows = []
    for W, Ls in sorted(by_w.items()):
        if len(Ls) < 2:
            continue
        result = extract_dl_rsd(W, Ls)
        if result is None:
            continue
        dl, rsd = result
        dl_rsd_rows.append({"W": W, "DeltaL": dl, "RSD_ohm": rsd, "n_L": len(Ls)})
    dl_rsd_df = pd.DataFrame(dl_rsd_rows)
    print("\nPer-width DeltaL/RSD (TLM extraction, after sanity filtering):")
    print(dl_rsd_df.to_string(index=False))
    if len(dl_rsd_df) < 2:
        raise RuntimeError("Not enough widths with a physically sane TLM "
                            "(DeltaL, RSD) extraction to fit DeltaL(W)/RSD(W).")

    dl_fit = np.polyfit(dl_rsd_df["W"], dl_rsd_df["DeltaL"], 1)
    rsd_fit = np.polyfit(dl_rsd_df["W"], dl_rsd_df["RSD_ohm"], 1)
    deltaL_of = lambda W: np.polyval(dl_fit, W)
    rsd_of = lambda W: np.polyval(rsd_fit, W)

    # ---- Step 3: alpha, kappa, G0 per geometry, both regimes ----
    for i, row in von_df.iterrows():
        W, L = row["W"], row["L"]
        Lp = L - deltaL_of(W)
        rc = rsd_of(W) / 2.0

        df_lin = curves_lin[(W, L)]
        lin_result = fit_alpha_kappa_G0(
            df_lin["VG"], df_lin["ID"], row["Von_lin"], row["Ioff_lin"], W, Lp,
            vds_prime_of=lambda v, i_ds: VDS_LIN - 2 * rc * i_ds,
        )

        df_sat = curves_sat[(W, L)]
        sat_result = fit_alpha_kappa_G0(
            df_sat["VG"], df_sat["ID"], row["Von_sat"], row["Ioff_sat"], W, Lp,
            vds_prime_of=lambda v, i_ds, von=row["Von_sat"]: v - von,
        )

        if lin_result:
            von_df.loc[i, ["alpha_lin", "kappa_lin", "G0_lin"]] = lin_result
        if sat_result:
            von_df.loc[i, ["alpha_sat", "kappa_sat", "G0_sat"]] = sat_result

    von_df = von_df.dropna().reset_index(drop=True)
    print(f"\n{len(von_df)}/{len(geoms)} geometries yielded a full parameter set:")
    print(von_df.to_string(index=False))
    von_df.to_csv(os.path.join(OUT_DIR, "extracted_params.csv"), index=False)
    dl_rsd_df.to_csv(os.path.join(OUT_DIR, "dl_rsd_per_width.csv"), index=False)

    # ---- Step 4: scalable bivariate fit of each parameter vs (L, W) ----
    # Per-parameter outlier rejection first: the direct NLS fit above is far
    # more stable than the log-log derivative method, but a handful of
    # geometries can still land on a degenerate (non-physical) optimum --
    # the thesis hits the same issue on its own noisier datasets and
    # excludes individual devices by hand (Sec. 3.3.2). We instead reject,
    # independently per parameter (a geometry can be fine in one regime and
    # bad in another), any value outside a generous physically-plausible
    # band before fitting the (L, W) surface, so a handful of bad geometries
    # can't dominate the least-squares fit for every parameter.
    PHYSICAL_BOUNDS = {
        "Von_lin": (-5, 5), "Von_sat": (-5, 5),
        "alpha_lin": (-3, 3), "alpha_sat": (-3, 3),
        "kappa_lin": (-200, 200), "kappa_sat": (-200, 200),
        "G0_lin": (1e-10, 1), "G0_sat": (1e-10, 1),
        # strictly positive floor (not 0): both go into a log-space fit below
        "Ioff_lin": (1e-15, 1e-8), "Ioff_sat": (1e-15, 1e-8),
    }
    coefficients = {}
    for name in SCALABLE_PARAMS:
        lo, hi = PHYSICAL_BOUNDS[name]
        sub = von_df[(von_df[name] >= lo) & (von_df[name] <= hi)]
        if len(sub) < 8:
            print(f"WARNING: only {len(sub)} geometries survive filtering "
                  f"for {name}; using all {len(von_df)} unfiltered instead.")
            sub = von_df[von_df[name] > 0] if name in LOG_PARAMS else von_df
        log = name in LOG_PARAMS
        coef = fit_bivariate(sub["L"], sub["W"], sub[name], log=log)
        # R^2 computed in whichever space was actually fit (log for G0/Ioff)
        target = np.log(sub[name].to_numpy()) if log else sub[name].to_numpy()
        pred = eval_bivariate(coef, sub["L"].to_numpy(), sub["W"].to_numpy())
        pred_space = np.log(pred) if log else pred
        resid = target - pred_space
        r2 = 1 - np.sum(resid ** 2) / np.sum((target - target.mean()) ** 2)
        coefficients[name] = coef
        print(f"{name:10s} n={len(sub):2d}/{len(von_df)}  R^2={r2:7.4f}  coeffs={coef}")

    out = {
        "source": "data_cleaned_2 (one QC-passed, positive-VT device per geometry)",
        "reference": "Carolina de Almeida, 'Development of IGZO TFTs compact models' "
                     "(2025), CAMCAS model (Sec. 2.3.4, Ch. 3, Appendix F)",
        "VH": VH, "VDS_lin": VDS_LIN, "VDS_sat": VDS_SAT,
        "n_geometries_fit": len(von_df),
        "DeltaL_of_W": {"a": float(dl_fit[0]), "b": float(dl_fit[1])},
        "RSD_ohm_of_W": {"a": float(rsd_fit[0]), "b": float(rsd_fit[1])},
        "params": coefficients,
    }
    with open(os.path.join(OUT_DIR, "camcas_coefficients.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'camcas_coefficients.json')}")


if __name__ == "__main__":
    main()
