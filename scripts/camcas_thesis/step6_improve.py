"""Step 6 -- does one CAMCAS model work across the geometry space, and if
not, why? Builds and scores coefficient variants that all keep the thesis
equations and scaling forms, differing only in HOW the coefficients are
obtained, then picks the final one by leave-one-geometry-out cross-validation
over the fit geometries (the 3 held-out geometries are scored afterwards,
never used to choose).

Variants
  thesis        independent two-step fit of every parameter (step 3 as-is)
  thesis_x      same, W160_L20 excluded from the coefficient fit
  sequential    the thesis's extraction chain (Sec. 3.2: Von -> alpha,kappa
                -> G0 "using the previously extracted values of alpha and
                kappa") applied to the SCALED values, so each parameter is
                re-extracted from the measured curve given the fitted
                scaling of the ones before it:
                  1. Von_lin, Von_sat, Ioff, DeltaL(W), RSD(W) as in step 3
                  2. alpha from the U-function (Eqs. 3.3/3.4) at the scaled
                     Von_eff, scaled RSD -> fit alpha(L, W)
                  3. kappa from the same U-regression with its slope held at
                     1 - alpha(L, W) -> fit kappa(L, W)
                  4. G0 from Eqs. 2.32/2.33 with the scaled Von, alpha,
                     kappa, DeltaL, RSD -> fit G0(L, W)
                  5. m (Eq. 2.35) re-extracted with the scaled branches
  sequential_x  same, W160_L20 excluded from the coefficient fit
  sequential_k  sequential, plus one more link of the same chain:
                  4b. kappa re-extracted from Eqs. 2.32/2.33 solved for kappa
                      with G0 at its scaled value, -> refit kappa(L, W).
                G0 alone is not smooth in L (it compensates exp(kappa*xH^alpha)
                as alpha varies), but ln G(VH) is, and kappa = (ln G(VH) -
                ln G0)/xH^alpha inherits that smoothness.
  sequential_k_x  same, W160_L20 excluded
  *_g           the variant above followed by a global refinement of the
                Von/alpha/kappa/G0 scaling coefficients against the fit
                devices' measured currents (refine.py): same functional
                forms, parameters held near their extracted values, G0 > 0
                and kappa, alpha < 0 over the whole W/L range.

Usage: python step6_improve.py
Outputs: step6_variant_comparison.csv, final_variant.txt, the final
verilogA/tft_camcas_thesis.va and its step-5 validation
"""
import json
import os

import numpy as np
import pandas as pd

import step3_fit_scaling as s3
import step4_export_va as s4
import step5_validate as s5
from common import (ON_LEVEL, OUT, TRAIN, VA_PATH, VDS_LIN, VH, camcas_ids, lin_W, poly_LW,
                    scaled_params)
from refine import refine
from step1_extract_params import curve, extract_m, ufit

OUTLIER = (160, 20)
VARIANTS = ("thesis", "thesis_x", "sequential", "sequential_x", "sequential_k", "sequential_k_x",
            "thesis_g", "thesis_x_g", "sequential_g", "sequential_x_g")
VAR_DIR = os.path.join(OUT, "variants")


def scaled_geom(coef, w, l):
    von = max(poly_LW(coef["poly"]["Von_lin"], w, l), poly_LW(coef["poly"]["Von_sat"], w, l))
    return von, lin_W(coef["DeltaL"], w), max(lin_W(coef["RSD_kohm"], w), 0.0) * 1e3


def u_points(w, l, br, von, rsd, ioff):
    g = curve(w, l, "linear" if br == "lin" else "saturation")
    vg, i = g.VG.to_numpy(), np.abs(g.ID.to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        y = (i - ioff) / (VDS_LIN - rsd * i) if br == "lin" else (i - ioff) / (vg - von)
    x, u = ufit(vg, y, von)
    ok = (i > ON_LEVEL) & (x > 0) & np.isfinite(u) & (u > 0)
    return np.log(x[ok]), np.log(u[ok]), vg, i


def sequential(table, exclude, kappa_last=False):
    coef, _ = s3.fit(table, exclude)                      # stage 1 (Von, Ioff, DL, RSD)
    idx = [k for k, (w, l) in enumerate(zip(table.W, table.L))
           if (w, l) in TRAIN and (w, l) not in exclude]
    t = table.loc[idx]
    new = {}
    # stage 2: alpha at the scaled Von / RSD
    for br in ("lin", "sat"):
        vals = []
        for r in t.itertuples():
            von, _, rsd = scaled_geom(coef, r.W, r.L)
            lx, lu, _, _ = u_points(r.W, r.L, br, von, rsd, getattr(r, f"Ioff_{br}"))
            vals.append(1.0 - np.polyfit(lx, lu, 1)[0])
        new[f"alpha_{br}"] = pd.Series(vals, index=t.index)
    coef, _ = s3.fit(table, exclude, params=new)
    # stage 3: kappa with the slope fixed by alpha(L, W)
    for br in ("lin", "sat"):
        vals = []
        for r in t.itertuples():
            von, _, rsd = scaled_geom(coef, r.W, r.L)
            a = poly_LW(coef["poly"][f"alpha_{br}"], r.W, r.L)
            lx, lu, _, _ = u_points(r.W, r.L, br, von, rsd, getattr(r, f"Ioff_{br}"))
            b = np.mean(lu - (1.0 - a) * lx)
            vals.append(np.exp(-b) / a)
        new[f"k_{br}"] = pd.Series(vals, index=t.index)
    coef, _ = s3.fit(table, exclude, params=new)
    # stage 4: G0 from Eqs. 2.32/2.33 with everything upstream scaled
    for br in ("lin", "sat"):
        vals = []
        for r in t.itertuples():
            von, dl, rsd = scaled_geom(coef, r.W, r.L)
            a = poly_LW(coef["poly"][f"alpha_{br}"], r.W, r.L)
            k = poly_LW(coef["poly"][f"k_{br}"], r.W, r.L)
            g = curve(r.W, r.L, "linear" if br == "lin" else "saturation")
            ih, ioff, xh = abs(g.ID.iloc[-1]), getattr(r, f"Ioff_{br}"), VH - von
            vp = (VDS_LIN - rsd * ih) if br == "lin" else xh
            vals.append((ih - ioff) / ((r.W / (r.L - dl)) * np.exp(k * xh ** a) * vp))
        new[f"G0_{br}"] = pd.Series(vals, index=t.index)
    coef, quads = s3.fit(table, exclude, params=new)
    if kappa_last:
        # stage 4b: Eq. 2.32/2.33 solved for kappa with G0 at its scaled value
        for br in ("lin", "sat"):
            vals = []
            for r in t.itertuples():
                von, dl, rsd = scaled_geom(coef, r.W, r.L)
                a = poly_LW(coef["poly"][f"alpha_{br}"], r.W, r.L)
                g0 = poly_LW(coef["poly"][f"G0_{br}"], r.W, r.L)
                g = curve(r.W, r.L, "linear" if br == "lin" else "saturation")
                ih, ioff, xh = abs(g.ID.iloc[-1]), getattr(r, f"Ioff_{br}"), VH - von
                vp = (VDS_LIN - rsd * ih) if br == "lin" else xh
                vals.append(np.log((ih - ioff) / ((r.W / (r.L - dl)) * g0 * vp)) / xh ** a)
            new[f"k_{br}"] = pd.Series(vals, index=t.index)
        coef, quads = s3.fit(table, exclude, params=new)
    # stage 5: m with the scaled branches
    ms = []
    for r in t.itertuples():
        p = {k: poly_LW(coef["poly"][k], r.W, r.L) for k in coef["poly"]}
        _, p["DeltaL"], p["RSD"] = scaled_geom(coef, r.W, r.L)
        ms.append(extract_m(r.W, r.L, p)[0])
    coef["m"] = float(np.nanmedian(ms))
    reex = t[["W", "L"]].copy()
    for k, v in new.items():
        reex[k] = v
    reex["m"] = ms
    return coef, quads, reex


def fit_variant(table, variant, extra_exclude=(), points=None):
    refined = variant.endswith("_g")
    base = variant[:-2] if refined else variant
    exclude = ((OUTLIER,) if base.endswith("_x") else ()) + tuple(extra_exclude)
    if base.startswith("sequential"):
        coef, quads, reex = sequential(table, exclude, kappa_last="_k" in base)
    else:
        (coef, quads), reex = s3.fit(table, exclude), None
    if refined:
        fit = [g for g in TRAIN if g not in exclude]
        on = lambda df: df[[(w, l) in fit for w, l in zip(df.W, df.L)]]
        coef = refine(coef, on(points), on(table))
    return coef, quads, reex, exclude


def logo_cv(table, variant, points):
    """Leave-one-geometry-out over the variant's own fit geometries: refit
    without geometry k, predict k's measured on-state points (Python mirror
    of the .va, identical to 1e-9). Uses no held-out data."""
    fit_geoms = [g for g in TRAIN if not ("_x" in variant and g == OUTLIER)]
    errs = {}
    for g in fit_geoms:
        try:
            coef = fit_variant(table, variant, (g,), points)[0]
        except ValueError:
            errs[g] = np.nan
            continue
        s = points[(points.W == g[0]) & (points.L == g[1])]
        pred = camcas_ids(s.VG.to_numpy(), s.VD.to_numpy(), g[0], g[1], scaled_params(coef, *g))
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.abs(np.log10(pred) - np.log10(np.abs(s.ID.to_numpy())))
        errs[g] = float(np.mean(e)) if np.isfinite(e).all() else np.nan
    return errs


def build(table, variant, points):
    coef, quads, reex, exclude = fit_variant(table, variant, (), points)
    if reex is not None:
        reex.to_csv(os.path.join(OUT, f"step6_{variant}_reextracted.csv"), index=False,
                    float_format="%.6g")
    coef["variant"] = variant
    coef["excluded_from_fit"] = [list(e) for e in exclude]
    s3.save(coef, quads, variant)
    return coef


def score(variant, plots):
    os.makedirs(VAR_DIR, exist_ok=True)
    path = s4.main(variant, os.path.join(VAR_DIR, f"tft_camcas_thesis_{variant}.va"))
    df, summary, per_geom = s5.run(variant, path, plots=plots, quiet=True)
    on = df[df.on]
    typ = on[(on.split == "train") & ~((on.W == OUTLIER[0]) & (on.L == OUTLIER[1]))]
    row = dict(variant=variant, nan_points=int(on.ID_va.isna().sum()))

    def mae(s):
        e = s.err_dec.to_numpy()
        return float(np.mean(np.abs(e))) if np.isfinite(e).all() else np.nan

    row["fit16_mae_dec"] = mae(on[on.split == "train"])
    row["fit15_typical_mae_dec"] = mae(typ)
    row["W160_L20_mae_dec"] = mae(on[(on.W == OUTLIER[0]) & (on.L == OUTLIER[1])])
    row["unseen3_mae_dec"] = mae(on[on.split == "holdout"])
    for sw in ("linear", "saturation", "output"):
        row[f"fit16_{sw}"] = mae(on[(on.split == "train") & (on.sweep == sw)])
        row[f"unseen3_{sw}"] = mae(on[(on.split == "holdout") & (on.sweep == sw)])
    g = per_geom[per_geom.split == "train"].groupby(["W", "L"]).mae_dec.mean()
    row["fit16_worst_geom_mae_dec"] = float(g.max())
    row["fit16_worst_geom"] = f"{g.idxmax()[0]}/{g.idxmax()[1]}" if g.notna().all() else "NaN"
    return row


def main():
    table = s3.load_table()
    points = s5.measured()
    points = points[(np.abs(points.ID) > ON_LEVEL) & (points.split == "train")]
    rows, cv_rows = [], []
    for v in VARIANTS:
        try:
            build(table, v, points)
        except ValueError as e:
            rows.append(dict(variant=v, nan_points=-1, note=f"not buildable: {e}"))
            continue
        row = score(v, plots=(v == "thesis"))
        cv = logo_cv(table, v, points)
        cv_rows += [dict(variant=v, W=g[0], L=g[1], cv_mae_dec=e) for g, e in cv.items()]
        row["logo_cv_mae_dec"] = float(np.mean(list(cv.values())))
        row["logo_cv_worst"] = float(np.max(list(cv.values())))
        rows.append(row)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(os.path.join(OUT, "step6_variant_comparison.csv"), index=False,
                float_format="%.4f")
    pd.set_option("display.width", 250)
    print(cmp_.T.to_string())

    pd.DataFrame(cv_rows).to_csv(os.path.join(OUT, "step6_logo_cv.csv"), index=False,
                                 float_format="%.4f")
    # choose without the held-out geometries: the model must be defined on the
    # whole fit grid (no NaN), then the lowest leave-one-geometry-out error
    ok = cmp_[(cmp_.nan_points == 0) & cmp_.logo_cv_mae_dec.notna()]
    final = ok.sort_values("logo_cv_mae_dec").variant.iloc[0]
    with open(os.path.join(OUT, "final_variant.txt"), "w") as fh:
        fh.write(final + "\n")
    print("\nFINAL VARIANT (chosen by leave-one-geometry-out CV on the fit geometries):", final)
    s4.main(final, VA_PATH)
    s5.run(final, VA_PATH, plots=True, quiet=False)


if __name__ == "__main__":
    main()
