"""Validate the CAMCAS-style scalable model fit by scripts/fit_camcas_model.py
against the measured transfer curves it was fit to, mirroring the thesis's
own model-validation figures (Sec. 3.4.2 / Ch. 4): per-geometry measured vs.
modeled ID-VGS in both regimes, plus a relative-error summary.

Usage:
    python scripts/plot_camcas_fit.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data_cleaned_2")
OUT_DIR = os.path.join(REPO, "outputs_camcas")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
VDS_LIN = 0.1


def eval_bivariate(coef, L, W):
    val = ((coef["a"] * W + coef["b"]) * L ** 2
           + (coef["c"] * W + coef["d"]) * L
           + (coef["e"] * W + coef["f"]))
    return np.exp(val) if coef.get("log") else val


class CamcasModel:
    def __init__(self, coeffs):
        self.c = coeffs
        self.p = coeffs["params"]
        self.VH = coeffs["VH"]
        self.vds_lin = coeffs["VDS_lin"]
        self.vds_sat = coeffs["VDS_sat"]

    def params_at(self, L, W):
        out = {name: eval_bivariate(self.p[name], L, W) for name in self.p}
        out["DeltaL"] = self.c["DeltaL_of_W"]["a"] * W + self.c["DeltaL_of_W"]["b"]
        out["RSD"] = self.c["RSD_ohm_of_W"]["a"] * W + self.c["RSD_ohm_of_W"]["b"]
        return out

    def ids(self, vg, vds, L, W, m=2.2):
        """Full unified drain current (thesis Eq. 2.34) using the scalable
        (L, W) parameter surface -- the actual compact model, usable at any
        geometry, not just the 19 measured ones. `vds` is the ACTUAL bias at
        which to evaluate (0.1V when comparing against our linear-sweep
        curves, 5V against our saturation-sweep curves) -- see _ids_core."""
        return self._ids_core(vg, vds, self.params_at(L, W), L, W, m)

    @staticmethod
    def _ids_core(vg, vds, p, L, W, m=2.2):
        """Full unified drain current (thesis Eq. 2.34), for an array of VGS
        at fixed geometry and A SINGLE ACTUAL VDS -- combines the linear-
        and saturation-regime branches with harmonic-mean smoothing.
        Matches Appendix F's structure exactly: the linear branch uses the
        ACTUAL instance Vds (Appendix F: `Ids_lin = (G*Vds + Ioff)/(1+RSD*G)`
        with `Vds = V(d,s)`, not a fixed reference), while the saturation
        branch is Vds-independent by construction. The harmonic mean is what
        lets a single expression transition from triode (small Vds, current
        set by the linear branch, ~proportional to Vds) to saturation (large
        Vds, current pinned by the Vds-independent saturation branch) --
        get this wrong (e.g. hardcode Vds in the linear branch) and the
        model can never actually saturate at high Vds, which is exactly the
        bug an earlier version of this script had (see outputs_camcas/README.md).
        Below Von_eff, output the appropriate off-current."""
        Lp = max(L - p["DeltaL"], 0.1)
        von_eff = max(p["Von_lin"], p["Von_sat"])

        vg = np.asarray(vg, dtype=float)
        ids = np.empty_like(vg)
        on = vg > von_eff

        # off-state: constant-current floor from whichever regime applies
        # (thesis Appendix F: linear floor if VDS < VGS-Von, else saturation)
        ids[~on] = np.where(vds < (vg[~on] - von_eff), p["Ioff_lin"], p["Ioff_sat"])

        x = vg[on] - von_eff
        expo_lin = np.clip(p["kappa_lin"] * np.power(x, p["alpha_lin"]), -700, 700)
        G = p["G0_lin"] * (W / Lp) * np.exp(expo_lin)
        # linear branch uses the self-consistent VDS' = VDS - 2*Rc*IDS; solve
        # the implicit G*Vds'/(1+RSD*G) form directly since it's linear in IDS
        ids_lin = (G * vds + p["Ioff_lin"]) / (1 + p["RSD"] * G)

        expo_sat = np.clip(p["kappa_sat"] * np.power(x, p["alpha_sat"]), -700, 700)
        ids_sat = p["G0_sat"] * (W / Lp) * np.exp(expo_sat) * x + p["Ioff_sat"]

        ids_lin = np.maximum(ids_lin, 1e-15)
        ids_sat = np.maximum(ids_sat, 1e-15)
        ids[on] = (ids_lin ** (-m) + ids_sat ** (-m)) ** (-1.0 / m)
        return ids


def list_geometries():
    geoms = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_linear_best.csv"))):
        base = os.path.basename(path)
        wl = base[:-len("_linear_best.csv")]
        w_str, l_str = wl.split("_L")
        geoms.append((float(w_str[1:]), float(l_str)))
    return geoms


def main():
    with open(os.path.join(OUT_DIR, "camcas_coefficients.json")) as f:
        coeffs = json.load(f)
    model = CamcasModel(coeffs)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Per-geometry "oracle" parameters (scripts/fit_camcas_model.py's direct
    # NLS fit to that one device, before any (L,W) generalization) let us
    # separate two very different questions: (1) does the CAMCAS functional
    # FORM fit a real device's curve well? and (2) does the scalable (L,W)
    # surface -- the actual usable compact model, needed at unmeasured
    # geometries -- reproduce that? These can differ a lot when per-geometry
    # parameters are noisy relative to how sensitive the exponential model
    # is to them (see outputs_camcas/README.md).
    extracted = pd.read_csv(os.path.join(OUT_DIR, "extracted_params.csv"))
    dl_rsd = pd.read_csv(os.path.join(OUT_DIR, "dl_rsd_per_width.csv")).set_index("W")

    def oracle_params(L, W):
        row = extracted[(extracted.W == W) & (extracted.L == L)]
        if row.empty or W not in dl_rsd.index:
            return None
        row = row.iloc[0]
        # Some per-device NLS fits land on a degenerate optimum where kappa,
        # alpha and G0 trade off without enough curvature in that one noisy
        # curve to pin all three down -- G0 runs away to its search bound
        # (~30% of our geometries in one regime or the other). These are
        # already excluded from the scalable (L,W) surface fit by the same
        # physical-bounds check in fit_camcas_model.py; apply it here too so
        # the "oracle" comparison reflects genuinely converged fits.
        if not (1e-10 < row.G0_lin < 1) or not (1e-10 < row.G0_sat < 1):
            return None
        return {
            "Von_lin": row.Von_lin, "Von_sat": row.Von_sat,
            "alpha_lin": row.alpha_lin, "alpha_sat": row.alpha_sat,
            "kappa_lin": row.kappa_lin, "kappa_sat": row.kappa_sat,
            "G0_lin": row.G0_lin, "G0_sat": row.G0_sat,
            "Ioff_lin": row.Ioff_lin, "Ioff_sat": row.Ioff_sat,
            "DeltaL": dl_rsd.loc[W, "DeltaL"], "RSD": dl_rsd.loc[W, "RSD_ohm"],
        }

    geoms = list_geometries()
    error_rows = []
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    axes = axes.ravel()

    for i, (W, L) in enumerate(geoms):
        df_lin = pd.read_csv(os.path.join(DATA_DIR, f"W{int(W)}_L{int(L)}_linear_best.csv")).sort_values("VG")
        df_sat = pd.read_csv(os.path.join(DATA_DIR, f"W{int(W)}_L{int(L)}_saturation_best.csv")).sort_values("VG")

        vg_lin = df_lin["VG"].to_numpy()
        vg_sat = df_sat["VG"].to_numpy()
        meas_lin = df_lin["ID"].to_numpy()
        meas_sat = df_sat["ID"].to_numpy()

        # Evaluate at the ACTUAL bias each curve was measured at (0.1V /
        # 5V -- data_cleaned_2/README.md) so the harmonic-mean blend
        # correctly hands off to the (Vds-independent) saturation branch
        # instead of always using the linear branch's small reference Vds.
        pred_lin = model.ids(vg_lin, model.vds_lin, L, W)
        pred_sat = model.ids(vg_sat, model.vds_sat, L, W)

        op = oracle_params(L, W)
        oracle_lin = model._ids_core(vg_lin, model.vds_lin, op, L, W) if op else None
        oracle_sat = model._ids_core(vg_sat, model.vds_sat, op, L, W) if op else None

        on_lin = meas_lin > 1e-11
        on_sat = meas_sat > 1e-11
        rel_err_lin = np.abs(pred_lin[on_lin] - meas_lin[on_lin]) / np.abs(meas_lin[on_lin])
        rel_err_sat = np.abs(pred_sat[on_sat] - meas_sat[on_sat]) / np.abs(meas_sat[on_sat])
        row = {
            "W": W, "L": L,
            "median_rel_err_lin": np.median(rel_err_lin) if on_lin.any() else np.nan,
            "median_rel_err_sat": np.median(rel_err_sat) if on_sat.any() else np.nan,
        }
        if op:
            oe_lin = np.abs(oracle_lin[on_lin] - meas_lin[on_lin]) / np.abs(meas_lin[on_lin])
            oe_sat = np.abs(oracle_sat[on_sat] - meas_sat[on_sat]) / np.abs(meas_sat[on_sat])
            row["oracle_median_rel_err_lin"] = np.median(oe_lin) if on_lin.any() else np.nan
            row["oracle_median_rel_err_sat"] = np.median(oe_sat) if on_sat.any() else np.nan
        error_rows.append(row)

        ax = axes[i]
        ax.semilogy(vg_lin, np.abs(meas_lin), "b-", label="measured (lin)")
        ax.semilogy(vg_lin, np.abs(pred_lin), "b--", label="CAMCAS scalable (lin)")
        ax.semilogy(vg_sat, np.abs(meas_sat), "r-", label="measured (sat)")
        ax.semilogy(vg_sat, np.abs(pred_sat), "r--", label="CAMCAS scalable (sat)")
        if op:
            ax.semilogy(vg_lin, np.abs(oracle_lin), "b:", alpha=0.6, label="CAMCAS per-device (lin)")
            ax.semilogy(vg_sat, np.abs(oracle_sat), "r:", alpha=0.6, label="CAMCAS per-device (sat)")
        ax.set_title(f"W={W:.0f} L={L:.0f} um", fontsize=9)
        ax.set_ylim(1e-13, 1e-3)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(len(geoms), len(axes)):
        axes[j].axis("off")

    fig.suptitle("CAMCAS-style scalable model vs. measured transfer curves (data_cleaned_2)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "camcas_vs_measured_all_geometries.png"), dpi=130)
    print(f"Wrote {os.path.join(PLOT_DIR, 'camcas_vs_measured_all_geometries.png')}")

    err_df = pd.DataFrame(error_rows)
    err_df.to_csv(os.path.join(OUT_DIR, "camcas_relative_error.csv"), index=False)
    print("\nMedian relative error (on-state, |ID| > 1e-11 A):")
    print(err_df.to_string(index=False))
    print("\n                          scalable (L,W) model   per-device oracle fit")
    print(f"median rel. err (lin):    {err_df['median_rel_err_lin'].median():21.3f}   "
          f"{err_df['oracle_median_rel_err_lin'].median():21.3f}")
    print(f"median rel. err (sat):    {err_df['median_rel_err_sat'].median():21.3f}   "
          f"{err_df['oracle_median_rel_err_sat'].median():21.3f}")


if __name__ == "__main__":
    main()
