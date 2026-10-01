"""Steps 2-4 -- geometry scaling, overlap/channel separation, choice of the
bias-dependent form, and the comparison with the existing model.

Data used (all at VDS = 0 -- the only bias for which the data are
unambiguous, see README):
  * cg   (S+D on CMU-low): total gate capacitance Cgg(VG)
  * "cgd"/"cgs" (one electrode on CMU-low, the other floating): off-state
    plateau = that electrode's overlap capacitance; on-state = Cgg
  * cg of the "Vd=0.1" file (closest to VD = 0 of that family)

Outputs: ac_scaling.csv, ac_shape_fit.json, ac_fit_points.csv,
ac_validation_*.csv, ac_baseline_comparison.csv, plots/ac_step2_*.png ...
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import acmodel as M
from cvdata import GEOMS, OUT, PLOTS, load_all

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e4e0"
# median linear-regime subthreshold swing of the 4 C-V geometries in the DC
# data (66-92 mV/dec, outputs_camcas_thesis/selected_curves) / ln 10
SS_DC = 0.0685
S1_DC = SS_DC / np.log(10)
GCOL = {(20, 20): "#2a78d6", (40, 20): "#eb6834", (160, 15): "#1baf7a", (160, 20): "#eda100"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


# ------------------------------------------------------------------ data ----
def curves(df):
    """The VDS = 0 total-gate-capacitance curves used for the shape fit."""
    out = []
    for (w, l) in GEOMS:
        g = df[(df.W == w) & (df.L == l)]
        b = g[(g.family == "basic") & (g.kind == "cg")]
        v = g[(g.family == "vd") & np.isclose(g.VD, 0.1)]
        for tag, s in (("cg", b), ("cg_vd0.1", v)):
            out.append(dict(W=w, L=l, tag=tag, time=s.time.iloc[0],
                            VG=s.VG.to_numpy(), C=s.C.to_numpy(), G=s.G.to_numpy(),
                            f=s.f.to_numpy()))
    return out


def plateaus(df):
    rows = []
    b = df[df.family == "basic"]
    for (w, l) in GEOMS:
        g = lambda k: b[(b.W == w) & (b.L == l) & (b.kind == k)]
        off = lambda k: g(k)[g(k).VG <= -2].C.mean()
        rows.append(dict(W=w, L=l, Cg_off=off("cg"), Cov_d=off("cgd"), Cov_s=off("cgs"),
                         Cg_on5=g("cg")[np.isclose(g("cg").VG, 5)].C.mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- scaling ----
def scaling(pl, Cch_inf):
    """Fit each plateau with candidate geometry laws; leave-one-geometry-out
    prediction error for each law."""
    pl = pl.copy()
    pl["Cch"] = Cch_inf
    W, L = pl.W.to_numpy(float), pl.L.to_numpy(float)
    laws = {"k*W": lambda W, L: np.c_[W],
            "k*W*L": lambda W, L: np.c_[W * L],
            "a*W + b*W*L": lambda W, L: np.c_[W, W * L],
            "k*W*(L - dL)": None}
    rows = []
    for q in ("Cg_off", "Cov_d", "Cov_s", "Cch"):
        y = pl[q].to_numpy()
        for name, X in laws.items():
            if X is None:   # nonlinear in dL -> same as a*W + b*W*L with dL = -a/b
                X = laws["a*W + b*W*L"]
            beta = np.linalg.lstsq(X(W, L), y, rcond=None)[0]
            fit = X(W, L) @ beta
            loo = []
            for k in range(len(y)):
                m = np.arange(len(y)) != k
                bk = np.linalg.lstsq(X(W, L)[m], y[m], rcond=None)[0]
                loo.append((X(W, L)[k] @ bk - y[k]) / y[k])
            row = dict(quantity=q, law=name, coef=" ".join(f"{c:.4e}" for c in beta),
                       max_fit_err_pct=100 * np.max(np.abs(fit - y) / y),
                       loo_max_err_pct=100 * np.max(np.abs(loo)),
                       loo_rms_err_pct=100 * np.sqrt(np.mean(np.square(loo))))
            if name == "k*W*(L - dL)":
                row["coef"] = f"k={beta[1]:.4e} F/um^2 dL={-beta[0] / beta[1]:.3f} um"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------- shape fit ------
def fit_shape(cv, form, region=None):
    """Joint fit of all VDS=0 Cgg curves: shared shape (s1[,beta,s2,d2]),
    per-curve V0 (turn-on position, which drifts with bias stress), per-curve
    off-plateau Coff and channel amplitude Cch. Points where the 10 kHz
    signal is RC-limited (G/omega > 5 % of C) are excluded from the fit."""
    # mix: the sharp width s1 is fixed from the DC subthreshold swing (the
    # 10 kHz transition points are excluded as non-quasi-static, so they
    # cannot constrain it); logistic: its single width is free
    shared = ["s1"] if form == "logistic" else ["beta", "s2", "d2"]
    x0 = {"s1": 0.15, "beta": 0.3, "s2": 0.8, "d2": 0.3}
    lo = {"s1": 0.01, "beta": 0.0, "s2": 0.05, "d2": -1.0}
    hi = {"s1": 2.0, "beta": 1.0, "s2": 5.0, "d2": 3.0}
    nc = len(cv)
    masks = [(c["G"] / (2 * np.pi * c["f"]) < 0.05 * c["C"]) for c in cv]

    def unpack(x):
        p = dict(zip(shared, x[:len(shared)]))
        if form != "logistic":
            p["s1"] = S1_DC
        per = x[len(shared):].reshape(nc, 3)   # V0 (V), Coff (pF), Cch (pF)
        return p, per

    def resid(x):
        p, per = unpack(x)
        r = []
        for c, (v0, coff, cch), m in zip(cv, per, masks):
            q = dict(p, V0=v0)
            model = (coff + cch * M.n_shape(c["VG"], q, form)) * 1e-12
            r.append(((model - c["C"]) / c["C"])[m])
        return np.concatenate(r)

    xs = [x0[k] for k in shared]
    for c in cv:
        n = (c["C"] - c["C"][:5].mean()) / (c["C"][-1] - c["C"][:5].mean())
        xs += [float(np.interp(0.5, n[c["VG"] > -1.5], c["VG"][c["VG"] > -1.5])),
               c["C"][:5].mean() * 1e12, (c["C"][-1] - c["C"][:5].mean()) * 1e12]
    lb = [lo[k] for k in shared] + [-3, 0, 0] * nc
    ub = [hi[k] for k in shared] + [3, 100, 100] * nc
    res = least_squares(resid, xs, bounds=(lb, ub), x_scale="jac")
    p, per = unpack(res.x)
    return p, per, masks, res


def main():
    os.makedirs(PLOTS, exist_ok=True)
    df = load_all()
    cv = curves(df)
    pl = plateaus(df)

    results = {}
    for form in ("logistic", "mix"):
        p, per, masks, res = fit_shape(cv, form)
        r = res.fun
        results[form] = dict(shape=p, per=per, masks=masks, rms_pct=100 * np.sqrt(np.mean(r ** 2)),
                             max_pct=100 * np.max(np.abs(r)), n=len(r))
        print(f"{form}: shape {p}  rms {results[form]['rms_pct']:.2f}%  max {results[form]['max_pct']:.2f}%")
    best = results["mix"]
    per = best["per"]
    # channel amplitude (asymptotic) per geometry: mean over that geometry's curves
    fitp = pd.DataFrame([dict(W=c["W"], L=c["L"], tag=c["tag"], time=c["time"], V0=v0,
                              Coff_pF=coff, Cch_inf_pF=cch)
                         for c, (v0, coff, cch) in zip(cv, per)])
    fitp.to_csv(os.path.join(OUT, "ac_shape_fit_per_curve.csv"), index=False, float_format="%.5g")
    cch_geo = fitp.groupby(["W", "L"]).Cch_inf_pF.mean().reindex(
        pd.MultiIndex.from_tuples(GEOMS)).to_numpy() * 1e-12
    sc = scaling(pl, cch_geo)
    sc.to_csv(os.path.join(OUT, "ac_scaling.csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(fitp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(sc.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    A = np.array([w * l for w, l in GEOMS], float) * M.UM2
    P = dict(best["shape"])
    P["Cch_area"] = float(np.sum(cch_geo * A) / np.sum(A * A))
    P["Cov_d_area"] = float(np.sum(pl.Cov_d.to_numpy() * A) / np.sum(A * A))
    P["Cov_s_area"] = float(np.sum(pl.Cov_s.to_numpy() * A) / np.sum(A * A))
    P["Cg_off_area"] = float(np.sum(pl.Cg_off.to_numpy() * A) / np.sum(A * A))
    P["V0_per_curve"] = {f"W{c['W']}_L{c['L']}_{c['tag']}": float(v0)
                         for c, (v0, _, _) in zip(cv, per)}
    P["fit_rms_pct"] = {k: v["rms_pct"] for k, v in results.items()}
    P["fit_max_pct"] = {k: v["max_pct"] for k, v in results.items()}
    with open(os.path.join(OUT, "ac_model_params.json"), "w") as f:
        json.dump(P, f, indent=2)
    print(json.dumps({k: v for k, v in P.items() if k != "V0_per_curve"}, indent=1))
    plot_shapes(cv, results)
    plot_scaling(pl, cch_geo)


def plot_shapes(cv, results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        style(ax)
    for c, (v0, coff, cch) in zip(cv, results["mix"]["per"]):
        col = GCOL[(c["W"], c["L"])]
        n = (c["C"] * 1e12 - coff) / cch
        ls = "o" if c["tag"] == "cg" else "s"
        axes[0].plot(c["VG"] - v0, n, ls, ms=3, color=col, mfc="none" if ls == "s" else col,
                     label=f"{c['W']}x{c['L']} {c['tag']}")
        axes[1].semilogy(c["VG"] - v0, np.clip(1 - n, 1e-4, None), ls, ms=3, color=col,
                         mfc="none" if ls == "s" else col)
    x = np.linspace(-2, 5, 400)
    for form, ls, lab in (("mix", "-", "two-step form (proposed)"), ("logistic", "--", "single logistic")):
        p = dict(results[form]["shape"], V0=0.0)
        axes[0].plot(x, M.n_shape(x, p, form), ls, color=INK, lw=1.8, label=lab)
        axes[1].semilogy(x, 1 - M.n_shape(x, p, form), ls, color=INK, lw=1.8)
    axes[0].set(xlabel="VG - V0 (V)", ylabel="(C - C_off) / C_ch,max", xlim=(-2, 5))
    axes[1].set(xlabel="VG - V0 (V)", ylabel="1 - normalized C  (approach to plateau)",
                xlim=(-0.5, 5), ylim=(1e-3, 1.5))
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Normalized channel capacitance at VDS = 0, all curves aligned by their "
                 "fitted turn-on V0", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step4_shape.png"), dpi=110)
    plt.close(fig)


def plot_scaling(pl, cch):
    WL = (pl.W * pl.L).to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    items = [("channel C_ch (VG -> inf)", cch), ("off-state C_g,off (cg)", pl.Cg_off.to_numpy())]
    for ax, (lab, y) in zip(axes[:2], items):
        style(ax)
        k = np.sum(y * WL) / np.sum(WL * WL)
        x = np.linspace(0, 3500, 10)
        ax.plot(x, k * x * 1e12, "-", color=MUTED, lw=1.2, label=f"{k * 1e15:.3f} fF/um^2 x W*L")
        for (w, l), yy, a in zip(GEOMS, y, WL):
            ax.plot(a, yy * 1e12, "o", ms=9, color=GCOL[(w, l)], label=f"W{w} L{l}")
        ax.set(xlabel="W*L (um^2)", ylabel="pF", title=lab)
        ax.legend(fontsize=7.5, frameon=False)
    ax = axes[2]
    style(ax)
    for (w, l), cd, cs, a in zip(GEOMS, pl.Cov_d, pl.Cov_s, WL):
        ax.plot(a, cd * 1e15 / a, "o", ms=9, color=GCOL[(w, l)])
        ax.plot(a, cs * 1e15 / a, "s", ms=9, color=GCOL[(w, l)], mfc="none", mew=2)
    ax.set(xlabel="W*L (um^2)", ylabel="fF/um^2", ylim=(0, 0.35),
           title='overlap per area: "cgd" off (filled) / "cgs" off (open)')
    fig.suptitle("Geometry scaling of the measured capacitance plateaus", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step2_scaling.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
