"""Steps 5-6 -- the proposed charge model against every usable measurement,
the existing (baseline) charge block, and model sanity checks.

Model parameters: outputs_camcas_ac/ac_model_params.json (step2_4_analyse).
Overlaps: C'ov,d and C'ov,s keep the measured "cgd"/"cgs" off-state ratio
and are scaled to sum to the directly measured cg off-state value (the
floating-electrode measurements each include a ~15-20 fF series path
through the other electrode).

Two ways of placing the turn-on V0 are scored:
  fitted     V0 per measured curve (absorbs the bias-stress drift; tests
             geometry scaling + shape)
  predictive V0 = Von_eff(W, L) of the validated DC model + dV_C, one
             offset from the least-stressed sweep of each device

Outputs: ac_validation_metrics.csv, ac_baseline_comparison.csv,
ac_conservation_check.csv, plots/ac_step6_*.png, plots/ac_step5_vds.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acmodel as M
from cvdata import GEOMS, OUT, PLOTS, REPO, load_all
from step2_4_analyse import GCOL, INK, MUTED, curves, plateaus, style

sys.path.insert(0, os.path.join(REPO, "scripts", "camcas_thesis"))
from common import OUT as DC_OUT, scaled_params  # noqa: E402

LEAST_STRESSED = {(20, 20): "cg_vd0.1", (40, 20): "cg", (160, 15): "cg_vd0.1", (160, 20): "cg"}


def dc_von(w, l):
    with open(os.path.join(DC_OUT, "step3_coefficients_thesis_g.json")) as f:
        p = scaled_params(json.load(f), w, l)
    return max(p["Von_lin"], p["Von_sat"])


def load_params(pl):
    P = json.load(open(os.path.join(OUT, "ac_model_params.json")))
    A = np.array([w * l for w, l in GEOMS], float) * M.UM2
    frac_d = pl.Cov_d / (pl.Cov_d + pl.Cov_s)
    covd = (pl.Cg_off * frac_d).to_numpy()
    covs = (pl.Cg_off * (1 - frac_d)).to_numpy()
    P["Cov_d_area"] = float(np.sum(covd * A) / np.sum(A * A))
    P["Cov_s_area"] = float(np.sum(covs * A) / np.sum(A * A))
    offs = {g: P["V0_per_curve"][f"W{g[0]}_L{g[1]}_{LEAST_STRESSED[g]}"] - dc_von(*g) for g in GEOMS}
    P["dV_C"] = float(np.median(list(offs.values())))
    P["dV_C_per_device"] = {f"W{g[0]}_L{g[1]}": v for g, v in offs.items()}
    return P


def model_cgg(vg, w, l, P, v0):
    p = dict(P, V0=v0)
    A = w * l * M.UM2
    return (P["Cov_d_area"] + P["Cov_s_area"]) * A + P["Cch_area"] * A * M.n_shape(vg, p, "mix")


def region(vg, v0, c, g, f):
    nqs = g / (2 * np.pi * f) > 0.05 * c
    return np.where(nqs, "transition (non-QS)", np.where(vg < v0, "off", "on"))


def metrics(cv, P):
    rows, pts = [], []
    for c in cv:
        w, l = c["W"], c["L"]
        v0_fit = P["V0_per_curve"][f"W{w}_L{l}_{c['tag']}"]
        v0_pred = dc_von(w, l) + P["dV_C"]
        reg = region(c["VG"], v0_fit, c["C"], c["G"], c["f"])
        for mode, v0 in (("fitted V0", v0_fit), ("predictive V0", v0_pred)):
            m = model_cgg(c["VG"], w, l, P, v0)
            e = (m - c["C"]) / c["C"]
            pts.append(pd.DataFrame(dict(W=w, L=l, curve=c["tag"], mode=mode, VG=c["VG"],
                                         C_meas=c["C"], C_model=m, rel_err=e, region=reg)))
    pts = pd.concat(pts, ignore_index=True)
    for (w, l, mode, reg), g in pts.groupby(["W", "L", "mode", "region"]):
        rows.append(dict(W=w, L=l, mode=mode, region=reg, n=len(g),
                         mean_abs_err_pct=100 * g.rel_err.abs().mean(),
                         max_abs_err_pct=100 * g.rel_err.abs().max(),
                         bias_pct=100 * g.rel_err.mean()))
    for (mode, reg), g in pts.groupby(["mode", "region"]):
        rows.append(dict(W="all", L="all", mode=mode, region=reg, n=len(g),
                         mean_abs_err_pct=100 * g.rel_err.abs().mean(),
                         max_abs_err_pct=100 * g.rel_err.abs().max(),
                         bias_pct=100 * g.rel_err.mean()))
    return pd.DataFrame(rows), pts


def floating_check(df, P):
    rows = []
    b = df[df.family == "basic"]
    for (w, l) in GEOMS:
        A = w * l * M.UM2
        for kind, cov in (("cgd", P["Cov_d_area"]), ("cgs", P["Cov_s_area"])):
            g = b[(b.W == w) & (b.L == l) & (b.kind == kind)]
            off = g[g.VG <= -2].C.mean()
            on = g[g.VG >= 3].C.to_numpy()
            v0 = P["V0_per_curve"][f"W{w}_L{l}_cg"]
            mon = model_cgg(g[g.VG >= 3].VG.to_numpy(), w, l, P, v0)
            rows.append(dict(W=w, L=l, config=kind, off_meas_pF=off * 1e12,
                             off_model_pF=cov * A * 1e12, off_err_pct=100 * (cov * A - off) / off,
                             on_mean_abs_err_pct=100 * np.mean(np.abs(mon - on) / on)))
    return pd.DataFrame(rows)


def baseline(pl, P):
    rows = []
    for (w, l), r in zip(GEOMS, pl.itertuples()):
        A = w * l * M.UM2
        meas = dict(Cg_off=r.Cg_off, Cg_on=r.Cg_on5, Cov_d=r.Cov_d, Cov_s=r.Cov_s,
                    Cch=r.Cg_on5 - r.Cg_off)
        for name, b in (("as coded (Listing F.1)", M.baseline_as_coded(w, l)),
                        ("thesis intent (Table 4.1)", M.baseline_intended(w, l))):
            t = M.baseline_terminal_caps(b)
            mod = dict(Cg_off=t["Cgg"], Cg_on=t["Cgg"], Cov_d=b["Covd"], Cov_s=b["Covs"], Cch=b["Cch"])
            for q in meas:
                rows.append(dict(W=w, L=l, model=name, quantity=q, measured_pF=meas[q] * 1e12,
                                 model_pF=mod[q] * 1e12, ratio=mod[q] / meas[q]))
        v0 = P["V0_per_curve"][f"W{w}_L{l}_cg"]
        prop = dict(Cg_off=(P["Cov_d_area"] + P["Cov_s_area"]) * A,
                    Cg_on=float(model_cgg(np.array([5.0]), w, l, P, v0)[0]),
                    Cov_d=P["Cov_d_area"] * A, Cov_s=P["Cov_s_area"] * A,
                    Cch=float(model_cgg(np.array([5.0]), w, l, P, v0)[0])
                    - (P["Cov_d_area"] + P["Cov_s_area"]) * A)
        for q in meas:
            rows.append(dict(W=w, L=l, model="proposed", quantity=q, measured_pF=meas[q] * 1e12,
                             model_pF=prop[q] * 1e12, ratio=prop[q] / meas[q]))
    return pd.DataFrame(rows)


def conservation(P):
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        vg, vd, vs = rng.uniform(-3, 5), rng.uniform(-5, 5), rng.uniform(-1, 1)
        C = M.terminal_caps(vg, vd, vs, 160, 20, dict(P, V0=0.0))
        qg, qd, qs = M.charges(vg - vs, vd - vs, 160, 20, dict(P, V0=0.0))
        cgg = abs(C["gg"])
        rows.append(dict(VG=vg, VD=vd, VS=vs,
                         charge_sum_rel=abs(qg + qd + qs) / max(abs(qg), 1e-30),
                         col_sum_rel=max(abs(C["g" + j] + C["d" + j] + C["s" + j]) for j in "gds") / cgg,
                         row_sum_rel=max(abs(C[i + "g"] + C[i + "d"] + C[i + "s"]) for i in "gds") / cgg))
    sym = []
    A = 160 * 20 * M.UM2
    for vg in np.linspace(-2, 5, 15):
        C = M.terminal_caps(vg, 0.0, 0.0, 160, 20, dict(P, V0=0.0))
        ch_s = -C["gs"] - P["Cov_s_area"] * A      # channel part of Cgs
        ch_d = -C["gd"] - P["Cov_d_area"] * A
        sym.append(abs(ch_s - ch_d) / C["gg"])
    d = pd.DataFrame(rows)
    return pd.DataFrame([dict(check="Qg+Qd+Qs = 0", worst_rel=d.charge_sum_rel.max()),
                         dict(check="sum_i C_ij = 0 (charge conservation)", worst_rel=d.col_sum_rel.max()),
                         dict(check="sum_j C_ij = 0 (only voltage differences matter)",
                              worst_rel=d.row_sum_rel.max()),
                         dict(check="channel Cgs = channel Cgd at VDS = 0 (S/D symmetry)", worst_rel=max(sym))])


# ------------------------------------------------------------------ plots ---
def plot_cg(pts, P):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    for ax, (w, l) in zip(axes.ravel(), GEOMS):
        style(ax)
        for cur, mk in (("cg", "o"), ("cg_vd0.1", "s")):
            s = pts[(pts.W == w) & (pts.L == l) & (pts.curve == cur) & (pts["mode"] == "fitted V0")]
            ax.plot(s.VG, s.C_meas * 1e12, mk, ms=4, color=GCOL[(w, l)], mfc="none" if mk == "s" else
                    GCOL[(w, l)], label=f"measured {cur}")
            ax.plot(s.VG, s.C_model * 1e12, "-", color=INK, lw=1.4,
                    label="model, fitted V0" if cur == "cg" else None)
        s = pts[(pts.W == w) & (pts.L == l) & (pts.curve == "cg") & (pts["mode"] == "predictive V0")]
        ax.plot(s.VG, s.C_model * 1e12, "--", color=MUTED, lw=1.6, label="model, V0 from DC Von")
        b = M.baseline_terminal_caps(M.baseline_intended(w, l))["Cgg"]
        ax.axhline(b * 1e12, color="#e34948", lw=1.4, ls=":", label="existing model, thesis intent")
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
        ax.set_xlabel("VG (V)", fontsize=9, color=MUTED)
        ax.set_ylabel("Cgg = Cgs + Cgd (pF)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="center right")
    fig.suptitle("Total gate capacitance at VDS = 0: measured vs proposed charge model "
                 "(existing as-coded model is 350-1500x off scale)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step6_cgg_validation.png"), dpi=110)
    plt.close(fig)


def plot_err(pts):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    style(ax)
    s = pts[pts["mode"] == "fitted V0"]
    for (w, l), g in s.groupby(["W", "L"]):
        v0 = g.VG  # plotted vs VG
        ax.plot(g.VG, 100 * g.rel_err, "o", ms=3, color=GCOL[(w, l)], label=f"W{w} L{l}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xlabel="VG (V)", ylabel="(model - measured)/measured  (%)",
           title="Relative error of Cgg, fitted-V0 mode (spikes = non-quasi-static transition)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(-40, 40)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step6_error.png"), dpi=110)
    plt.close(fig)


def plot_floating(df, P):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    b = df[df.family == "basic"]
    for ax, (w, l) in zip(axes.ravel(), GEOMS):
        style(ax)
        A = w * l * M.UM2
        for kind, col, cov in (("cgd", "#eb6834", P["Cov_d_area"]), ("cgs", "#1baf7a", P["Cov_s_area"])):
            g = b[(b.W == w) & (b.L == l) & (b.kind == kind)]
            ax.plot(g.VG, g.C * 1e12, "o", ms=3.5, color=col, label=f'measured "{kind}"')
            ax.axhline(cov * A * 1e12, color=col, lw=1.4, ls="--",
                       label=f"model C_ov,{kind[-1]} (off plateau)")
        v0 = P["V0_per_curve"][f"W{w}_L{l}_cg"]
        x = np.linspace(-3, 5, 300)
        ax.plot(x, model_cgg(x, w, l, P, v0) * 1e12, "-", color=INK, lw=1.2,
                label="model Cgg (what a floating-electrode\nset-up reads once the channel conducts)")
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
        ax.set_xlabel("VG (V)", fontsize=9, color=MUTED)
        ax.set_ylabel("C (pF)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=6.8, frameon=False, loc="center right")
    fig.suptitle('"cgd" / "cgs" sweeps (other electrode floating): off plateau = overlap of the '
                 "measured electrode; on plateau = full Cgg", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step6_floating.png"), dpi=110)
    plt.close(fig)


def plot_baseline(base):
    q = ["Cov_d", "Cov_s", "Cg_off", "Cch", "Cg_on"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    cols = {"as coded (Listing F.1)": "#e34948", "thesis intent (Table 4.1)": "#eda100",
            "proposed": "#2a78d6"}
    for ax, (w, l) in zip(axes, GEOMS):
        style(ax)
        s = base[(base.W == w) & (base.L == l)]
        x = np.arange(len(q))
        for i, (m, c) in enumerate(cols.items()):
            r = s[s.model == m].set_index("quantity").loc[q, "ratio"]
            ax.bar(x + (i - 1) * 0.27, r, 0.25, color=c, label=m, edgecolor="white")
        ax.axhline(1, color=INK, lw=1)
        ax.set_yscale("log")
        ax.set_xticks(x, q, fontsize=8)
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
    axes[0].set_ylabel("model / measured", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Existing charge block vs measured capacitance plateaus (1 = perfect)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step3_baseline.png"), dpi=110)
    plt.close(fig)


def plot_vds(P):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    vds = np.linspace(0, 5, 101)
    p = dict(P, V0=0.0)
    ramp = ["#9fc3ee", "#4b8bdc", "#1d5aa6", "#0b2a50"]
    for c, vgs in zip(ramp, (1, 2, 3, 5)):
        C = [M.terminal_caps(vgs, v, 0.0, 160, 20, p) for v in vds]
        axes[0].plot(vds, [-x["gs"] * 1e12 for x in C], color=c, lw=2, label=f"VGS-V0 = {vgs} V")
        axes[1].plot(vds, [-x["gd"] * 1e12 for x in C], color=c, lw=2)
        axes[2].plot(vds, [x["gg"] * 1e12 for x in C], color=c, lw=2)
    for ax, t in zip(axes, ("Cgs = -dQg/dVs", "Cgd = -dQg/dVd", "Cgg = dQg/dVg")):
        style(ax)
        ax.set(xlabel="VDS (V)", ylabel="pF", title=t + "   (W160 L20)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Proposed model vs VDS -- standard Ward-Dutton behaviour, NOT validated: no "
                 "usable VDS-resolved split C-V exists in the data", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step5_vds.png"), dpi=110)
    plt.close(fig)


def main():
    df = load_all()
    cv = curves(df)
    pl = plateaus(df)
    P = load_params(pl)
    with open(os.path.join(OUT, "ac_model_params_final.json"), "w") as f:
        json.dump({k: v for k, v in P.items()}, f, indent=2)
    met, pts = metrics(cv, P)
    met.to_csv(os.path.join(OUT, "ac_validation_metrics.csv"), index=False, float_format="%.3f")
    pts.to_csv(os.path.join(OUT, "ac_validation_points.csv"), index=False, float_format="%.5g")
    fl = floating_check(df, P)
    fl.to_csv(os.path.join(OUT, "ac_floating_check.csv"), index=False, float_format="%.4f")
    base = baseline(pl, P)
    base.to_csv(os.path.join(OUT, "ac_baseline_comparison.csv"), index=False, float_format="%.5g")
    cons = conservation(P)
    cons.to_csv(os.path.join(OUT, "ac_conservation_check.csv"), index=False, float_format="%.3g")
    plot_cg(pts, P)
    plot_err(pts)
    plot_floating(df, P)
    plot_baseline(base)
    plot_vds(P)
    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.3g}"
    print({k: v for k, v in P.items() if k != "V0_per_curve"})
    print(met.to_string(index=False, float_format=f))
    print(fl.to_string(index=False, float_format=f))
    print(base.pivot_table(index=["W", "L", "quantity"], columns="model", values="ratio").to_string(float_format=f))
    print(cons.to_string(index=False))


if __name__ == "__main__":
    main()
