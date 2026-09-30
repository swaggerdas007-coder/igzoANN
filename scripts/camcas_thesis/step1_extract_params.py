"""Step 1 -- per-device extraction of every CAMCAS parameter, following the
thesis's automatic procedure (Sec. 3.2) on the step-0 curves.

  (Von, Ioff)   derivative-onset point of the linear and saturation transfer
                curves (common.onset). The higher of the two Von's is the
                device's reference Von (thesis Sec. 3.3.2/3.3.3) for alpha,
                kappa, G0, and Ioff_lin / Ioff_sat are the currents of each
                curve at that Von (thesis Table 3.10).
  (DeltaL, RSD) TLM per width (thesis Fig. 3.1): Rtot = VDS/IDS at VDS=0.1 V
                vs L, one straight line per VGS, common intersection point
                = (DeltaL, RSD). The intersection is the least-squares point
                closest to all lines. RSD is a physical resistance, so the
                intersection is constrained to RSD >= 0 (reported alongside
                the unconstrained, thesis-literal value).
  (alpha, kappa) U-functions (Eqs. 3.3/3.4) on I' = IDS - Ioff:
                  U_lin = (I'/V') / d(I'/V')/dVGS,  V' = VDS - RSD*IDS
                  U_sat = (I'/x) / d(I'/x)/dVGS,    x  = VGS - Von
                ln U vs ln x is a straight line; slope a, intercept b give
                alpha = 1 - a, kappa = exp(-b)/alpha. Only valid points
                (U > 0, x > 0, device on: |IDS| > 1e-10 A) are used.
  G0            Eqs. 2.32/2.33 at VH = 5 V (the highest VGS).
  m             Eq. 2.35, m = 1/log2(Isat/Is), per on-state output trace:
                Isat = saturation-branch current (Eq. 2.31, VDS-independent)
                and Is = measured IDS at the VDS where the linear and
                saturation branches cross. Device value = median over traces.

Also reported: R^2 of every ln U regression, and each device's own-parameter
reconstruction error (the device's own extracted parameters in the Appendix F
equations) -- the floor any geometry scaling can reach.

Outputs: outputs_camcas_thesis/step1_parameter_table.csv (master table),
step1_tlm.csv, plots/step1_tlm.png, plots/step1_ufunction.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ALL_GEOMS, HOLDOUT, ON_LEVEL, OUT, PLOTS, TRAIN, VDS_LIN, VH,
                    camcas_ids, onset, tag)

TLM_VGS = np.arange(2.0, 5.01, 0.5)   # every device is >= 2 V above turn-on here
# Width-series devices excluded from the TLM, with the reason (see README):
TLM_EXCLUDE = {(160, 20): "low-mobility die (I*L/W 0.57x the typical device); "
                          "breaks the TLM's equal-sheet-resistance assumption"}


def curve(w, l, sweep):
    return pd.read_csv(os.path.join(OUT, "selected_curves", f"{tag(w, l)}_{sweep}.csv"))


def rtot(w, l, vg):
    g = curve(w, l, "linear")
    return VDS_LIN / np.interp(vg, g.VG, np.abs(g.ID))


def tlm(w, lengths):
    lines = np.array([np.polyfit(lengths, [rtot(w, l, vg) for l in lengths], 1) for vg in TLM_VGS])
    m, c = lines[:, 0], lines[:, 1]
    # point (x0, y0) minimizing sum_i (m_i*x0 + c_i - y0)^2
    x0, y0 = np.linalg.lstsq(np.c_[m, -np.ones_like(m)], -c, rcond=None)[0]
    xc, yc = (x0, y0) if y0 >= 0 else (-np.dot(m, c) / np.dot(m, m), 0.0)
    return dict(W=w, lengths=" ".join(map(str, lengths)), DeltaL_unconstrained=x0,
                RSD_unconstrained=y0, DeltaL=xc, RSD=yc), lines


def tlm_all():
    rows, lines = [], {}
    for w in sorted({w for w, _ in ALL_GEOMS}):
        ls = [l for ww, l in ALL_GEOMS if ww == w and (ww, l) not in TLM_EXCLUDE]
        if len(ls) < 2:
            continue
        r, ln = tlm(w, ls)
        rows.append(r)
        lines[w] = (ls, ln)
    return pd.DataFrame(rows).set_index("W"), lines


def ufit(vg, y, von):
    x = vg - von
    with np.errstate(divide="ignore", invalid="ignore"):
        u = y / np.gradient(y, vg)
    return x, u


def extract_alpha_kappa(x, u, on):
    ok = on & (x > 0) & np.isfinite(u) & (u > 0)
    lx, lu = np.log(x[ok]), np.log(u[ok])
    a, b = np.polyfit(lx, lu, 1)
    r2 = 1 - np.sum((lu - (a * lx + b)) ** 2) / np.sum((lu - lu.mean()) ** 2)
    alpha = 1.0 - a
    return alpha, np.exp(-b) / alpha, r2, (lx, lu, a, b)


def extract_m(w, l, p):
    o = curve(w, l, "output")
    von = max(p["Von_lin"], p["Von_sat"])
    lp = l - p["DeltaL"]
    ms = []
    for vg, t in o.groupby("VG"):
        x = vg - von
        if x < 1.5:
            continue
        t = t.sort_values("VD")
        G = p["G0_lin"] * (w / lp) * np.exp(p["k_lin"] * x ** p["alpha_lin"])
        i_sat = p["G0_sat"] * (w / lp) * np.exp(p["k_sat"] * x ** p["alpha_sat"]) * x + p["Ioff_sat"]
        vds_x = (i_sat * (1 + p["RSD"] * G) - p["Ioff_lin"]) / G
        if not 0 < vds_x < t.VD.max():
            continue
        i_s = np.interp(vds_x, t.VD, t.ID)
        if i_sat / i_s > 1:
            ms.append(1.0 / np.log2(i_sat / i_s))
    return (float(np.median(ms)) if ms else np.nan), len(ms)


def own_fit_error(w, l, p):
    """log10 MAE of the device's own parameters on its own curves, on-state
    points only (|ID| > ON_LEVEL; the off-state is the instrument floor)."""
    errs = {}
    for sweep in ("linear", "saturation", "output"):
        g = curve(w, l, sweep)
        g = g[(g.VD > 0) & (np.abs(g.ID) > ON_LEVEL)]
        pred = camcas_ids(g.VG.to_numpy(), g.VD.to_numpy(), w, l, p)
        errs[f"own_mae_dec_{sweep[:3]}"] = float(np.mean(np.abs(
            np.log10(pred) - np.log10(np.abs(g.ID.to_numpy())))))
    return errs


def main():
    os.makedirs(PLOTS, exist_ok=True)
    sel = pd.read_csv(os.path.join(OUT, "step0_selection.csv")).set_index(["W", "L"])
    tlm_df, tlm_lines = tlm_all()
    tlm_df.to_csv(os.path.join(OUT, "step1_tlm.csv"), float_format="%.6g")

    rows, udiag = [], {}
    for w, l in ALL_GEOMS:
        lin, sat = curve(w, l, "linear"), curve(w, l, "saturation")
        vg = lin.VG.to_numpy()
        il, isat = np.abs(lin.ID.to_numpy()), np.abs(sat.ID.to_numpy())
        von_l, ioff_l_own, _ = onset(vg, il)
        von_s, ioff_s_own, _ = onset(sat.VG, isat)
        von = max(von_l, von_s)
        k0 = int(np.argmin(np.abs(vg - von)))
        # Ioff: the current at Von (Table 3.10), averaged over the 5 points
        # ending at Von -- a single point is one sample of +-5 pA noise
        ioff_l, ioff_s = il[k0 - 4:k0 + 1].mean(), isat[k0 - 4:k0 + 1].mean()

        tw = w if w in tlm_df.index else min(tlm_df.index, key=lambda ww: abs(ww - w))
        dl, rsd = tlm_df.loc[tw, "DeltaL"], tlm_df.loc[tw, "RSD"]
        lp = l - dl

        vds_p = VDS_LIN - rsd * il
        x, u = ufit(vg, (il - ioff_l) / vds_p, von)
        a_l, k_l, r2_l, dl_ = extract_alpha_kappa(x, u, il > ON_LEVEL)
        with np.errstate(divide="ignore", invalid="ignore"):
            x, u = ufit(vg, (isat - ioff_s) / (vg - von), von)
        a_s, k_s, r2_s, ds_ = extract_alpha_kappa(x, u, isat > ON_LEVEL)
        udiag[(w, l)] = (dl_, ds_)

        xh = VH - von
        ih_l, ih_s = il[-1], isat[-1]
        g0_l = (ih_l - ioff_l) / ((w / lp) * np.exp(k_l * xh ** a_l) * (VDS_LIN - rsd * ih_l))
        g0_s = (ih_s - ioff_s) / ((w / lp) * np.exp(k_s * xh ** a_s) * xh)

        p = dict(Von_lin=von_l, Von_sat=von_s, alpha_lin=a_l, alpha_sat=a_s, k_lin=k_l,
                 k_sat=k_s, G0_lin=g0_l, G0_sat=g0_s, Ioff_lin=ioff_l, Ioff_sat=ioff_s,
                 DeltaL=dl, RSD=rsd, m=2.2)
        m_dev, n_m = extract_m(w, l, p)
        p["m"] = m_dev
        rows.append(dict(device=f"{tag(w, l)}_{sel.loc[(w, l), 'device']}", W=w, L=l,
                         split="holdout" if (w, l) in HOLDOUT else "train",
                         Von_eff=von, **p, RSD_kohm=rsd / 1e3,
                         DeltaL_RSD_source="TLM" if tw == w else f"TLM of W={tw} (nearest)",
                         Ioff_lin_own_onset=ioff_l_own, Ioff_sat_own_onset=ioff_s_own,
                         r2_lnU_lin=r2_l, r2_lnU_sat=r2_s, m_traces=n_m,
                         norm_ion_lin=ih_l * l / w, norm_ion_sat=ih_s * l / w,
                         **own_fit_error(w, l, p)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "step1_parameter_table.csv"), index=False, float_format="%.6g")
    plot_tlm(tlm_df, tlm_lines)
    plot_u(udiag)

    pd.set_option("display.width", 250)
    print(tlm_df.to_string(float_format=lambda v: f"{v:.4g}"))
    cols = ["device", "Von_lin", "Von_sat", "alpha_lin", "alpha_sat", "k_lin", "k_sat", "G0_lin",
            "G0_sat", "Ioff_lin", "Ioff_sat", "DeltaL", "RSD", "m", "r2_lnU_lin", "r2_lnU_sat",
            "own_mae_dec_lin", "own_mae_dec_sat", "own_mae_dec_out"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def plot_tlm(tlm_df, tlm_lines):
    ws = list(tlm_lines)
    fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 4))
    for ax, w in zip(axes, ws):
        ls, lines = tlm_lines[w]
        lx = np.linspace(-5, 22, 50)
        for vg, (m, c) in zip(TLM_VGS, lines):
            pts = [rtot(w, l, vg) / 1e3 for l in ls]
            h = ax.plot(ls, pts, "o", ms=4)[0]
            ax.plot(lx, (m * lx + c) / 1e3, "-", lw=0.8, color=h.get_color(), label=f"VGS={vg:g}")
        if w == 160:
            ax.plot([20], [rtot(160, 20, 5.0) / 1e3], "x", color="red", ms=9,
                    label="W160_L20 (excluded)")
        r = tlm_df.loc[w]
        ax.plot(r.DeltaL, r.RSD / 1e3, "k*", ms=12, label="intersection (RSD>=0)")
        ax.plot(r.DeltaL_unconstrained, r.RSD_unconstrained / 1e3, "k^", mfc="none", ms=8,
                label="unconstrained")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlim(-5, 22)
        ax.set_ylim(-20, None)
        ax.set_title(f"W = {w} um: dL={r.DeltaL:.2f} um, RSD={r.RSD / 1e3:.2f} kOhm", fontsize=9)
        ax.set_xlabel("L (um)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Rtot = VDS/IDS at VDS=0.1 V (kOhm)")
    axes[0].legend(fontsize=6)
    fig.suptitle("Step 1: TLM per width (thesis Fig. 3.1)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step1_tlm.png"), dpi=90)
    plt.close(fig)


def plot_u(udiag):
    fig, axes = plt.subplots(4, 5, figsize=(21, 15))
    for ax, (w, l) in zip(axes.ravel(), ALL_GEOMS):
        for (lx, lu, a, b), c, lab in zip(udiag[(w, l)], ("C0", "C3"), ("lin", "sat")):
            ax.plot(lx, lu, "o", ms=2.5, color=c, alpha=0.6)
            ax.plot(lx, a * lx + b, "-", color=c, label=f"{lab}: alpha={1 - a:.3f}")
        ax.set_title(tag(w, l), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Step 1: U-function linearization ln U vs ln(VGS - Von) (thesis Eqs. 3.3/3.4)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step1_ufunction.png"), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
