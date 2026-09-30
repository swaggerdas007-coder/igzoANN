"""Step 2 -- inspect how every extracted parameter depends on geometry
before fitting anything: each parameter vs L (one line per W) and vs W (one
line per L), held-out devices marked, plus the alpha-kappa-G0 correlation
that the per-parameter scaling of step 3 cannot see.

Outputs: plots/step2_params_vs_L.png, plots/step2_params_vs_W.png,
plots/step2_correlations.png, step2_summary.csv (per-parameter spread and
simple trend statistics over the 16 training geometries)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, PLOTS, TRAIN_L, TRAIN_W, VH

SHOW = ["Von_lin", "Von_sat", "alpha_lin", "alpha_sat", "k_lin", "k_sat", "G0_lin", "G0_sat",
        "Ioff_lin", "Ioff_sat", "DeltaL", "RSD_kohm", "m", "G_VH_lin", "G_VH_sat"]
LABEL = {"G_VH_lin": "G0*exp(k*xH^a), lin (S)", "G_VH_sat": "G0*exp(k*xH^a), sat (S)",
         "RSD_kohm": "RSD (kOhm)", "DeltaL": "DeltaL (um)"}
LOGY = {"G0_lin", "G0_sat", "Ioff_lin", "Ioff_sat", "G_VH_lin", "G_VH_sat"}


def load():
    df = pd.read_csv(os.path.join(OUT, "step1_parameter_table.csv"))
    xh = VH - df.Von_eff
    for br in ("lin", "sat"):
        df[f"G_VH_{br}"] = df[f"G0_{br}"] * np.exp(df[f"k_{br}"] * xh ** df[f"alpha_{br}"])
    return df


def grid(df, xcol, group, fname, title):
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    groups = sorted(df[group].unique())
    cmap = plt.get_cmap("viridis")
    for ax, p in zip(axes.ravel(), SHOW):
        for i, gv in enumerate(groups):
            s = df[df[group] == gv].sort_values(xcol)
            tr = s[s.split == "train"]
            c = cmap(i / max(1, len(groups) - 1))
            ax.plot(tr[xcol], tr[p], "o-", color=c, label=f"{group}={gv}")
            ho = s[s.split == "holdout"]
            ax.plot(ho[xcol], ho[p], "s", color=c, mfc="white", mew=1.5, ms=7)
        out = df[(df.W == 160) & (df.L == 20)]
        ax.plot(out[xcol], out[p], "x", color="red", ms=10, mew=2)
        if p in LOGY:
            ax.set_yscale("log")
        ax.set_title(LABEL.get(p, p), fontsize=10)
        ax.set_xlabel(xcol + " (um)", fontsize=8)
        if xcol == "W":
            ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title + "   (filled = 16 training geometries, open squares = held out "
                 "W5/W10, red x = W160_L20)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=80)
    plt.close(fig)


def correlations(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for br, c in (("lin", "C0"), ("sat", "C3")):
        t = df[df.split == "train"]
        axes[0].plot(t[f"alpha_{br}"], t[f"k_{br}"], "o", color=c, label=br)
        axes[1].semilogy(t[f"alpha_{br}"], t[f"G0_{br}"], "o", color=c, label=br)
        axes[2].semilogy(t[f"alpha_{br}"], t[f"G_VH_{br}"], "o", color=c, label=br)
        r1 = np.corrcoef(t[f"alpha_{br}"], t[f"k_{br}"])[0, 1]
        r2 = np.corrcoef(t[f"alpha_{br}"], np.log(t[f"G0_{br}"]))[0, 1]
        axes[0].text(0.02, 0.95 if br == "lin" else 0.87, f"r({br}) = {r1:+.2f}",
                     transform=axes[0].transAxes, color=c)
        axes[1].text(0.02, 0.95 if br == "lin" else 0.87, f"r({br}) = {r2:+.2f}",
                     transform=axes[1].transAxes, color=c)
    axes[0].set(xlabel="alpha", ylabel="kappa", title="kappa vs alpha (16 training devices)")
    axes[1].set(xlabel="alpha", ylabel="G0 (S)", title="G0 vs alpha")
    axes[2].set(xlabel="alpha", ylabel="G at VGS = VH (S)",
                title="G0*exp(kappa*(VH-Von)^alpha): what the data pins down")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step2_correlations.png"), dpi=90)
    plt.close(fig)


def summary(df):
    t = df[df.split == "train"]
    rows = []
    for p in SHOW:
        y = np.log10(t[p]) if p in LOGY else t[p]
        # share of variance explained by the thesis form (quadratic L x linear W)
        X = np.c_[np.ones(len(t)), t.W, t.L, t.W * t.L, t.L ** 2, t.W * t.L ** 2]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2) if y.std() > 0 else np.nan
        rows.append(dict(parameter=p, log10=p in LOGY, median=float(np.median(t[p])),
                         min=float(t[p].min()), max=float(t[p].max()),
                         rel_spread=float(y.std() / abs(y.mean())) if p not in LOGY else float(y.std()),
                         corr_W=float(np.corrcoef(t.W, y)[0, 1]) if y.std() > 0 else np.nan,
                         corr_L=float(np.corrcoef(t.L, y)[0, 1]) if y.std() > 0 else np.nan,
                         r2_thesis_form=r2))
    return pd.DataFrame(rows)


def main():
    df = load()
    grid(df, "L", "W", "step2_params_vs_L.png", "Step 2: extracted parameters vs L")
    grid(df, "W", "L", "step2_params_vs_W.png", "Step 2: extracted parameters vs W")
    correlations(df)
    s = summary(df)
    s.to_csv(os.path.join(OUT, "step2_summary.csv"), index=False, float_format="%.4g")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
