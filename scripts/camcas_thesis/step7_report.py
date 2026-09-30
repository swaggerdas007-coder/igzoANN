"""Step 7 -- report the final model: the fitted scaling equations with
their coefficients (markdown), the extracted parameters with the final
scaling functions overlaid, and the headline metrics table.

Outputs: scaling_equations.md, plots/step7_scaling_fits.png,
final_metrics.csv
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HOLDOUT, OUT, PARAMS, PLOTS, lin_W, poly_LW

NAME = {"Von_lin": "Von,lin (V)", "Von_sat": "Von,sat (V)", "alpha_lin": "alpha_lin",
        "alpha_sat": "alpha_sat", "k_lin": "kappa_lin", "k_sat": "kappa_sat",
        "G0_lin": "G0,lin (S)", "G0_sat": "G0,sat (S)", "Ioff_lin": "Ioff,lin (A)",
        "Ioff_sat": "Ioff,sat (A)"}


def equations(coef, variant):
    lines = [f"# Fitted CAMCAS scaling equations (variant `{variant}`)", "",
             "Form of every regime parameter (thesis Eqs. 3.5-3.16, Appendix F `*_func`), "
             "W and L in um:", "",
             "    X(L, W) = (aW*W + a0)*L^2 + (bW*W + b0)*L + (cW*W + c0)", "",
             "| parameter | aW | a0 | bW | b0 | cW | c0 |", "|---|---|---|---|---|---|---|"]
    for p in PARAMS:
        c = coef["poly"][p]
        lines.append(f"| {NAME[p]} | " + " | ".join(f"{c[k]:.6e}" for k in
                                                    ("aW", "a0", "bW", "b0", "cW", "c0")) + " |")
    dl, r = coef["DeltaL"], coef["RSD_kohm"]
    lines += ["", "Width-only relations (thesis Eqs. 3.17/3.18):", "",
              f"    DeltaL(W) = {dl['s']:.6e} * W + {dl['i']:.6e}    [um]",
              f"    RSD(W)    = max({r['s']:.6e} * W + {r['i']:.6e}, 0)    [kOhm]", "",
              f"Global smoothing exponent (Eq. 2.35, Appendix F `parameter m`): m = {coef['m']:.6g}",
              "", "Fit geometries (W/L um): " + ", ".join(f"{w}/{l}" for w, l in coef["fit_devices"]),
              "", "Held out (never used): 5/5, 10/5, 10/10"]
    if coef.get("excluded_from_fit"):
        lines.append("Excluded from the coefficient fit: " +
                     ", ".join(f"{w}/{l}" for w, l in coef["excluded_from_fit"]))
    return "\n".join(lines) + "\n"


def plot(coef, table, variant):
    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
    cmap = plt.get_cmap("viridis")
    lf = np.linspace(4, 21, 100)
    ws = (20, 40, 80, 160)
    shown = PARAMS + ["DeltaL", "RSD_kohm"]
    for ax, p in zip(axes.T.ravel(), shown):
        for i, w in enumerate(ws):
            c = cmap(i / 3)
            s = table[(table.W == w)].sort_values("L")
            ax.plot(s.L, s[p], "o", color=c, ms=6, label=f"W={w}")
            if p in PARAMS:
                ax.plot(lf, poly_LW(coef["poly"][p], w, lf), "-", color=c, lw=1.6)
            else:
                ax.axhline(lin_W(coef[p], w) if p == "DeltaL" else max(lin_W(coef[p], w), 0),
                           color=c, lw=1.6)
        for w, l in HOLDOUT:
            s = table[(table.W == w) & (table.L == l)]
            ax.plot(s.L, s[p], "s", color="#eb6834", mfc="white", mew=1.5, ms=7)
            if p in PARAMS:
                ax.plot(l, poly_LW(coef["poly"][p], w, l), "+", color="#eb6834", ms=10, mew=2)
        if p.startswith("G0"):
            ax.set_yscale("log")
        ax.set_title(NAME.get(p, p), fontsize=10)
        ax.set_xlabel("L (um)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Extracted per-device parameters (dots) and final scaling functions (lines), "
                 f"variant '{variant}'.  Open squares: unseen devices' extracted values; "
                 "orange +: model value there", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "step7_scaling_fits.png"), dpi=80)
    plt.close(fig)


def metrics(variant):
    s = pd.read_csv(os.path.join(OUT, f"step5_{variant}_summary.csv"))
    return s


def main():
    variant = open(os.path.join(OUT, "final_variant.txt")).read().strip()
    with open(os.path.join(OUT, f"step3_coefficients_{variant}.json")) as fh:
        coef = json.load(fh)
    table = pd.read_csv(os.path.join(OUT, "step1_parameter_table.csv"))
    with open(os.path.join(OUT, "scaling_equations.md"), "w") as fh:
        fh.write(equations(coef, variant))
    plot(coef, table, variant)
    m = metrics(variant)
    m.to_csv(os.path.join(OUT, "final_metrics.csv"), index=False, float_format="%.4f")
    print(open(os.path.join(OUT, "scaling_equations.md")).read())
    print(m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
