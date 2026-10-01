"""Step 1 -- inspect the measured C-V data before fitting anything.

Writes:
  ac_step1_summary.csv        per (geometry, family, kind, VD): plateau values,
                              plateau noise, mid-transition VG, record time
  plots/ac_step1_cv_basic.png C vs VG: cg, "cgd", "cgs" (VD = 0) + fine sweep
  plots/ac_step1_loss.png     G/omega vs VG (Cp-G loss term) -- flags where the
                              10 kHz signal is RC-limited (not quasi-static)
  plots/ac_step1_vd_family.png  total-gate C vs VG at each SMU drain bias
  plots/ac_step1_cf.png       C-f sweep (160x20, VG = 5 V)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvdata import GEOMS, OUT, PLOTS, load_all, load_cf

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e4e0"
KIND = {"cg": ("#2a78d6", 'cg (S+D on CMU-low)'),
        "cgd": ("#eb6834", '"cgd" (D on CMU-low, S floating)'),
        "cgs": ("#1baf7a", '"cgs" (S on CMU-low, D floating)')}
VD_RAMP = ["#c6dbf5", "#9fc3ee", "#74a7e6", "#4b8bdc", "#2a78d6", "#1d5aa6", "#123c70"]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def mid_vg(g):
    """VG where C crosses halfway between its off and on plateaus."""
    g = g.sort_values("VG")
    lo, hi = g.C.iloc[:5].mean(), g.C.iloc[-1]
    k = np.argmax(g.C.to_numpy() > (lo + hi) / 2)
    return float(np.interp((lo + hi) / 2, g.C.iloc[k - 1:k + 1], g.VG.iloc[k - 1:k + 1]))


def summary(df):
    rows = []
    for key, g in df.groupby(["W", "L", "family", "kind", "VD"], dropna=False):
        off = g[g.VG <= -2].C
        rows.append(dict(W=key[0], L=key[1], family=key[2], kind=key[3], VD=key[4],
                         time=g.time.iloc[0], C_off_pF=off.mean() * 1e12 if len(off) else np.nan,
                         C_off_noise_fF=off.std() * 1e15 if len(off) else np.nan,
                         C_at_VG5_pF=g[np.isclose(g.VG, 5)].C.mean() * 1e12,
                         VG_mid=mid_vg(g) if g.family.iloc[0] != "fine" else np.nan,
                         G_over_w_max_pF=(g.G / (2 * np.pi * g.f)).max() * 1e12,
                         VG_at_G_max=g.VG.iloc[int(np.argmax(g.G.to_numpy()))]))
    return pd.DataFrame(rows).sort_values(["W", "L", "time"])


def plot_basic(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (w, l) in zip(axes.ravel(), GEOMS):
        style(ax)
        for kind, (c, lab) in KIND.items():
            g = df[(df.family == "basic") & (df.kind == kind) & (df.W == w) & (df.L == l)]
            ax.plot(g.VG, g.C * 1e12, "-o", color=c, lw=2, ms=3.5, label=lab)
        f = df[(df.family == "fine") & (df.W == w) & (df.L == l)]
        ax.plot(f.VG, f.C * 1e12, "s", color=MUTED, ms=3, mfc="none",
                label="cg, fine sweep (taken earlier)")
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
        ax.set_xlabel("VG (V)", fontsize=9, color=MUTED)
        ax.set_ylabel("C (pF)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Measured gate capacitance at 10 kHz, 50 mV, VD = 0 "
                 "(B1500 CMU, Cp-G)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step1_cv_basic.png"), dpi=110)
    plt.close(fig)


def plot_loss(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (w, l) in zip(axes.ravel(), GEOMS):
        style(ax)
        for kind, (c, lab) in KIND.items():
            g = df[(df.family == "basic") & (df.kind == kind) & (df.W == w) & (df.L == l)]
            ax.plot(g.VG, g.G / (2 * np.pi * g.f) * 1e12, "-o", color=c, lw=2, ms=3, label=lab)
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
        ax.set_xlabel("VG (V)", fontsize=9, color=MUTED)
        ax.set_ylabel("G / (2 pi f)  (pF)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Loss term G/omega: peaks where the channel resistance limits charging "
                 "at 10 kHz (C there is not quasi-static)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step1_loss.png"), dpi=110)
    plt.close(fig)


def plot_vd(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (w, l) in zip(axes.ravel(), GEOMS):
        style(ax)
        g = df[(df.family == "vd") & (df.W == w) & (df.L == l)]
        for c, (vd, s) in zip(VD_RAMP, g.groupby("VD")):
            ax.plot(s.VG, s.C * 1e12, "-", color=c, lw=2,
                    label=f"VD = {vd:g} V  ({s.time.iloc[0][-8:]})")
        b = df[(df.family == "basic") & (df.kind == "cg") & (df.W == w) & (df.L == l)]
        ax.plot(b.VG, b.C * 1e12, "--", color=INK, lw=1.2,
                label=f"cg, VD = 0  ({b.time.iloc[0][-8:]})")
        ax.set_title(f"W = {w} um, L = {l} um", fontsize=10, color=INK)
        ax.set_xlabel("VG (V)", fontsize=9, color=MUTED)
        ax.set_ylabel("C (pF)", fontsize=9, color=MUTED)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle('"C-V Vd=x" family: gate C with an SMU on the drain (record time in '
                 'brackets; sweeps were taken in VD order)', fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step1_vd_family.png"), dpi=110)
    plt.close(fig)


def plot_cf():
    _, d = load_cf()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style(ax)
    ax.semilogx(d.Freq, d.Cp * 1e12, "-o", color="#2a78d6", lw=2, ms=3)
    ax.axvline(1e4, color=MUTED, lw=1, ls=":")
    ax.annotate("C-V sweeps: 10 kHz", (1.1e4, d.Cp.min() * 1e12), fontsize=8, color=MUTED)
    ax.set_xlabel("frequency (Hz)", fontsize=9, color=MUTED)
    ax.set_ylabel("Cp (pF)", fontsize=9, color=MUTED)
    ax.set_title(f"W160 L20 total gate C vs frequency at VG = {d.DcMon.mean():.2f} V",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ac_step1_cf.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    df = load_all()
    s = summary(df)
    s.to_csv(os.path.join(OUT, "ac_step1_summary.csv"), index=False, float_format="%.5g")
    plot_basic(df)
    plot_loss(df)
    plot_vd(df)
    plot_cf()
    pd.set_option("display.width", 220)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
