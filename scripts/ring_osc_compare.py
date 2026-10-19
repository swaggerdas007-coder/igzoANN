"""Four ring oscillators -- 3, 7, 11 and 23 stages -- of the recommended gate.

Same inverter in all four (W1/W2/W3/W4 = 5/20/160/40 um, L = 5 um, Cboot = 1 pF,
chosen in scripts/pseudo_cmos_best.py), so every difference below is the ring
length and nothing else.

    python scripts/ring_osc_compare.py

Writes ring_compare.csv, ring_compare_waveforms.png, ring_compare_performance.png.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pseudo_cmos import Sizing                       # noqa: E402
from src.ring import run_ring_auto                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)

NS = [3, 7, 11, 23]
VDDS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
VNOM = 3.0
TPD_REF = 515e-9           # at VNOM, for sizing the first transient window

# dataviz reference palette, categorical slots 1-4, assigned in fixed order.
# Validated (light, surface #fcfcfb): all checks pass; the contrast WARN on
# slots 3 and 4 is relieved by direct labels on every series, which are shipped.
COLORS = {3: "#2a78d6", 7: "#eb6834", 11: "#1baf7a", 23: "#eda100"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#d8d7d2"

KEEP = ["nstage", "vdd", "freq", "period", "swing", "swing_pct", "vmin", "vmax",
        "x_max", "boost", "power", "i_supply", "e_per_cycle", "e_per_transition",
        "tpd", "jitter", "n_periods", "mode_index", "oob_frac", "steps",
        "oscillates", "reason"]


def gate():
    r = pd.read_csv(os.path.join(OUT, "recommended_gate.csv")).iloc[0]
    return Sizing(w1=r.w1 * 1e-6, w2=r.w2 * 1e-6, w3=r.w3 * 1e-6,
                  w4=r.w4 * 1e-6, cboot=float(r.cboot))


def _job(a):
    s, n, vdd = a
    t_end = max(9.0 * 2.0 * n * TPD_REF * (VNOM / vdd) ** 1.1, 20e-6)
    try:
        r, _ = run_ring_auto(s, "pCb" if s.cboot > 0 else "pC", n, vdd,
                             t_end=t_end, max_steps=600000)
    except Exception as e:
        r = dict(oscillates=False, reason=f"{type(e).__name__}: {e}")
    d = {k: r.get(k, np.nan) for k in KEEP}
    d.update(nstage=n, vdd=vdd)
    return d


def waveform(s, n, vdd=VNOM, ncycle=2.2):
    """Transient at the nominal supply, trimmed to the last `ncycle` periods."""
    t_end = 9.0 * 2.0 * n * TPD_REF
    r, (t, V, net, outs, xs) = run_ring_auto(s, "pCb", n, vdd, t_end=t_end,
                                             max_steps=600000)
    t0 = t[-1] - ncycle * r["period"]
    m = t >= t0
    return r, (t[m] - t0), V[m, net._idx[outs[0]]], V[m, net._idx[xs[0]]]


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10.5, pad=8)
    ax.grid(True, color=GRID, lw=.7, alpha=.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def end_labels(ax, items, group_px=11, gap_px=14, fmt=None, sort_keys=False):
    """Direct-label each series at its right end, in text ink, without collisions.

    Series whose ends land within `group_px` are merged into one label -- in these
    panels N = 7, 11 and 23 genuinely coincide, so "N = 7, 11, 23" on one line is
    both tidier and more informative than three labels fighting for the same
    pixel. Whatever is left is pushed apart to `gap_px`. Works on log axes
    because the arithmetic happens in display space and is mapped back.
    """
    if not items:
        return
    ax.figure.canvas.draw()
    T, Ti = ax.transData, ax.transData.inverted()
    pts = sorted(((T.transform((x, y))[1], x, key) for x, y, key in items))
    groups = []
    for py, x, key in pts:
        if groups and py - groups[-1][-1][0] < group_px:
            groups[-1].append((py, x, key))
        else:
            groups.append([(py, x, key)])
    merged = [(float(np.mean([g[0] for g in grp])), max(g[1] for g in grp),
               sorted([g[2] for g in grp]) if sort_keys else [g[2] for g in grp])
              for grp in groups]
    for i in range(1, len(merged)):
        if merged[i][0] - merged[i - 1][0] < gap_px:
            merged[i] = (merged[i - 1][0] + gap_px, merged[i][1], merged[i][2])
    for py, x, keys in merged:
        y = Ti.transform((T.transform((x, ax.get_ylim()[0]))[0], py))[1]
        txt = fmt(keys) if fmt else ", ".join(str(k) for k in keys)
        ax.annotate(txt, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                    color=INK, fontsize=8.5, va="center", ha="left")


def fig_waveforms(waves):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 9.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, n in zip(axs, NS):
        r, t, vo, vx = waves[n]
        c = COLORS[n]
        tu = t * 1e6
        ax.axhline(VNOM, color=INK2, ls=(0, (4, 3)), lw=1, zorder=1)
        ax.plot(tu, vx, color=c, lw=1.3, alpha=.5, zorder=2)
        ax.plot(tu, vo, color=c, lw=2.2, zorder=3)
        style(ax, "time [us]" if n == NS[-1] else "", "V",
              f"{r['freq']/1e3:,.1f} kHz   ·   swing {r['swing_pct']:.1f}% of VDD"
              f"   ·   {r['tpd']*1e9:.0f} ns/stage   ·   "
              + (f"$V_X$ peaks {r['boost']:+.2f} V vs VDD"))
        ax.set_ylim(-0.4, 3.85)
        ax.set_xlim(0, tu[-1] * 1.035)
        end_labels(ax, [(tu[-1], vo[-1], "$V_{OUT}$"), (tu[-1], vx[-1], "$V_X$")],
                   group_px=5, gap_px=11,
                   fmt=lambda k: k[0] if len(k) == 1 else " / ".join(k))
        ax.text(0.006, 0.94, f"N = {n}", transform=ax.transAxes, color=INK,
                fontsize=12, fontweight="bold", va="top")
        if n == NS[0]:
            ax.annotate("VDD = 3 V", xy=(0.5, VNOM), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", color=INK2,
                        fontsize=8.5, va="bottom", ha="center")
    fig.suptitle("Four a-IGZO pseudo-CMOS bootstrapped ring oscillators, "
                 "identical gate (5/20/160/40 um, L = 5 um, Cboot = 1 pF), VDD = 3 V\n"
                 "$V_X$ is the bootstrap node: it rides above the supply in every "
                 "ring except the 3-stage one, which never gets there",
                 color=INK, fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "ring_compare_waveforms.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_performance(df):
    import matplotlib.pyplot as plt
    o = df[df.oscillates]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    fig.patch.set_facecolor(SURFACE)
    A = axs.ravel()
    panels = [
        ("freq", 1e-3, True, "f [kHz]", "oscillation frequency", None),
        ("swing_pct", 1.0, False, "swing [% of VDD]", "output swing", 90.0),
        ("tpd", 1e9, False, "$t_{pd}$ per stage [ns]", "delay per stage", None),
        ("power", 1e6, True, "P [uW]", "average supply power", None),
        ("e_per_transition", 1e12, True, "energy / transition [pJ]",
         "switching energy", None),
        ("boost", 1.0, False, "$V_{X,max}$ - VDD [V]", "bootstrap headroom", 0.0),
    ]
    for ax, (col, sc, logy, ylab, title, ref) in zip(A, panels):
        ends = []
        for n in NS:
            d = o[o.nstage == n].sort_values("vdd")
            if not len(d):
                continue
            y = d[col] * sc
            ax.plot(d.vdd, y, "-o", color=COLORS[n], lw=2, ms=6,
                    mec=SURFACE, mew=1.2, zorder=3)
            ends.append((d.vdd.iloc[-1], y.iloc[-1], n))
        if ref is not None:
            ax.axhline(ref, color=INK2, ls=(0, (4, 3)), lw=1, zorder=1)
        if logy:
            ax.set_yscale("log")
        style(ax, "VDD [V]", ylab, title)
        ax.set_xlim(0.8, 5.5)
        end_labels(ax, ends, sort_keys=True,
                   fmt=lambda ks: "N = " + ", ".join(str(k) for k in ks))
    A[1].text(0.95, 91.3, "90%", color=INK2, fontsize=8)
    A[5].text(0.95, 0.03, "VDD", color=INK2, fontsize=8)
    h = [plt.Line2D([], [], color=COLORS[n], lw=2.4, marker="o", ms=6,
                    mec=SURFACE, mew=1.2, label=f"{n} stages") for n in NS]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=10, labelcolor=INK)
    fig.suptitle("Ring length vs supply: the same inverter in 3, 7, 11 and 23 stages",
                 color=INK, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    p = os.path.join(OUT, "ring_compare_performance.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def table(df):
    n = df[(df.vdd == VNOM) & df.oscillates].sort_values("nstage")
    t = pd.DataFrame({
        "N": n.nstage.astype(int),
        "f [kHz]": n.freq / 1e3,
        "period [us]": n.period * 1e6,
        "tpd/stage [ns]": n.tpd * 1e9,
        "swing [% VDD]": n.swing_pct,
        "VOH [V]": n.vmax,
        "VOL [V]": n.vmin,
        "VX max [V]": n.x_max,
        "boost [V]": n.boost,
        "P [uW]": n.power * 1e6,
        "P/stage [uW]": n.power * 1e6 / n.nstage,
        "E/transition [pJ]": n.e_per_transition * 1e12,
        "jitter [ps]": n.jitter * 1e12,
    })
    return t


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    s = gate()
    print(f"gate: W1/W2/W3/W4 = {s.w1*1e6:.0f}/{s.w2*1e6:.0f}/{s.w3*1e6:.0f}/"
          f"{s.w4*1e6:.0f} um, L = {s.l*1e6:.0f} um, Cboot = {s.cboot*1e12:.1f} pF")
    csv = os.path.join(OUT, "ring_compare.csv")
    if os.path.exists(csv) and "--refresh" not in sys.argv:
        df = pd.read_csv(csv)
        print(f"reusing {os.path.basename(csv)} (--refresh to re-simulate)")
    else:
        jobs = [(s, n, v) for n in NS for v in VDDS]
        print(f"{len(jobs)} ring transients ({len(NS)} lengths x "
              f"{len(VDDS)} supplies)")
        with Pool(os.cpu_count()) as pool:
            df = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
        df.to_csv(csv, index=False)

    print(f"\n--- at VDD = {VNOM} V ---")
    print(table(df).to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    print("\n--- supply window per ring ---")
    for n in NS:
        d = df[(df.nstage == n) & df.oscillates]
        s90 = d[d.swing_pct >= 90]
        print(f"  N = {n:2d}: oscillates {d.vdd.min():.2f}-{d.vdd.max():.2f} V, "
              f"f = {d.freq.min()/1e3:7.2f}-{d.freq.max()/1e3:7.2f} kHz"
              + (f", swing >= 90% over {s90.vdd.min():.1f}-{s90.vdd.max():.1f} V"
                 if len(s90) else
                 f", peak swing {d.swing_pct.max():.0f}% (never reaches 90%)"))

    wcache = os.path.join(OUT, "ring_compare_waveforms.npz")
    if os.path.exists(wcache) and "--refresh" not in sys.argv:
        z = np.load(wcache, allow_pickle=True)
        waves = {n: (z[f"r{n}"].item(), z[f"t{n}"], z[f"o{n}"], z[f"x{n}"])
                 for n in NS}
        print(f"\nreusing {os.path.basename(wcache)} (--refresh to re-simulate)")
    else:
        waves = {n: waveform(s, n) for n in NS}
        np.savez(wcache, **{f"{k}{n}": v for n in NS
                            for k, v in zip(("r", "t", "o", "x"), waves[n])})
    print("wrote " + fig_waveforms(waves))
    print("wrote " + fig_performance(df))
    print(f"wrote {OUT}/ring_compare.csv")
