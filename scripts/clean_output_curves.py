"""Quality-check and cleaning pipeline for the raw a-GIZO TFT *output* curve
measurements in data/ (files named "tft output [...] ..._s5_W{W}_L{L}_{pos}_{n}.csv").

Each (W, L) combination was measured up to 4 times (top_1, top_2, bot_1, bot_2 --
device position on the wafer), occasionally re-run (e.g. "_run2") when a
measurement glitched. Every file is a full output-curve family: VG stepped from
-5V to 5V in 1V blocks, VD swept 0-5V (0.1V steps) within each block.

This script:
  1. Indexes every raw output-curve csv by (W, L, position, replicate).
  2. Scores each file against physical expectations for an n-type TFT output
     family (current increases with VG, ID(VD) is non-decreasing/saturating,
     no sign flips or spikes in the on-state, off-state near the noise floor).
  3. Renders a per-(W,L) diagnostic grid (Id-Vd and derived Id-Vg for every
     replicate) so the scoring can be checked visually.
  4. Writes a machine-readable summary (csv + json) used by
     build_clean_dataset.py to pick the 2 best replicates per (W, L) and to
     flag whole (W, L) combinations that never look right.
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
QC_PLOT_DIR = os.path.join(ROOT, "outputs", "plots", "output_curve_qc")
SUMMARY_CSV = os.path.join(ROOT, "outputs", "output_curve_qc_summary.csv")

FNAME_RE = re.compile(
    r"^tft output \[(?P<tag>[^;]+);.*\]\.csv_s5_W(?P<W>\d+)_L(?P<L>\d+)_"
    r"(?P<pos>top|bot)_(?P<rep>\d+)(?P<run>_run\d+)?\.csv$"
)

# On-state threshold: below this |ID| we're just looking at simulator/noise floor,
# not real channel current (see src/dataset.py docstring: leakage floor ~1e-14..1e-12A).
NOISE_FLOOR = 5e-11
OFF_STATE_VG_MAX = 0  # VG <= 0V is expected to be off for this n-type device


def index_files():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "tft output *.csv"))):
        fname = os.path.basename(path)
        m = FNAME_RE.match(fname)
        if not m:
            raise ValueError(f"Unrecognized output-curve filename: {fname}")
        rows.append({
            "path": path,
            "fname": fname,
            "W": int(m.group("W")),
            "L": int(m.group("L")),
            "pos": m.group("pos"),
            "rep": int(m.group("rep")),
            "run": m.group("run") or "",
            "label": f"{m.group('pos')}_{m.group('rep')}{m.group('run') or ''}",
        })
    return pd.DataFrame(rows)


def score_file(df):
    """Return a dict of diagnostic metrics for one output-curve csv.

    Lower `n_violations` / higher `on_state_frac_monotonic` etc. == cleaner data.
    """
    vg_levels = sorted(df["VG"].unique())
    on_levels = [vg for vg in vg_levels if vg > OFF_STATE_VG_MAX]
    off_levels = [vg for vg in vg_levels if vg <= OFF_STATE_VG_MAX]

    vd_monotonic_fracs = []
    sign_flip_counts = []
    spike_counts = []
    max_id_by_vg = {}

    for vg in on_levels:
        s = df[df["VG"] == vg].sort_values("VD")
        id_vals = s["ID"].to_numpy()
        max_id_by_vg[vg] = np.max(id_vals)

        # Fraction of adjacent VD steps where current doesn't increase
        # (allow 2% relative slack for measurement noise near saturation).
        diffs = np.diff(id_vals)
        tol = 0.02 * np.maximum(np.abs(id_vals[:-1]), NOISE_FLOOR)
        vd_monotonic_fracs.append(np.mean(diffs >= -tol))

        # Sign flips once the current is clearly above the noise floor.
        on_mask = np.abs(id_vals) > NOISE_FLOOR
        signs = np.sign(id_vals[on_mask])
        sign_flip_counts.append(int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0)

        # Spikes: points that jump >5x above both neighbors (log space) then fall back.
        if len(id_vals) > 2:
            log_abs = np.log10(np.maximum(np.abs(id_vals), 1e-15))
            spike = (log_abs[1:-1] - log_abs[:-2] > 0.7) & (log_abs[1:-1] - log_abs[2:] > 0.7)
            spike_counts.append(int(np.sum(spike)))
        else:
            spike_counts.append(0)

    # VG ordering: at fixed VD, higher VG should give >= current (10% slack).
    vg_order_violations = 0
    vg_order_checks = 0
    vd_values = sorted(df["VD"].unique())
    for vd in vd_values:
        s = df[df["VD"] == vd].sort_values("VG")
        s_on = s[s["VG"] > OFF_STATE_VG_MAX]
        id_vals = s_on["ID"].to_numpy()
        for i in range(len(id_vals) - 1):
            vg_order_checks += 1
            tol = 0.1 * max(abs(id_vals[i]), NOISE_FLOOR)
            if id_vals[i + 1] < id_vals[i] - tol:
                vg_order_violations += 1

    off_state_max = float(np.max(np.abs(df[df["VG"].isin(off_levels)]["ID"]))) if off_levels else 0.0

    # Real output curves rise with VD along the strongest VG trace, and separate
    # out across VG at fixed VD. A device with a shorted/floating gate or a
    # compliance-limited probe instead reads back a near-constant current that
    # is technically "monotonic" and "ordered" but physically meaningless --
    # neither check above catches that, so score it directly.
    vg_top = max(on_levels)
    top_trace = df[df["VG"] == vg_top].sort_values("VD")["ID"].to_numpy()
    id_at_vdlo = top_trace[np.argmin(np.abs(df[df["VG"] == vg_top].sort_values("VD")["VD"].to_numpy() - 0.5))]
    id_at_vdhi = top_trace[-1]
    vd_rise_ratio = abs(id_at_vdhi) / max(abs(id_at_vdlo), NOISE_FLOOR)

    vg_bottom_on = min(on_levels)
    s_hi = df[(df["VG"] == vg_top) & (df["VD"] == vd_values[-1])]["ID"]
    s_lo = df[(df["VG"] == vg_bottom_on) & (df["VD"] == vd_values[-1])]["ID"]
    id_vg_hi = float(np.abs(s_hi).mean()) if len(s_hi) else 0.0
    id_vg_lo = float(np.abs(s_lo).mean()) if len(s_lo) else 0.0
    vg_modulation_ratio = id_vg_hi / max(id_vg_lo, NOISE_FLOOR)

    return {
        "vd_monotonic_frac": float(np.mean(vd_monotonic_fracs)) if vd_monotonic_fracs else np.nan,
        "sign_flips": int(np.sum(sign_flip_counts)),
        "spikes": int(np.sum(spike_counts)),
        "vg_order_violation_frac": (vg_order_violations / vg_order_checks) if vg_order_checks else np.nan,
        "off_state_max_abs_id": off_state_max,
        "on_state_max_abs_id": float(np.max(list(max_id_by_vg.values()))) if max_id_by_vg else 0.0,
        "off_state_leaky": off_state_max > 5e-9,
        "vd_rise_ratio": float(vd_rise_ratio),
        "vg_modulation_ratio": float(vg_modulation_ratio),
        "no_vd_dependence": bool(vd_rise_ratio < 3.0),
        "no_vg_modulation": bool(vg_modulation_ratio < 3.0),
    }


def composite_score(row):
    """Higher is better. Penalize every failure mode; used only to rank
    replicates within a (W, L) group, not as an absolute pass/fail cutoff."""
    penalty = 0.0
    penalty += (1 - row["vd_monotonic_frac"]) * 40
    penalty += (row["vg_order_violation_frac"]) * 40
    penalty += min(row["sign_flips"], 20) * 0.5
    penalty += min(row["spikes"], 20) * 1.0
    penalty += 15.0 if row["off_state_leaky"] else 0.0
    # These two catch a dead/disconnected device reading back a flat, physically
    # meaningless current -- worse than any of the noise-shape issues above.
    penalty += 60.0 if row["no_vd_dependence"] else 0.0
    penalty += 60.0 if row["no_vg_modulation"] else 0.0
    return 100.0 - penalty


def plot_group(w, l, group_df, out_path):
    n = len(group_df)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8), dpi=130, squeeze=False)
    fig.patch.set_facecolor("white")

    for col, (_, row) in enumerate(group_df.iterrows()):
        df = pd.read_csv(row["path"])
        vg_levels = sorted(df["VG"].unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(vg_levels)))

        ax = axes[0][col]
        for vg, color in zip(vg_levels, cmap):
            s = df[df["VG"] == vg].sort_values("VD")
            ax.plot(s["VD"], s["ID"] * 1e6, color=color, lw=1.3, label=f"{vg:g}V")
        ax.set_xlabel("VD (V)")
        ax.set_ylabel(r"$I_D$ ($\mu$A)")
        ax.set_title(f"{row['label']}  Id-Vd\nscore={row['score']:.0f}")
        ax.grid(alpha=0.25)
        if col == n - 1:
            ax.legend(fontsize=6, frameon=False, title="VG", loc="upper left")

        ax = axes[1][col]
        vd_levels = sorted(df["VD"].unique())
        vd_pick = [vd_levels[i] for i in np.linspace(0, len(vd_levels) - 1, 5).astype(int)]
        cmap2 = plt.cm.plasma(np.linspace(0.1, 0.85, len(vd_pick)))
        for vd, color in zip(vd_pick, cmap2):
            s = df[df["VD"] == vd].sort_values("VG")
            ax.plot(s["VG"], np.abs(s["ID"]), color=color, lw=1.3, label=f"{vd:g}V")
        ax.set_xlabel("VG (V)")
        ax.set_ylabel(r"$|I_D|$ (A)")
        ax.set_yscale("log")
        ax.set_ylim(1e-13, None)
        ax.set_title(f"{row['label']}  Id-Vg (derived)")
        ax.grid(alpha=0.25)
        if col == n - 1:
            ax.legend(fontsize=6, frameon=False, title="VD", loc="lower right")

    fig.suptitle(f"W={w}um, L={l}um -- all replicates", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    os.makedirs(QC_PLOT_DIR, exist_ok=True)
    index = index_files()
    print(f"Indexed {len(index)} output-curve files "
          f"across {index.groupby(['W', 'L']).ngroups} (W, L) combinations")

    metric_rows = []
    for _, row in index.iterrows():
        df = pd.read_csv(row["path"])
        metrics = score_file(df)
        metric_rows.append({**row.to_dict(), **metrics})

    summary = pd.DataFrame(metric_rows)
    summary["score"] = summary.apply(composite_score, axis=1)
    summary = summary.sort_values(["W", "L", "score"], ascending=[True, True, False])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote per-file QC summary to {SUMMARY_CSV}")

    for (w, l), group in summary.groupby(["W", "L"]):
        out_path = os.path.join(QC_PLOT_DIR, f"W{w}_L{l}.png")
        plot_group(w, l, group, out_path)
    print(f"Wrote {summary.groupby(['W', 'L']).ngroups} diagnostic grid images to {QC_PLOT_DIR}")


if __name__ == "__main__":
    main()
