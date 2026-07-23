"""Focused re-pass on just the "output" (Id-Vd family) sweep: score every raw
device's output curve on its own terms (not the linear/saturation-driven
pick in data_cleaned_2/), keep the single cleanest/most uniform device per
(W, L) geometry, and compile that into cleaned_output_meas/.

A "proper, uniform" output family:
  - rises roughly monotonically with VD then flattens (saturates) for every
    VG member, not just the top one
  - orders correctly everywhere: at any fixed VD, a higher VG member should
    sit at or above a lower VG member's curve (no crossings)
  - has a healthy on/off separation between the top and bottom VG members
  - "bot" devices are preferred over "top" ones when scores are close --
    the wafer's "top" die position is disproportionately dead/marginal
    (see data_cleaned/README.md), so this is a soft, evidence-based tie
    breaker, not a hard rule.

Writes:
  - cleaned_output_meas/W{W}_L{L}_output_clean.csv (one file per geometry)
  - cleaned_output_meas/output_selection_manifest.json (winner + every
    candidate's score, for all 19 geometries)
  - cleaned_output_meas/merged_output_dataset.csv (all 19 chosen output
    curves concatenated, long-format, ready for src/dataset.py)

Run: python scripts/select_clean_output_curves.py
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from clean_transistor_curves import inventory  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "cleaned_output_meas")

BOT_BONUS = 1.5  # soft tie-breaker in favor of "bot" wafer-position devices


def score_output(df):
    """Returns (score, metrics). Higher score = cleaner, more uniform family."""
    vg_levels = sorted(df["VG"].unique())
    vd_max = df["VD"].max()

    frac_decr_all = []
    neg_frac_all = []
    for vgv in vg_levels:
        sub = df[df["VG"] == vgv].sort_values("VD")
        id_ = sub["ID"].to_numpy()
        if len(id_) < 4:
            continue
        diffs = np.diff(id_)
        span = max(np.abs(id_).max(), 1e-16)
        frac_decr_all.append(np.mean(diffs < -0.05 * span))
        neg_frac_all.append(np.mean(id_ < -0.05 * span))
    frac_decr_mean = float(np.mean(frac_decr_all)) if frac_decr_all else 1.0
    neg_frac_mean = float(np.mean(neg_frac_all)) if neg_frac_all else 1.0

    # Full pairwise VG ordering check (not just top-vs-bottom): at several VD
    # checkpoints, a higher VG member's |ID| should be >= a lower VG
    # member's. Count violating (VG_i < VG_j but ID_i > ID_j) pairs.
    checkpoints = [c for c in (1.0, 2.0, 3.0, 4.0, vd_max) if c <= vd_max]
    violations, total_pairs = 0, 0
    for vd_chk in checkpoints:
        near = df[(df["VD"] >= vd_chk - 0.1) & (df["VD"] <= vd_chk + 0.1)]
        levels = []
        for vgv in vg_levels:
            v = near[near["VG"] == vgv]["ID"].abs().median()
            if pd.notna(v):
                levels.append((vgv, v))
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                total_pairs += 1
                if levels[i][1] > levels[j][1]:  # lower VG has higher |ID| than higher VG
                    violations += 1
    crossing_frac = violations / total_pairs if total_pairs else 1.0

    near_vdmax = df[df["VD"] >= vd_max - 0.05]
    lo_id = near_vdmax[near_vdmax["VG"] == vg_levels[0]]["ID"].abs().median()
    hi_id = near_vdmax[near_vdmax["VG"] == vg_levels[-1]]["ID"].abs().median()
    on_off_ratio = (hi_id / max(lo_id, 1e-16)) if pd.notna(lo_id) and pd.notna(hi_id) else 0.0

    score = (
        math.log10(max(on_off_ratio, 1e-6))
        - 8.0 * frac_decr_mean
        - 8.0 * neg_frac_mean
        - 10.0 * crossing_frac
    )
    metrics = dict(on_off_ratio=on_off_ratio, frac_decr_mean=frac_decr_mean,
                    neg_frac_mean=neg_frac_mean, crossing_frac=crossing_frac,
                    n_vg_levels=len(vg_levels))
    return score, metrics


def main():
    recs = inventory()
    by_combo = defaultdict(list)
    for (W, L, dev), paths in recs.items():
        if "output" not in paths:
            continue
        by_combo[(W, L)].append((dev, paths["output"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    merged_frames = []

    for (W, L), devs in sorted(by_combo.items()):
        scored = []
        for dev, path in sorted(devs):
            df = pd.read_csv(path)
            score, metrics = score_output(df)
            is_bot = dev.startswith("bot")
            rank_score = score + (BOT_BONUS if is_bot else 0.0)
            scored.append((rank_score, dev, path, score, metrics, is_bot))
        scored.sort(key=lambda t: t[0], reverse=True)

        _, best_dev, best_path, best_raw_score, best_metrics, best_is_bot = scored[0]
        df = pd.read_csv(best_path)
        df = df.copy()
        df.insert(0, "device", best_dev)
        df.to_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_output_clean.csv"), index=False)

        m = df[["VG", "VD", "ID", "device"]].copy()
        m["W"] = W
        m["L"] = L
        merged_frames.append(m)

        manifest.append({
            "W": W, "L": L,
            "chosen_device": best_dev,
            "chosen_raw_score": best_raw_score - (BOT_BONUS if best_is_bot else 0.0),
            "chosen_rank_score": best_raw_score,
            "chosen_metrics": best_metrics,
            "candidates": [
                {"device": d, "raw_score": s, "rank_score": rs, "metrics": mt}
                for rs, d, _, s, mt, _ in scored
            ],
        })

    with open(os.path.join(OUT_DIR, "output_selection_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    merged = pd.concat(merged_frames, ignore_index=True)
    merged["abs_ID"] = merged["ID"].abs()
    merged["log_ID"] = np.log10(merged["abs_ID"].clip(lower=1e-16))
    merged = merged[["VG", "VD", "W", "L", "ID", "abs_ID", "log_ID", "device"]]
    merged.to_csv(os.path.join(OUT_DIR, "merged_output_dataset.csv"), index=False)

    print(f"{len(manifest)} W,L combos written to {OUT_DIR}")
    print(f"merged_output_dataset.csv: {len(merged)} rows")
    for m in manifest:
        others = [c["device"] for c in m["candidates"][1:]]
        print(f"  W{m['W']}_L{m['L']}: chose {m['chosen_device']} "
              f"(score={m['chosen_raw_score']:.2f}) over {others}")


if __name__ == "__main__":
    main()
