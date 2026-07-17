"""Build data_cleaned_2/: an even more strictly cleaned dataset that keeps
only the SINGLE best device replicate per (W, L) geometry, instead of
concatenating every device that passed QC (data_cleaned/'s approach).

Rationale: different device replicates at the same (W, L) are different
physical devices with their own threshold-voltage/mobility variation, but
the model only ever sees (VG, VD, W, L) -- it has no way to know which
device a row came from. Mixing several devices' worth of transfer curves
under one (W, L) label teaches the model contradictory examples (same
input, different device-to-device output). Picking one representative
"best" device per geometry removes that ambiguity.

A device is only a *candidate* if it already passes the same three-sweep
QC used for data_cleaned/ (scripts/clean_transistor_curves.py). Among
candidates for a geometry, the best one maximizes:

    score = sum_kind( log10(on_off_ratio_kind) ) - 2 * sum_kind(non_monotonicity_kind)

i.e. the cleanest on/off separation with the least non-monotonic wiggle,
averaged across linear, saturation and output.

Writes, per (W, L):
  - W{W}_L{L}_linear_best.csv
  - W{W}_L{L}_saturation_best.csv
  - W{W}_L{L}_output_best.csv
plus best_device_manifest.json (chosen device + score + runner-up scores)
and merged_ann_dataset.csv (all three kinds of the single best device per
geometry, long-format, ready for src/dataset.py).

Run (after scripts/clean_transistor_curves.py, which this imports from):
    python scripts/select_best_devices.py
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
from clean_transistor_curves import inventory, run_qc, load_kind, device_ok  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "data_cleaned_2")


def quality_score(v):
    lin, sat, out = v["linear_metrics"], v["saturation_metrics"], v["output_metrics"]
    log_on_off = sum(
        math.log10(max(m["on_off_ratio"], 1e-6))
        for m in (lin, sat, out)
    )
    non_monotonicity = lin["frac_decreasing"] + sat["frac_decreasing"] + out["frac_decr_mean"]
    return log_on_off - 2.0 * non_monotonicity


def main():
    recs = inventory()
    results = run_qc(recs)

    by_combo = defaultdict(list)
    for v in results.values():
        by_combo[(v["W"], v["L"])].append(v)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    merged_frames = []

    for (W, L), devs in sorted(by_combo.items()):
        candidates = [v for v in devs if device_ok(v)]
        if not candidates:
            raise RuntimeError(f"No usable devices left for W={W} L={L}")

        scored = sorted(
            ((quality_score(v), v["device"], v) for v in candidates),
            key=lambda t: t[0], reverse=True,
        )
        best_score, best_dev, best_v = scored[0]

        for kind in ("linear", "output", "saturation"):
            df = load_kind(best_v[f"{kind}_path"], kind).copy()
            df.insert(0, "device", best_dev)
            df.to_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_{kind}_best.csv"), index=False)

            m = df[["VG", "VD", "ID", "device"]].copy()
            m["W"] = W
            m["L"] = L
            m["sweep"] = kind
            merged_frames.append(m)

        manifest.append({
            "W": W, "L": L,
            "chosen_device": best_dev,
            "chosen_score": best_score,
            "candidates": [{"device": d, "score": s} for s, d, _ in scored],
        })

    with open(os.path.join(OUT_DIR, "best_device_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    merged = pd.concat(merged_frames, ignore_index=True)
    merged["abs_ID"] = merged["ID"].abs()
    merged["log_ID"] = np.log10(merged["abs_ID"].clip(lower=1e-16))
    merged = merged[["VG", "VD", "W", "L", "ID", "abs_ID", "log_ID", "sweep", "device"]]
    merged.to_csv(os.path.join(OUT_DIR, "merged_ann_dataset.csv"), index=False)

    print(f"{len(manifest)} W,L combos written to {OUT_DIR}")
    print(f"merged_ann_dataset.csv: {len(merged)} rows, one device per geometry")
    for m in manifest:
        print(f"  W{m['W']}_L{m['L']}: chose {m['chosen_device']} "
              f"(score={m['chosen_score']:.2f}) over {[c['device'] for c in m['candidates'][1:]]}")


if __name__ == "__main__":
    main()
