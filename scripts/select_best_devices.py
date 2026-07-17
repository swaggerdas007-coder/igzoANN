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
candidates, devices whose linear AND saturation curves both turn on at a
positive VG (positive threshold voltage -- empirically almost always the
"bot2" wafer position) are strongly preferred: negative-VT candidates are
only used as a fallback when a geometry has no positive-VT candidate at
all. Within whichever pool applies, the winner maximizes:

    score = sum_kind( log10(on_off_ratio_kind) ) - 2 * sum_kind(non_monotonicity_kind)

i.e. the cleanest on/off separation with the least non-monotonic wiggle,
averaged across linear, saturation and output.

Writes, per (W, L):
  - W{W}_L{L}_linear_best.csv
  - W{W}_L{L}_saturation_best.csv
  - W{W}_L{L}_output_best.csv
plus best_device_manifest.json (chosen device + score + VT + runner-ups)
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

# Constant-current threshold-voltage estimate. Fixed absolute thresholds
# (rather than a multiple of each device's own off-floor) because they need
# to sit solidly in the steep subthreshold slope for every W,L -- off-floor
# noise tops out around 1e-11..1e-10 A and on-currents reach 1e-6..1e-2 A,
# so 1e-9/1e-8 comfortably clears the floor without creeping into the knee.
VT_THRESHOLD_LINEAR = 1e-9
VT_THRESHOLD_SATURATION = 1e-8


def quality_score(v):
    lin, sat, out = v["linear_metrics"], v["saturation_metrics"], v["output_metrics"]
    log_on_off = sum(
        math.log10(max(m["on_off_ratio"], 1e-6))
        for m in (lin, sat, out)
    )
    non_monotonicity = lin["frac_decreasing"] + sat["frac_decreasing"] + out["frac_decr_mean"]
    return log_on_off - 2.0 * non_monotonicity


def estimate_vt(df, threshold):
    """VG at which |ID| first rises through `threshold`, log-interpolated.

    Searches only from the curve's global minimum onward, so a residual
    decaying-oscillation tail at the very start of the trimmed sweep (some
    devices still have a little of this right at VG=-2) can't be mistaken
    for an early "turn-on".
    """
    df = df.sort_values("VG").reset_index(drop=True)
    vg = df["VG"].to_numpy()
    absid = df["ID"].abs().to_numpy()
    start = int(np.argmin(absid))
    for i in range(start + 1, len(vg)):
        if absid[i] >= threshold:
            x0, x1 = vg[i - 1], vg[i]
            y0 = np.log10(max(absid[i - 1], 1e-16))
            y1 = np.log10(max(absid[i], 1e-16))
            if y1 == y0:
                return float(x1)
            frac = (np.log10(threshold) - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return None  # never reaches threshold after the valley


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

        scored = []
        for v in candidates:
            lin_df = load_kind(v["linear_path"], "linear")
            sat_df = load_kind(v["saturation_path"], "saturation")
            vt_lin = estimate_vt(lin_df, VT_THRESHOLD_LINEAR)
            vt_sat = estimate_vt(sat_df, VT_THRESHOLD_SATURATION)
            positive_vt = vt_lin is not None and vt_sat is not None and vt_lin > 0 and vt_sat > 0
            vt_sum = (vt_lin or -99) + (vt_sat or -99)
            score = quality_score(v)
            # Positive-VT candidates always outrank non-positive ones. Within
            # the positive tier, prefer the more positive VT (this is what
            # consistently favors "bot2"-like devices), quality_score breaks
            # ties. Within the fallback tier (no candidate is positive-VT),
            # rank by quality_score first -- a device that's merely "least
            # negative" by a hundredth of a volt but otherwise much noisier
            # is not a better choice than a clearly higher-quality curve.
            rank_key = (positive_vt, vt_sum, score) if positive_vt else (positive_vt, score, vt_sum)
            scored.append((rank_key, v["device"], v, vt_lin, vt_sat, positive_vt))

        scored.sort(key=lambda t: t[0], reverse=True)
        _, best_dev, best_v, best_vt_lin, best_vt_sat, best_positive = scored[0]
        best_score = quality_score(best_v)

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
            "chosen_vt_linear": best_vt_lin,
            "chosen_vt_saturation": best_vt_sat,
            "positive_vt_available": any(c[5] for c in scored),
            "chosen_is_positive_vt": best_positive,
            "candidates": [
                {"device": d, "score": quality_score(v), "vt_linear": vl, "vt_saturation": vs, "positive_vt": p}
                for _, d, v, vl, vs, p in scored
            ],
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
    n_fallback = sum(1 for m in manifest if not m["chosen_is_positive_vt"])
    print(f"{len(manifest) - n_fallback}/{len(manifest)} geometries got a positive-VT device; "
          f"{n_fallback} fell back to the least-negative-VT candidate")
    for m in manifest:
        flag = "" if m["chosen_is_positive_vt"] else "  [FALLBACK: no positive-VT candidate]"
        print(f"  W{m['W']}_L{m['L']}: chose {m['chosen_device']} "
              f"(VT_lin={m['chosen_vt_linear']:.2f}, VT_sat={m['chosen_vt_saturation']:.2f}){flag}")


if __name__ == "__main__":
    main()
