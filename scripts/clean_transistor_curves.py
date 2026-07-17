"""Quality-control and cleaning of raw a-GIZO TFT curve-tracer exports.

Each (W, L) geometry was measured on up to 4 device instances (top1, top2,
bot1, bot2 -- wafer positions), some with extra repeat runs, each producing
a "linear" (Id-Vg transfer sweep at fixed small VD) and an "output"
(Id-Vd family-of-curves sweep) csv. A number of those device instances are
non-functional (dead/open contact) or unstable and must be dropped before
the data is used to train a model.

This script:
  1. Parses data/*.csv into (W, L, device, kind) records.
  2. Scores each device's linear and output curve against basic transistor
     sanity checks (on/off ratio, monotonic turn-on, off-state stability).
  3. Writes data_cleaned/W{W}_L{L}_linear_clean.csv and
     data_cleaned/W{W}_L{L}_output_clean.csv, concatenating only the
     retained ("proper curve") device replicates for that geometry, plus a
     cleaning_manifest.json recording what was kept/dropped and why.

Run: python scripts/clean_transistor_curves.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "data_cleaned")

FILENAME_RE = re.compile(r'tft (linear|output) \[s5tft(\d+)-(\d+)(top|bot)(\d+)(rep)?\((\d+)\)')


def inventory():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    files = [f for f in files if "final_ann_dataset" not in f]
    recs = defaultdict(dict)  # (W, L, device) -> {'linear': path, 'output': path}
    for f in files:
        m = FILENAME_RE.search(os.path.basename(f))
        if not m:
            continue
        kind, W, L, pos, posnum, rep, _run = m.groups()
        device_id = f"{pos}{posnum}{'rep' if rep else ''}"
        recs[(int(W), int(L), device_id)][kind] = f
    return recs


def qc_linear(df):
    """Id-Vg transfer curve at fixed (small) VD."""
    reasons = []
    df = df.sort_values("VG").reset_index(drop=True)
    vg = df["VG"].to_numpy()
    id_ = df["ID"].to_numpy()
    absid = np.abs(id_)

    if np.any(~np.isfinite(id_)) or len(df) < 5:
        return False, ["nan_or_too_short"], {}

    off_mask = vg < -2.0
    on_mask = vg > 3.0
    off_level = np.median(absid[off_mask]) if off_mask.sum() > 3 else np.median(absid[:5])
    on_level = np.median(absid[on_mask]) if on_mask.sum() > 3 else absid[-5:].mean()
    on_off_ratio = on_level / max(off_level, 1e-16)

    log_absid = np.log10(np.clip(absid, 1e-16, None))
    k = max(3, len(log_absid) // 15)
    smooth = np.convolve(log_absid, np.ones(k) / k, mode="valid")
    frac_decreasing = np.mean(np.diff(smooth) < -0.02)
    net_rise = smooth[-1] - smooth[0]

    # Sign flips only matter once the device is genuinely "on" (well above the
    # off-state numerical floor) -- a fixed VG cutoff misfires on high-VT
    # devices that only turn on near VG=+3..+4.
    on_state_mask = absid > max(off_level * 50, 1e-13)
    on_region = id_[on_state_mask]
    frac_negative_on = np.mean(on_region < 0) if len(on_region) > 3 else 0.0

    # Off-state should sit at the simulator's numerical noise floor (tiny, low
    # spread). Large-magnitude, high-spread "noise" before turn-on indicates
    # real measurement instability/oscillation, not just floor noise.
    off_vals = absid[off_mask] if off_mask.sum() > 3 else absid[:5]
    off_log_std = float(np.std(np.log10(np.clip(off_vals, 1e-16, None))))
    off_median = float(np.median(off_vals))

    ok = True
    if on_off_ratio < 1e2:
        reasons.append(f"low_on_off_ratio({on_off_ratio:.1e})")
        ok = False
    if frac_decreasing > 0.35:
        reasons.append(f"non_monotonic(frac_decr={frac_decreasing:.2f})")
        ok = False
    if net_rise < 1.0:
        reasons.append(f"flat_curve(net_rise={net_rise:.2f}dec)")
        ok = False
    if frac_negative_on > 0.15:
        reasons.append(f"sign_flips_in_on_state({frac_negative_on:.2f})")
        ok = False
    if off_median > 1e-10 and off_log_std > 1.0:
        reasons.append(f"unstable_off_state(median={off_median:.1e},std={off_log_std:.2f}dec)")
        ok = False

    metrics = dict(on_off_ratio=on_off_ratio, frac_decreasing=frac_decreasing,
                    net_rise_decades=net_rise, frac_negative_on=frac_negative_on,
                    off_median=off_median, off_log_std=off_log_std)
    return ok, reasons, metrics


def qc_output(df):
    """Id-Vd output family: multiple VG sweeps of VD."""
    reasons = []
    ok = True
    vg_levels = sorted(df["VG"].unique())
    top_vgs = [v for v in vg_levels if v >= max(vg_levels) - 1.0] or vg_levels[-2:]

    frac_decr_list, neg_frac_list = [], []
    for vgv in top_vgs:
        sub = df[df["VG"] == vgv].sort_values("VD")
        id_ = sub["ID"].to_numpy()
        if len(id_) < 4:
            continue
        diffs = np.diff(id_)
        span = max(np.abs(id_).max(), 1e-16)
        frac_decr_list.append(np.mean(diffs < -0.05 * span))
        neg_frac_list.append(np.mean(id_ < -0.05 * span))

    frac_decr_mean = float(np.mean(frac_decr_list)) if frac_decr_list else 1.0
    neg_frac_mean = float(np.mean(neg_frac_list)) if neg_frac_list else 1.0

    vd_max = df["VD"].max()
    near_vdmax = df[df["VD"] >= vd_max - 0.05]
    ordering_ok = True
    on_off_ratio = 0.0
    if len(vg_levels) >= 2:
        lo_id = near_vdmax[near_vdmax["VG"] == vg_levels[0]]["ID"].abs().median()
        hi_id = near_vdmax[near_vdmax["VG"] == vg_levels[-1]]["ID"].abs().median()
        if pd.notna(lo_id) and pd.notna(hi_id):
            on_off_ratio = hi_id / max(lo_id, 1e-16)
            if hi_id <= lo_id * 3:
                ordering_ok = False

    if frac_decr_mean > 0.35:
        reasons.append(f"non_monotonic_on_curves(frac_decr={frac_decr_mean:.2f})")
        ok = False
    if neg_frac_mean > 0.15:
        reasons.append(f"sign_flips_on_curves({neg_frac_mean:.2f})")
        ok = False
    if not ordering_ok:
        reasons.append(f"bad_vg_ordering(on/off={on_off_ratio:.1e})")
        ok = False
    if on_off_ratio < 1e2:
        reasons.append(f"low_on_off_ratio({on_off_ratio:.1e})")
        ok = False

    metrics = dict(frac_decr_mean=frac_decr_mean, neg_frac_mean=neg_frac_mean,
                    ordering_ok=ordering_ok, on_off_ratio=on_off_ratio)
    return ok, reasons, metrics


def run_qc(recs):
    results = {}
    for (W, L, dev), paths in sorted(recs.items()):
        entry = {"W": W, "L": L, "device": dev}
        if "linear" in paths:
            ok, reasons, metrics = qc_linear(pd.read_csv(paths["linear"]))
            entry.update(linear_ok=ok, linear_reasons=reasons, linear_metrics=metrics,
                         linear_path=paths["linear"])
        else:
            entry.update(linear_ok=False, linear_reasons=["missing_file"])
        if "output" in paths:
            ok, reasons, metrics = qc_output(pd.read_csv(paths["output"]))
            entry.update(output_ok=ok, output_reasons=reasons, output_metrics=metrics,
                         output_path=paths["output"])
        else:
            entry.update(output_ok=False, output_reasons=["missing_file"])
        results[f"W{W}_L{L}_{dev}"] = entry
    return results


def build_cleaned(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_combo = defaultdict(list)
    for v in results.values():
        by_combo[(v["W"], v["L"])].append((v["device"], v["linear_ok"] and v["output_ok"], v))

    manifest = []
    for (W, L), devs in sorted(by_combo.items()):
        good = [(d, v) for d, ok, v in devs if ok]
        bad = [(d, v) for d, ok, v in devs if not ok]
        if not good:
            raise RuntimeError(f"No usable devices left for W={W} L={L}")

        lin_frames, out_frames = [], []
        for dev, v in good:
            ldf = pd.read_csv(v["linear_path"])
            ldf.insert(0, "device", dev)
            lin_frames.append(ldf)
            odf = pd.read_csv(v["output_path"])
            odf.insert(0, "device", dev)
            out_frames.append(odf)

        lin_out = pd.concat(lin_frames, ignore_index=True)
        out_out = pd.concat(out_frames, ignore_index=True)
        lin_out.to_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_linear_clean.csv"), index=False)
        out_out.to_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_output_clean.csv"), index=False)

        manifest.append({
            "W": W, "L": L,
            "kept_devices": [d for d, _ in good],
            "dropped_devices": [d for d, _ in bad],
            "dropped_reasons": {
                d: (v["linear_reasons"] if not v["linear_ok"] else []) +
                   (v["output_reasons"] if not v["output_ok"] else [])
                for d, v in bad
            },
            "n_linear_rows": len(lin_out),
            "n_output_rows": len(out_out),
        })

    with open(os.path.join(OUT_DIR, "cleaning_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    recs = inventory()
    results = run_qc(recs)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "qc_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    manifest = build_cleaned(results)

    total_kept = sum(len(m["kept_devices"]) for m in manifest)
    total_dropped = sum(len(m["dropped_devices"]) for m in manifest)
    print(f"{len(manifest)} W,L combos written to {OUT_DIR}")
    print(f"kept {total_kept} device-replicates, dropped {total_dropped}")
    for m in manifest:
        if m["dropped_devices"]:
            print(f"  W{m['W']}_L{m['L']}: dropped {m['dropped_devices']}")


if __name__ == "__main__":
    main()
