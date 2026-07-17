"""Quality-control and cleaning of raw a-GIZO TFT curve-tracer exports.

Each (W, L) geometry was measured on up to 4 device instances (top1, top2,
bot1, bot2 -- wafer positions), some with extra repeat runs, each producing
three sweeps:
  - "linear":     Id-Vg transfer sweep at fixed small VD (0.1V)
  - "saturation": Id-Vg transfer sweep at fixed large VD (5V)
  - "output":     Id-Vd family-of-curves sweep (VD swept per fixed VG)

The linear and saturation sweeps are VG sweeps that go -5V -> +5V -> -5V (a
forward leg then a return leg, 202 rows = 101 + 101). Only the forward leg
is physically meaningful for transfer-curve fitting and the sub -2V region
is off-state noise floor, so both are trimmed to the forward leg restricted
to VG in [-2, 5] before anything else happens (QC, plots, merged dataset).

A number of device instances are non-functional (dead/open contact) or
unstable and must be dropped before the data is used to train a model.

This script:
  1. Parses data/*.csv into (W, L, device, kind) records for all 3 kinds.
  2. Scores each device's linear, saturation and output curves against
     basic transistor sanity checks (on/off ratio, monotonic turn-on,
     off-state stability, VG ordering).
  3. A device replicate is kept only if ALL THREE of its sweeps pass --
     it's the same physical device, so if it's dead/unstable in one sweep
     it's not a trustworthy data source for the others either.
  4. Writes data_cleaned/W{W}_L{L}_{linear,output,saturation}_clean.csv,
     concatenating only the retained device replicates for that geometry,
     plus a cleaning_manifest.json recording what was kept/dropped and why.
  5. Writes data_cleaned/merged_ann_dataset.csv: all three cleaned sweep
     kinds combined into one long-format (VG, VD, W, L, ID) table, ready
     to train the ANN (src/dataset.py) on the full operating region.

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

LINEAR_OUTPUT_RE = re.compile(r'tft (linear|output) \[s5tft(\d+)-(\d+)(top|bot)(\d+)(rep)?\((\d+)\)')
SATURATION_RE = re.compile(r'TFT Saturation \[S5TFT(\d+)-(\d+)(TOP|BOT)(\d+)(REP)?\((\d+)\)', re.IGNORECASE)

VG_MIN_KEEP = -2.0


def parse_saturation_file(path):
    """The B1500A 'TFT Saturation' export: a metadata header block, then a
    DataName header row and DataValue rows (comma-separated, one leading
    label column)."""
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("DataName"))
    cols = [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = [[v.strip() for v in l.split(",")[1:]] for l in lines[header_idx + 1:] if l.startswith("DataValue")]
    return pd.DataFrame(rows, columns=cols).apply(pd.to_numeric)


def trim_vg_forward_leg(df):
    """Keep only the forward VG sweep leg (first half of the 202-row -5V ->
    +5V -> -5V trace) restricted to VG >= VG_MIN_KEEP."""
    half = len(df) // 2
    fwd = df.iloc[:half].reset_index(drop=True)
    return fwd[fwd["VG"] >= VG_MIN_KEEP].reset_index(drop=True)


def inventory():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    recs = defaultdict(dict)  # (W, L, device) -> {'linear'/'output'/'saturation': path}

    for f in files:
        b = os.path.basename(f)
        if "final_ann_dataset" in b:
            continue
        m = LINEAR_OUTPUT_RE.search(b)
        if m:
            kind, W, L, pos, posnum, rep, _run = m.groups()
            device_id = f"{pos}{posnum}{'rep' if rep else ''}"
            recs[(int(W), int(L), device_id)][kind] = f
            continue
        m = SATURATION_RE.search(b)
        if m:
            W, L, pos, posnum, rep, _run = m.groups()
            device_id = f"{pos.lower()}{posnum}{'rep' if rep else ''}"
            recs[(int(W), int(L), device_id)]["saturation"] = f
    return recs


def qc_transfer_curve(df):
    """Id-Vg transfer curve (used for both linear VD=0.1 and saturation
    VD=5). Bounds are relative to the sweep's own VG range so this works
    whether VG spans -5..5 or the trimmed -2..5."""
    reasons = []
    df = df.sort_values("VG").reset_index(drop=True)
    vg = df["VG"].to_numpy()
    id_ = df["ID"].to_numpy()
    absid = np.abs(id_)

    if np.any(~np.isfinite(id_)) or len(df) < 5:
        return False, ["nan_or_too_short"], {}

    vg_lo, vg_hi = vg.min(), vg.max()
    off_mask = vg < vg_lo + 1.0
    on_mask = vg > vg_hi - 1.0
    off_level = np.median(absid[off_mask]) if off_mask.sum() > 3 else np.median(absid[:5])
    on_level = np.median(absid[on_mask]) if on_mask.sum() > 3 else absid[-5:].mean()
    on_off_ratio = on_level / max(off_level, 1e-16)

    log_absid = np.log10(np.clip(absid, 1e-16, None))
    k = max(3, len(log_absid) // 15)
    smooth = np.convolve(log_absid, np.ones(k) / k, mode="valid")
    frac_decreasing = np.mean(np.diff(smooth) < -0.02)
    net_rise = smooth[-1] - smooth[0]

    # Sign flips only matter once the device is genuinely "on" (well above
    # the off-state numerical floor) -- a fixed VG cutoff misfires on
    # high-VT devices that only turn on near the top of the sweep.
    on_state_mask = absid > max(off_level * 50, 1e-13)
    on_region = id_[on_state_mask]
    frac_negative_on = np.mean(on_region < 0) if len(on_region) > 3 else 0.0

    # Off-state should sit at the simulator's tiny numerical noise floor
    # with low spread. Large-magnitude, high-spread "noise" indicates real
    # measurement instability/oscillation, not just floor noise.
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


def load_kind(path, kind):
    if kind == "saturation":
        df = parse_saturation_file(path)
    else:
        df = pd.read_csv(path)
    if kind in ("linear", "saturation"):
        df = trim_vg_forward_leg(df)
    return df


def run_qc(recs):
    results = {}
    for (W, L, dev), paths in sorted(recs.items()):
        entry = {"W": W, "L": L, "device": dev}
        for kind in ("linear", "saturation", "output"):
            if kind not in paths:
                entry[f"{kind}_ok"] = False
                entry[f"{kind}_reasons"] = ["missing_file"]
                continue
            df = load_kind(paths[kind], kind)
            if kind == "output":
                ok, reasons, metrics = qc_output(df)
            else:
                ok, reasons, metrics = qc_transfer_curve(df)
            entry[f"{kind}_ok"] = ok
            entry[f"{kind}_reasons"] = reasons
            entry[f"{kind}_metrics"] = metrics
            entry[f"{kind}_path"] = paths[kind]
        results[f"W{W}_L{L}_{dev}"] = entry
    return results


def device_ok(v):
    return v["linear_ok"] and v["output_ok"] and v["saturation_ok"]


def build_cleaned(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_combo = defaultdict(list)
    for v in results.values():
        by_combo[(v["W"], v["L"])].append((v["device"], device_ok(v), v))

    manifest = []
    merged_frames = []
    for (W, L), devs in sorted(by_combo.items()):
        good = [(d, v) for d, ok, v in devs if ok]
        bad = [(d, v) for d, ok, v in devs if not ok]
        if not good:
            raise RuntimeError(f"No usable devices left for W={W} L={L}")

        for kind in ("linear", "output", "saturation"):
            frames = []
            for dev, v in good:
                df = load_kind(v[f"{kind}_path"], kind)
                df = df.copy()
                df.insert(0, "device", dev)
                frames.append(df)
            kind_df = pd.concat(frames, ignore_index=True)
            kind_df.to_csv(os.path.join(OUT_DIR, f"W{W}_L{L}_{kind}_clean.csv"), index=False)

            m = kind_df[["VG", "VD", "ID", "device"]].copy()
            m["W"] = W
            m["L"] = L
            m["sweep"] = kind
            merged_frames.append(m)

        manifest.append({
            "W": W, "L": L,
            "kept_devices": [d for d, _ in good],
            "dropped_devices": [d for d, _ in bad],
            "dropped_reasons": {
                d: {k: v[f"{k}_reasons"] for k in ("linear", "saturation", "output") if not v[f"{k}_ok"]}
                for d, v in bad
            },
        })

    with open(os.path.join(OUT_DIR, "cleaning_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    merged = pd.concat(merged_frames, ignore_index=True)
    merged["abs_ID"] = merged["ID"].abs()
    merged["log_ID"] = np.log10(merged["abs_ID"].clip(lower=1e-16))
    merged = merged[["VG", "VD", "W", "L", "ID", "abs_ID", "log_ID", "sweep", "device"]]
    merged.to_csv(os.path.join(OUT_DIR, "merged_ann_dataset.csv"), index=False)

    return manifest, merged


def main():
    recs = inventory()
    results = run_qc(recs)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "qc_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    manifest, merged = build_cleaned(results)

    total_kept = sum(len(m["kept_devices"]) for m in manifest)
    total_dropped = sum(len(m["dropped_devices"]) for m in manifest)
    print(f"{len(manifest)} W,L combos written to {OUT_DIR}")
    print(f"kept {total_kept} device-replicates, dropped {total_dropped} (require pass on all 3 sweeps)")
    print(f"merged_ann_dataset.csv: {len(merged)} rows")
    for m in manifest:
        if m["dropped_devices"]:
            print(f"  W{m['W']}_L{m['L']}: dropped {m['dropped_devices']}")


if __name__ == "__main__":
    main()
