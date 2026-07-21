"""Quality-control and cleaning of raw a-GIZO TFT C-V (capacitance) exports for
the parasitic-capacitance ANNs (C_GD and C_GS) of Bahubalindruni et al.,
"InGaZnO TFT behavioral model for IC design", Analog Integr Circ Sig Process
(2016) 87:73-80.

The paper models the device with three ANNs joined as an equivalent circuit:
one for the drain current I_D (already handled by clean_transistor_curves.py /
src/train.py) and one each for the gate-drain and gate-source parasitic
capacitances C_GD and C_GS. This script prepares the training data for the two
capacitance ANNs.

Raw data
--------
The individual capacitance components were measured with a Keithley 4200-SCS /
B1500A CVU using the test setups of the paper's Fig. 3:
  - C_GD : dc bias on the gate, small-signal on the drain (gate grounds the
           signal), giving a direct C_GD measurement (files ".. cgd(..)").
  - C_GS : the drain grounds the signal (files ".. cgs(..)").

Each raw file is a single-port C-V sweep: the gate voltage V is swept -3 -> 5 V
in 0.2 V steps (41 points) at f = 10 kHz, with the other terminal dc bias held
at 0 V. Columns exported are V, Freq, C, G, fc. We keep C (in farads) as the
target and treat the swept V as the gate voltage VG and the fixed bias as
VD = 0 V, so the capacitance datasets share the exact (VG, VD, W, L) input
interface used by the I_D ANN (src/dataset.py). This keeps all three ANNs
pin-compatible for the Verilog-A equivalent circuit.

  files available : 4 geometries x {cgd, cgs}
      W-L in {20-20, 40-20, 160-15, 160-20}   (um)

IMPORTANT baseline caveat: every individual C_GD/C_GS sweep here was taken at a
single drain bias (VD = 0 V, linear region). There is therefore no VD variation
in this data, so a model trained on it learns C(VG, W, L) only and cannot yet
reproduce the C-VDS dependence of the paper's Fig. 6. This is a deliberate
first baseline; VD-resolved component measurements are needed to lift it.

This script:
  1. Parses the cgd/cgs files in data/ into (W, L, cap_type) records.
  2. Runs light QC (finite values, expected sweep length, positive pF-scale
     capacitance, monotone accumulation trend).
  3. Writes data_cleaned_cv/{cgd,cgs}_clean.csv, one row per swept point.
  4. Writes data_cleaned_cv/merged_cap_dataset.csv: long-format
     (cap_type, VG, VD, W, L, freq, C_F, C_pF), ready for src/train_cap.py.
  5. Writes data_cleaned_cv/cv_cleaning_manifest.json.

Run: python scripts/clean_cv_curves.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "data_cleaned_cv")

# e.g.  "C-V [160-20 cgd(5) ; 6_11_2026 2_37_02 PM].csv"
CV_RE = re.compile(r"\[(\d+)-(\d+)\s+(cgd|cgs)\((\d+)\)", re.IGNORECASE)

EXPECTED_POINTS = 41
VG_LO, VG_HI = -3.0, 5.0


def parse_cv_file(path):
    """Parse a Keithley/B1500A C-V export: a metadata header block, then a
    'DataName' header row followed by 'DataValue' rows. Returns (meta, df)."""
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    meta = {}
    for l in lines:
        if l.startswith("TestParameter,"):
            parts = [p.strip() for p in l.split(",")]
            if len(parts) >= 3:
                meta[parts[1]] = ",".join(parts[2:]).strip()

    header_idx = next(i for i, l in enumerate(lines) if l.startswith("DataName"))
    cols = [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = [[v.strip() for v in l.split(",")[1:]]
            for l in lines[header_idx + 1:] if l.startswith("DataValue")]
    df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric)
    return meta, df


def inventory():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    recs = {}  # (W, L, cap) -> path
    for f in files:
        m = CV_RE.search(os.path.basename(f))
        if not m:
            continue
        W, L, cap, _n = m.groups()
        recs[(int(W), int(L), cap.lower())] = f
    return recs


def qc_cv_curve(df):
    """Sanity checks for a single C-V accumulation sweep."""
    reasons = []
    ok = True
    v = df["VG"].to_numpy()
    c = df["C_pF"].to_numpy()

    if np.any(~np.isfinite(c)) or len(df) < 10:
        return False, ["nan_or_too_short"], {}

    if len(df) != EXPECTED_POINTS:
        reasons.append(f"unexpected_length({len(df)})")

    if np.any(c <= 0):
        reasons.append("nonpositive_capacitance")
        ok = False

    # Accumulation trend: on-state (high VG) capacitance should sit well above
    # the depletion/off-state (low VG) value.
    off_level = float(np.median(c[v < v.min() + 1.0]))
    on_level = float(np.median(c[v > v.max() - 1.0]))
    ratio = on_level / max(off_level, 1e-6)
    if ratio < 1.5:
        reasons.append(f"weak_accumulation(on/off={ratio:.2f})")
        ok = False

    metrics = dict(off_pF=off_level, on_pF=on_level, on_off_ratio=ratio,
                   c_min_pF=float(c.min()), c_max_pF=float(c.max()))
    return ok, reasons, metrics


def load_curve(path, W, L, cap):
    meta, df = parse_cv_file(path)
    vcol = "V" if "V" in df.columns else "VG"
    freq = float(meta.get("Measurement.Secondary.Frequency", "nan"))
    bias = float(meta.get("Measurement.Bias.Source", "0"))

    out = pd.DataFrame({
        "cap_type": cap,
        "VG": df[vcol].astype(float),   # swept gate voltage
        "VD": bias,                     # fixed drain dc bias (0 V here)
        "W": W,
        "L": L,
        "freq": freq,
        "C_F": df["C"].astype(float),
        "C_pF": df["C"].astype(float) * 1e12,
    })
    return meta, out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = inventory()
    if not recs:
        raise RuntimeError(f"No cgd/cgs C-V files found under {DATA_DIR}")

    manifest = []
    per_cap_frames = {"cgd": [], "cgs": []}

    for (W, L, cap), path in sorted(recs.items()):
        _meta, curve = load_curve(path, W, L, cap)
        ok, reasons, metrics = qc_cv_curve(curve)
        manifest.append({
            "W": W, "L": L, "cap_type": cap,
            "file": os.path.basename(path),
            "n_points": int(len(curve)),
            "kept": bool(ok),
            "reasons": reasons,
            "metrics": metrics,
        })
        if ok:
            per_cap_frames[cap].append(curve)
        else:
            print(f"  DROP {cap} W{W}_L{L}: {reasons}")

    merged_frames = []
    for cap, frames in per_cap_frames.items():
        if not frames:
            print(f"WARNING: no usable {cap} curves")
            continue
        cap_df = pd.concat(frames, ignore_index=True)
        cap_df.to_csv(os.path.join(OUT_DIR, f"{cap}_clean.csv"), index=False)
        merged_frames.append(cap_df)

    merged = pd.concat(merged_frames, ignore_index=True)
    merged = merged[["cap_type", "VG", "VD", "W", "L", "freq", "C_F", "C_pF"]]
    merged.to_csv(os.path.join(OUT_DIR, "merged_cap_dataset.csv"), index=False)

    with open(os.path.join(OUT_DIR, "cv_cleaning_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    kept = sum(m["kept"] for m in manifest)
    print(f"parsed {len(manifest)} C-V files, kept {kept}")
    for cap in ("cgd", "cgs"):
        sub = merged[merged.cap_type == cap]
        geos = sorted(set(zip(sub.W, sub.L)))
        print(f"  {cap}: {len(sub)} rows over {len(geos)} geometries {geos}")
    print(f"merged_cap_dataset.csv: {len(merged)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
