"""Build a tidy CGD/CGS training dataset from the raw C-V csvs in data/.

Parses the 12 "C-V [W-L kind(n) ; ...].csv" files (4 device geometries x
{cg, cgd, cgs}), where:
  - cg  : total gate capacitance C_G-DS, source & drain shorted to ground
          (paper's Fig. 3(a) setup)
  - cgd : gate-to-drain capacitance, drain swept/measured, gate grounded
          for the AC signal (paper's Fig. 3(b) setup)
  - cgs : gate-to-source capacitance, measured the mirrored way (source
          grounded for the AC signal)
All 12 files share the same VG sweep (-3 to 5 V, 0.2 V step) at a single
fixed bias of 0 V on the non-swept terminal -- i.e. VDS = 0 for every row.
There is no VDS-swept CGD/CGS split in this dataset (only the *combined*
total Cg is swept over VDS, in the separate "C-V Vd=*" family, which this
script does NOT use). So the dataset built here trains a VG-only (VDS~0)
capacitance model, not the full VDS-dependent surface of Bahubalindruni
et al. Fig. 5/6 -- see data_cv_cleaned/README.md.

Usage:
    python scripts/build_cv_dataset.py
"""
import glob
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "data_cv_cleaned")


def parse_cv_file(path):
    meta = {}
    header = None
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            tag = parts[0]
            if tag == "SetupTitle":
                meta["SetupTitle"] = parts[1] if len(parts) > 1 else ""
            elif tag == "MetaData" and len(parts) > 2 and parts[1] == "TestRecord.TestTarget":
                meta["TestTarget"] = parts[2]
            elif tag == "TestParameter" and len(parts) > 2 and parts[1] == "Measurement.Secondary.Frequency":
                meta["Frequency"] = float(parts[2])
            elif tag == "TestParameter" and len(parts) > 2 and parts[1] == "Measurement.Secondary.ACLevel":
                meta["ACLevel"] = float(parts[2])
            elif tag == "DataName":
                header = parts[1:]
            elif tag == "DataValue":
                if header is not None:
                    rows.append(parts[1:1 + len(header)])
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return meta, df


def device_from_target(target):
    m = re.match(r"(\d+)-(\d+)\s+(cgd|cgs|cg)$", target or "")
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "C-V [*")))  # excludes "C-V Vd=" and "C-V fc"
    assert len(files) == 12, f"expected 12 plain C-V files, found {len(files)}"

    records = {}  # (W, L) -> {"cg": df, "cgd": df, "cgs": df, meta...}
    freqs, aclevels = set(), set()
    for f in files:
        meta, df = parse_cv_file(f)
        W, L, kind = device_from_target(meta.get("TestTarget", ""))
        assert W is not None, f"could not parse device/kind from {f}: {meta.get('TestTarget')}"
        freqs.add(meta.get("Frequency"))
        aclevels.add(meta.get("ACLevel"))
        records.setdefault((W, L), {})[kind] = df.rename(columns={"C": kind.upper(), "V": "VG"})[["VG", kind.upper()]]

    assert len(freqs) == 1 and len(aclevels) == 1, "expected identical freq/AC level across all files"
    freq_hz, ac_level_v = freqs.pop(), aclevels.pop()

    long_rows = []
    per_device_frames = {}
    for (W, L), kinds in sorted(records.items()):
        assert set(kinds) == {"cg", "cgd", "cgs"}, f"missing kind(s) for W={W} L={L}: {set(kinds)}"
        merged = kinds["cg"].merge(kinds["cgd"], on="VG").merge(kinds["cgs"], on="VG")
        merged.insert(0, "L", L)
        merged.insert(0, "W", W)
        merged["VDS"] = 0.0
        merged = merged[["VG", "VDS", "W", "L", "CGD", "CGS", "CG"]]
        per_device_frames[(W, L)] = merged
        out_path = os.path.join(OUT_DIR, f"W{W}_L{L}_cv_clean.csv")
        merged.to_csv(out_path, index=False)
        long_rows.append(merged)

    merged_all = pd.concat(long_rows, ignore_index=True)
    merged_all.to_csv(os.path.join(OUT_DIR, "merged_cv_dataset.csv"), index=False)

    manifest = {
        "n_devices": len(per_device_frames),
        "devices_W_L": sorted([[int(w), int(l)] for (w, l) in per_device_frames]),
        "vg_range": [float(merged_all["VG"].min()), float(merged_all["VG"].max())],
        "vg_step": 0.2,
        "vds_fixed_at": 0.0,
        "frequency_hz": freq_hz,
        "ac_level_v": ac_level_v,
        "rows_total": len(merged_all),
        "note": (
            "CGD/CGS measured at a single fixed VDS=0 bias per the raw "
            "'C-V [W-L cgd/cgs]' files (paper Fig. 3b/c setup); the combined "
            "'C-V Vd=*' family sweeps VDS but only for the *total* Cg, not "
            "the individual CGD/CGS split, so it is not usable for a "
            "VDS-resolved CGD/CGS model and is excluded here."
        ),
    }
    with open(os.path.join(OUT_DIR, "cv_dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(per_device_frames)} per-device files + merged_cv_dataset.csv "
          f"({len(merged_all)} rows) to {OUT_DIR}")
    print(f"frequency={freq_hz} Hz, AC level={ac_level_v} V, VDS fixed at 0 V")


if __name__ == "__main__":
    main()
