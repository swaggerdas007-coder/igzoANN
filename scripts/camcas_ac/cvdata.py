"""Loader for the raw B1500 C-V exports in data/ (AC analysis of the
CAMCAS model). Every number is returned in SI units exactly as exported:
C in F (Cp of the Cp-G model), G in S, voltages in V, frequency in Hz."""
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "outputs_camcas_ac")
PLOTS = os.path.join(OUT, "plots")
GEOMS = [(20, 20), (40, 20), (160, 15), (160, 20)]


def read_export(path):
    meta, names, rows = {}, None, []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        p = [x.strip() for x in line.rstrip("\n").split(",")]
        if len(p) < 2:
            continue
        if p[0] in ("TestParameter", "MetaData"):
            meta[p[1]] = p[2:]
        elif p[0] == "DataName":
            names = p[1:]
        elif p[0] == "DataValue":
            rows.append([float(x) if x not in ("", "nan") else float("nan") for x in p[1:]])
    return meta, pd.DataFrame(rows, columns=names)


def load_all():
    """Tidy table of every C-V point: family, kind (cg/cgd/cgs), W, L,
    VG, VD (SMU drain bias, NaN when no SMU), f, C, G."""
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA, "C-V*.csv"))):
        name = os.path.basename(path)
        m = re.match(r"C-V (Vd=([\d.]+) |fc )?\[(\d+)-(\d+) (cgd|cgs|cg)\(", name)
        fam = "vd" if m.group(1) and m.group(1).startswith("Vd") else (
            "fine" if m.group(1) else "basic")
        meta, d = read_export(path)
        vg = d["VG"] if "VG" in d else d["V"]
        frames.append(pd.DataFrame(dict(
            family=fam, kind=m.group(5), W=int(m.group(3)), L=int(m.group(4)),
            VG=vg.round(4), VD=d["VD"] if "VD" in d else float("nan"),
            f=d["Freq"], C=d["C"], G=d["G"], file=name,
            time=meta.get("TestRecord.RecordTime", [""])[0])))
    return pd.concat(frames, ignore_index=True)


def load_cf():
    meta, d = read_export(glob.glob(os.path.join(DATA, "Generic C-f*.csv"))[0])
    return meta, d
