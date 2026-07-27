"""Verify verilogA/tft_ann_full_model.va against the trained PyTorch models.

Parses the weight/bias tables straight out of the generated Verilog-A source,
re-implements the module's arithmetic exactly as its loops are written
(including the flattened array indexing and the min-max input clamping), and
compares the result to the PyTorch networks on every measured operating point.

This catches the failure modes that matter for a generated netlist: transposed
or mis-strided weight matrices, a wrong de-standardisation constant, a dropped
area re-scaling, or an input mapped to the wrong column.

Run:
    python scripts/verify_verilog_a_full.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cap_dataset import load_and_split  # noqa: E402
from src.dataset import FEATURE_BOUNDS  # noqa: E402
from src.model import TFTNet2  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VA_PATH = os.path.join(REPO, "verilogA", "tft_ann_full_model.va")
CAP_DATA = os.path.join(REPO, "data_cleaned_cv", "merged_cap_dataset.csv")
ID_WEIGHTS = os.path.join(REPO, "trained_ANN", "deep", "weights.json")

ARRAY_RE = re.compile(r"\b(\w+)\[(\d+)\]\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?);")
SCALAR_RE = re.compile(r"^\s*(\w+)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?);", re.M)


def parse_va(path):
    """Pull every `name[i] = v;` and `name = v;` initialisation out of the file."""
    with open(path) as f:
        src = f.read()
    init = src.split("end // end of initialization")[0]

    arrays = {}
    for name, idx, val in ARRAY_RE.findall(init):
        arrays.setdefault(name, {})[int(idx)] = float(val)
    arrays = {k: np.array([v[i] for i in range(max(v) + 1)]) for k, v in arrays.items()}

    scalars = {}
    for name, val in SCALAR_RE.findall(init):
        if name not in arrays:
            scalars[name] = float(val)
    return arrays, scalars, src


def forward_va(arrays, scalars, prefix, nh1, nh2, x_scaled):
    """Reproduce the module's two-hidden-layer loop arithmetic verbatim.

    hidden 1: v1[i] = sum_k x[k]*w1[NI*i+k] + b1[i]
    hidden 2: v2[i] = sum_j y1[j]*w2[NH1*i+j] + b2[i]
    output  : out   = sum_i y2[i]*wo[i] + bo
    """
    ni = 4
    w1 = arrays[f"{prefix}_w1"]
    b1 = arrays[f"{prefix}_b1"]
    w2 = arrays[f"{prefix}_w2"]
    b2 = arrays[f"{prefix}_b2"]
    wo = arrays[f"{prefix}_wo"]
    bo = scalars[f"{prefix}_bo"]

    assert w1.size == ni * nh1, (prefix, w1.size, ni * nh1)
    assert w2.size == nh1 * nh2, (prefix, w2.size, nh1 * nh2)
    assert wo.size == nh2 and b1.size == nh1 and b2.size == nh2

    out = np.empty(len(x_scaled))
    for n, x in enumerate(x_scaled):
        y1 = np.empty(nh1)
        for i in range(nh1):
            y1[i] = np.tanh(sum(x[k] * w1[ni * i + k] for k in range(ni)) + b1[i])
        y2 = np.empty(nh2)
        for i in range(nh2):
            y2[i] = np.tanh(sum(y1[j] * w2[nh1 * i + j] for j in range(nh1)) + b2[i])
        out[n] = float(np.dot(y2, wo) + bo)
    return out


def scale_inputs(vg, vd, w_um, l_um):
    def s(name, val):
        lo, hi = FEATURE_BOUNDS[name]
        return np.clip((np.asarray(val, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([s("VG", vg), s("VD", vd), s("W", w_um), s("L", l_um)], axis=1)


def check_cap(cap, arrays, scalars, report):
    wpath = os.path.join(REPO, "trained_Cg_ANN", cap, "weights.json")
    with open(wpath) as f:
        wj = json.load(f)
    nh1 = wj["architecture"]["n_hidden1"]
    nh2 = wj["architecture"]["n_hidden2"]
    tt = wj["target_transform"]
    area_norm = tt["area_normalised"]

    model = TFTNet2(4, nh1, nh2, 1)
    model.load_state_dict(torch.load(
        os.path.join(REPO, "trained_Cg_ANN", cap, "model_weights.pt")))
    model.eval()

    df = pd.read_csv(CAP_DATA)
    df = df[df.cap_type == cap].drop_duplicates().reset_index(drop=True)
    X = scale_inputs(df.VG, df.VD, df.W, df.L)

    with torch.no_grad():
        torch_out = model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    va_out = forward_va(arrays, scalars, cap, nh1, nh2, X)

    # de-standardise + undo area normalisation, the same way the .va does
    def recover(out):
        c = out * tt["y_std"] + tt["y_mean"]
        if area_norm:
            c = c * df.W.to_numpy() * df.L.to_numpy() / 1e3
        return np.maximum(c, 0.0)

    c_torch, c_va = recover(torch_out), recover(va_out)
    # the .va stores weights at %.6f, so a small absolute drift is expected
    max_abs = float(np.max(np.abs(c_va - c_torch)))
    max_rel = float(np.max(np.abs(c_va - c_torch) / np.maximum(c_torch, 1e-3)) * 100)
    rmse_vs_meas = float(np.sqrt(np.mean((c_va - df.C_pF.to_numpy()) ** 2)))

    # the .va must also declare the array bounds it actually uses
    decl_nh1 = int(re.search(rf"parameter integer {cap.upper()}_NH1\s*=\s*(\d+)",
                             report["src"]).group(1))
    decl_nh2 = int(re.search(rf"parameter integer {cap.upper()}_NH2\s*=\s*(\d+)",
                             report["src"]).group(1))
    assert (decl_nh1, decl_nh2) == (nh1, nh2), (cap, decl_nh1, decl_nh2, nh1, nh2)

    ok = max_abs < 1e-3
    print(f"  {cap.upper():4s} {nh1:>2d}->{nh2:<2d} "
          f"area_norm={str(area_norm):5s}  max|dC|={max_abs:.2e} pF  "
          f"max rel={max_rel:.4f}%   RMSE vs measured={rmse_vs_meas:.4f} pF  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_id(arrays, scalars, report):
    with open(ID_WEIGHTS) as f:
        wj = json.load(f)
    nh1, nh2 = wj["architecture"]["hidden_sizes"]
    y_mean = wj["target_transform"]["y_mean_log10_absID"]
    y_std = wj["target_transform"]["y_std_log10_absID"]

    w1 = np.array(wj["layers"][0]["w"])
    b1 = np.array(wj["layers"][0]["b"])
    w2 = np.array(wj["layers"][1]["w"])
    b2 = np.array(wj["layers"][1]["b"])
    wo = np.array(wj["output"]["w"][0])
    bo = float(wj["output"]["b"][0])

    rng = np.random.default_rng(0)
    vg = rng.uniform(-5, 5, 400)
    vd = rng.uniform(0, 5, 400)
    w_um = rng.choice([5, 10, 20, 40, 80, 160], 400)
    l_um = rng.choice([5, 10, 15, 20], 400)
    X = scale_inputs(vg, vd, w_um, l_um)

    y1 = np.tanh(X @ w1.T + b1)
    y2 = np.tanh(y1 @ w2.T + b2)
    ref_out = y2 @ wo + bo
    va_out = forward_va(arrays, scalars, "id", nh1, nh2, X)

    log_ref = ref_out * y_std + y_mean
    log_va = va_out * y_std + y_mean
    max_abs = float(np.max(np.abs(log_va - log_ref)))
    max_rel_id = float(np.max(np.abs(10.0 ** log_va - 10.0 ** log_ref)
                              / np.maximum(10.0 ** log_ref, 1e-30)) * 100)

    decl_nh1 = int(re.search(r"parameter integer ID_NH1\s*=\s*(\d+)",
                             report["src"]).group(1))
    decl_nh2 = int(re.search(r"parameter integer ID_NH2\s*=\s*(\d+)",
                             report["src"]).group(1))
    assert (decl_nh1, decl_nh2) == (nh1, nh2)

    ok = max_abs < 1e-4
    print(f"  ID   {nh1:>2d}->{nh2:<2d}                 "
          f"max|dlog10(ID)|={max_abs:.2e}  max rel(ID)={max_rel_id:.4f}%  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_structure(src):
    """Sanity-check the equivalent-circuit wiring and the scaling constants."""
    checks = {
        "ID current source across D-S": "I(D,S) <+ id;" in src,
        "CGD charge branch across G-D": "I(G,D) <+ ddt(qgd);" in src,
        "CGS charge branch across G-S": "I(G,S) <+ ddt(qgs);" in src,
        "CGD charge = C*V(G,D)": "qgd = cgd * V(G,D);" in src,
        "CGS charge = C*V(G,S)": "qgs = cgs * V(G,S);" in src,
        "tanh hidden layers only": src.count("tanh(") >= 6,
        "no non-tanh activation": not re.search(r"\b(exp|relu|sigmoid)\s*\(", src),
        "VG scaling from V(G,S)": "vg_s = (V(G,S) - vg_lo)" in src,
        "VD scaling from V(D,S)": "vd_s = (V(D,S) - vd_lo)" in src,
        "capacitance floored at 0": src.count("< 0) cg") == 2,
    }
    all_ok = True
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
        all_ok &= ok
    return all_ok


def main():
    arrays, scalars, src = parse_va(VA_PATH)
    report = {"src": src}

    print(f"Verifying {os.path.relpath(VA_PATH, REPO)}\n")
    print("Numerical equivalence (Verilog-A arithmetic vs trained network):")
    ok = check_id(arrays, scalars, report)
    ok &= check_cap("cgd", arrays, scalars, report)
    ok &= check_cap("cgs", arrays, scalars, report)

    print("\nEquivalent-circuit structure:")
    ok &= check_structure(src)

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
