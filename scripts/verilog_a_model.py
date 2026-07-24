"""Parses verilogA/tft_ann_static.va and evaluates it in Python.

This is a direct transcription of the module's `analog begin` block: same
weight/bias tables (read straight out of the .va source, not retyped by
hand), same [0,1] input clamp-scaling, same tanh hidden layer, same linear
output layer, same log10/standardization inverse for ID, and the same
analytic gm/gds chain-rule formulas the Verilog-A computes for the
small-signal operating point. Given identical (VG, VD, W, L) inputs this
reproduces bit-for-bit what a Spectre/ngspice-osdi `dc` analysis against
the .va module would return, since the model is purely algebraic (no
reactive elements, no state).
"""
import re

import numpy as np

ARRAY_ASSIGN_RE = re.compile(r"(\w+)\[(\d+)\]\s*=\s*(-?[\d.eE+-]+)\s*;")
SCALAR_RE = re.compile(r"\b({name})\s*=\s*(-?[\d.eE+-]+)\s*;")


def _extract_int_param(text, name):
    m = re.search(rf"parameter integer {name}\s*=\s*(\d+)\s*;", text)
    return int(m.group(1))


def _extract_scalar(text, name):
    m = re.search(SCALAR_RE.pattern.format(name=name), text)
    return float(m.group(2))


def parse_verilog_a(path):
    """Reads the .va source and returns a dict of numpy arrays/scalars
    that fully parameterize the analog block's forward pass."""
    with open(path) as f:
        text = f.read()

    ni = _extract_int_param(text, "NI")
    nnhl = _extract_int_param(text, "NNHL")
    no = _extract_int_param(text, "NO")

    hlayer_w = np.zeros(ni * nnhl)
    hlayer_b = np.zeros(nnhl)
    olayer_w = np.zeros(no * nnhl)
    olayer_b = np.zeros(no)

    for m in ARRAY_ASSIGN_RE.finditer(text):
        name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        if name == "hlayer_w":
            hlayer_w[idx] = val
        elif name == "hlayer_b":
            hlayer_b[idx] = val
        elif name == "olayer_w":
            olayer_w[idx] = val
        elif name == "olayer_b":
            olayer_b[idx] = val

    params = dict(
        ni=ni,
        nnhl=nnhl,
        no=no,
        hlayer_w=hlayer_w.reshape(nnhl, ni),   # [neuron, input]
        hlayer_b=hlayer_b,
        olayer_w=olayer_w,                     # [neuron] (NO == 1)
        olayer_b=olayer_b[0],
        vg_lo=_extract_scalar(text, "vg_lo"), vg_hi=_extract_scalar(text, "vg_hi"),
        vd_lo=_extract_scalar(text, "vd_lo"), vd_hi=_extract_scalar(text, "vd_hi"),
        w_lo=_extract_scalar(text, "w_lo"), w_hi=_extract_scalar(text, "w_hi"),
        l_lo=_extract_scalar(text, "l_lo"), l_hi=_extract_scalar(text, "l_hi"),
        y_mean=_extract_scalar(text, "y_mean"), y_std=_extract_scalar(text, "y_std"),
    )
    return params


class TftAnnModel:
    """Vectorized numpy re-implementation of the `analog begin` block."""

    def __init__(self, va_path):
        self.p = parse_verilog_a(va_path)

    def evaluate(self, vg, vd, w_um, l_um):
        """vg, vd in volts; w_um, l_um in micrometers (converted to SI to
        match the .va `w`/`l` parameters, which are declared in meters).
        Broadcasts like numpy; returns (id, gm, gds) arrays."""
        p = self.p
        vg, vd, w_um, l_um = np.broadcast_arrays(
            np.asarray(vg, dtype=float), np.asarray(vd, dtype=float),
            np.asarray(w_um, dtype=float), np.asarray(l_um, dtype=float),
        )
        w = w_um * 1e-6
        l = l_um * 1e-6

        vg_s = np.clip((vg - p["vg_lo"]) / (p["vg_hi"] - p["vg_lo"]), 0.0, 1.0)
        vd_s = np.clip((vd - p["vd_lo"]) / (p["vd_hi"] - p["vd_lo"]), 0.0, 1.0)
        w_s = np.clip((w - p["w_lo"]) / (p["w_hi"] - p["w_lo"]), 0.0, 1.0)
        l_s = np.clip((l - p["l_lo"]) / (p["l_hi"] - p["l_lo"]), 0.0, 1.0)

        x = np.stack([vg_s, vd_s, w_s, l_s], axis=-1)          # [..., NI]
        hv = x @ p["hlayer_w"].T + p["hlayer_b"]                # [..., NNHL]
        hy = np.tanh(hv)
        dtanh = 1.0 - hy * hy

        olayer_v = hy @ p["olayer_w"] + p["olayer_b"]           # [...]
        log_id = olayer_v * p["y_std"] + p["y_mean"]
        id_ = 10.0 ** log_id

        temp = np.sum(p["olayer_w"] * p["hlayer_w"][:, 0] * dtanh, axis=-1)
        temp1 = np.sum(p["olayer_w"] * p["hlayer_w"][:, 1] * dtanh, axis=-1)

        ln10 = np.log(10.0)
        gm = id_ * ln10 * p["y_std"] * temp / (p["vg_hi"] - p["vg_lo"])
        gds = id_ * ln10 * p["y_std"] * temp1 / (p["vd_hi"] - p["vd_lo"])
        return id_, gm, gds


if __name__ == "__main__":
    import os
    va_path = os.path.join(os.path.dirname(__file__), "..", "verilogA", "tft_ann_static.va")
    model = TftAnnModel(va_path)
    id_, gm, gds = model.evaluate(3.0, 2.0, 40.0, 10.0)
    print(f"VG=3V VD=2V W=40um L=10um -> ID={id_:.4e} A, gm={gm:.4e} S, gds={gds:.4e} S")
