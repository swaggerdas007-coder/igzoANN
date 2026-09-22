"""Python port of verilogA/tft_ann_full_model.va (a-IGZO n-type TFT ANN model).

The weights are *parsed straight out of the .va source* rather than re-exported
from the training checkpoints, so what is exercised here is exactly what a
Spectre/ADS run would see. The forward pass, the [0,1] min-max input clamping,
the log10 de-standardisation of ID and the analytic gm/gds chain rule all
mirror the Verilog-A line for line.

Ports: ID as a current source D->S, CGD across G-D, CGS across G-S.
"""
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VA = os.path.join(REPO, "verilogA", "ntft_full.va")

_ARR = re.compile(r"^\s*([a-z_0-9]+)\[(\d+)\]\s*=\s*(-?[\d.eE+-]+)\s*;")
_SCA = re.compile(r"^\s*([a-z_0-9]+)\s*=\s*(-?[\d.eE+-]+)\s*;")
_PARAM = re.compile(r"parameter\s+integer\s+([A-Z_0-9]+)\s*=\s*(\d+)")


def _parse(path):
    arrays, scalars, dims = {}, {}, {}
    with open(path) as f:
        for line in f:
            line = line.split("//")[0]
            for m in _PARAM.finditer(line):
                dims[m.group(1)] = int(m.group(2))
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                chunk += ";"
                m = _ARR.match(chunk)
                if m:
                    arrays.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
                    continue
                m = _SCA.match(chunk)
                if m:
                    scalars[m.group(1)] = float(m.group(2))
    return arrays, scalars, dims


def _vec(arrays, name, n):
    d = arrays[name]
    if len(d) != n:
        raise ValueError(f"{name}: parsed {len(d)} entries, expected {n}")
    return np.array([d[i] for i in range(n)])


class TFTModel:
    """One a-IGZO TFT instance; all methods are vectorised over bias arrays."""

    def __init__(self, va_path=DEFAULT_VA):
        a, s, dims = _parse(va_path)
        self.dims = dims
        NI = dims["NI"]
        self.NI = NI
        self.nets = {}
        for tag, h1, h2 in (
            ("id", dims["ID_NH1"], dims["ID_NH2"]),
            ("cgd", dims["CGD_NH1"], dims["CGD_NH2"]),
            ("cgs", dims["CGS_NH1"], dims["CGS_NH2"]),
        ):
            self.nets[tag] = dict(
                W1=_vec(a, f"{tag}_w1", NI * h1).reshape(h1, NI),
                b1=_vec(a, f"{tag}_b1", h1),
                W2=_vec(a, f"{tag}_w2", h1 * h2).reshape(h2, h1),
                b2=_vec(a, f"{tag}_b2", h2),
                wo=_vec(a, f"{tag}_wo", h2),
                bo=s[f"{tag}_bo"],
                mean=s[f"{tag}_y_mean"],
                std=s[f"{tag}_y_std"],
            )
        self.vg_lo, self.vg_hi = s["vg_lo"], s["vg_hi"]
        self.vd_lo, self.vd_hi = s["vd_lo"], s["vd_hi"]
        self.w_lo, self.w_hi = s["w_lo"], s["w_hi"]
        self.l_lo, self.l_hi = s["l_lo"], s["l_hi"]

    # -- shared min-max scaling, clamped to [0,1] exactly as the .va does --
    def _scale(self, vgs, vds, w, l):
        vg_s = np.clip((vgs - self.vg_lo) / (self.vg_hi - self.vg_lo), 0.0, 1.0)
        vd_s = np.clip((vds - self.vd_lo) / (self.vd_hi - self.vd_lo), 0.0, 1.0)
        w_s = np.clip((w - self.w_lo) / (self.w_hi - self.w_lo), 0.0, 1.0)
        l_s = np.clip((l - self.l_lo) / (self.l_hi - self.l_lo), 0.0, 1.0)
        return np.broadcast_arrays(*np.atleast_1d(vg_s, vd_s, w_s, l_s))

    def _forward(self, tag, x):
        n = self.nets[tag]
        y1 = np.tanh(x @ n["W1"].T + n["b1"])
        y2 = np.tanh(y1 @ n["W2"].T + n["b2"])
        out = y2 @ n["wo"] + n["bo"]
        return out * n["std"] + n["mean"], y1, y2

    def op(self, vgs, vds, w=20e-6, l=20e-6):
        """DC operating point: ID [A], gm [S], gds [S], CGD/CGS [F]."""
        vg_s, vd_s, w_s, l_s = self._scale(vgs, vds, w, l)
        x = np.stack([vg_s, vd_s, w_s, l_s], axis=-1)

        log_id, y1, y2 = self._forward("id", x)
        idd = np.power(10.0, log_id)

        n = self.nets["id"]
        # d(out)/d(input k) through both tanh layers
        d1 = (1 - y1 ** 2)[..., :, None] * n["W1"]            # (..., h1, NI)
        d2 = np.einsum("hj,...jk->...hk", n["W2"], d1)
        d2 *= (1 - y2 ** 2)[..., :, None]
        dout = np.einsum("h,...hk->...k", n["wo"], d2)
        k = idd * np.log(10.0) * n["std"]
        gm = k * dout[..., 0] / (self.vg_hi - self.vg_lo)
        gds = k * dout[..., 1] / (self.vd_hi - self.vd_lo)
        # clamped inputs have zero true sensitivity (the .va inherits this)
        gm = np.where((vg_s <= 0) | (vg_s >= 1), 0.0, gm)
        gds = np.where((vd_s <= 0) | (vd_s >= 1), 0.0, gds)

        area = (w * 1e6) * (l * 1e6) * 1e-3 * 1e-12   # .va units chain
        cgd = np.clip(self._forward("cgd", x)[0], 0.0, None) * area
        cgs = np.clip(self._forward("cgs", x)[0], 0.0, None) * area
        return dict(id=idd, gm=gm, gds=gds, cgd=cgd, cgs=cgs,
                    clamped_vg=(vg_s <= 0) | (vg_s >= 1),
                    clamped_vd=(vd_s <= 0) | (vd_s >= 1))

    def id_(self, vgs, vds, w=20e-6, l=20e-6):
        return self.op(vgs, vds, w, l)["id"]
