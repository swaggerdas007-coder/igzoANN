"""Unified CAMCAS drain-current model: ONE fixed parameter set for every
geometry, with (W, L) entering only through the physical scaling the model
equation already carries -- W/L' with L' = L - DeltaL, and a width-normalized
series resistance RSD = Rsw/W. No per-device tables, no (L, W) polynomial
surfaces. Shared by scripts/fit_camcas_unified.py (fitting/validation) and
scripts/export_verilog_a_camcas_unified.py (Verilog-A), which implements the
identical equations.

Per branch (lin / sat), the thesis's conductance (Eq. 2.27)

    G(VGS) = G0 * (W/L') * exp(kappa * x^alpha),   x = VGS - Von

is written in an equivalent reference form, anchored at the top of the
measured gate sweep x_H = VH - Von:

    G(VGS) = GH * (W/L') * exp( s * ((x/x_H)^alpha - 1) / alpha )

i.e. kappa = s / (alpha * x_H^alpha), G0 = GH * exp(-kappa * x_H^alpha).
Same curve family, but GH (sheet conductance at VG=VH) and s (log-log slope
there) are each pinned by the data on their own, whereas G0 and kappa are
nearly perfectly correlated: as alpha -> 0 only G0*e^kappa is determined,
which is exactly the degeneracy that sent G0 to its search bound in ~40% of
the per-device fits (outputs_camcas/extracted_params.csv).

Drain current (thesis Eqs. 2.28-2.35, Appendix F structure):

    I_lin = G_lin * VDS / (1 + RSD * G_lin)
    I_sat = G_sat * x_sat
    IDS   = (I_lin^-m + I_sat^-m)^(-1/m) + Ioff
"""
from dataclasses import dataclass, fields

import numpy as np

VH = 5.0
I_FLOOR = 1e-30


@dataclass
class UnifiedParams:
    Von_lin: float
    Von_sat: float
    lnGH_lin: float    # ln of sheet conductance (S per unit W/L') at VG = VH
    lnGH_sat: float
    s_lin: float       # d lnG / d ln x at x = x_H
    s_sat: float
    alpha_lin: float
    alpha_sat: float
    DeltaL: float      # um
    Rsw: float         # Ohm*um  (RSD = Rsw / W)
    m: float           # harmonic-mean smoothing exponent
    lnIoff: float      # ln A

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    def to_vector(self):
        return np.array([getattr(self, n) for n in self.names()], dtype=float)

    @classmethod
    def from_vector(cls, v):
        return cls(*[float(x) for x in v])

    def thesis_form(self):
        """(Von, alpha, kappa, G0) per branch in the thesis's own notation."""
        out = {}
        for br in ("lin", "sat"):
            von = getattr(self, f"Von_{br}")
            a = getattr(self, f"alpha_{br}")
            s = getattr(self, f"s_{br}")
            xh_a = (VH - von) ** a
            kappa = s / (a * xh_a)
            out[br] = {
                "Von": von, "alpha": a, "kappa": kappa,
                "G0": float(np.exp(getattr(self, f"lnGH_{br}") - kappa * xh_a)),
            }
        return out


def _sheet_conductance(vg, von, lnGH, s, alpha):
    x = vg - von
    xh = VH - von
    on = x > 0
    g = np.zeros_like(vg, dtype=float)
    r = np.log(x[on] / xh)
    # Box-Cox term ((x/xH)^alpha - 1)/alpha, via expm1 so alpha -> 0 stays exact
    bc = np.expm1(alpha * r) / alpha
    g[on] = np.exp(np.clip(lnGH + s * bc, -700.0, 700.0))
    return g, np.where(on, x, 0.0)


def ids(vg, vd, W, L, p: UnifiedParams):
    vg = np.asarray(vg, dtype=float)
    vd = np.broadcast_to(np.asarray(vd, dtype=float), vg.shape)
    W = np.broadcast_to(np.asarray(W, dtype=float), vg.shape)
    L = np.broadcast_to(np.asarray(L, dtype=float), vg.shape)

    Lp = np.maximum(L - p.DeltaL, 0.5)
    wl = W / Lp
    rsd = p.Rsw / W

    g_lin, _ = _sheet_conductance(vg, p.Von_lin, p.lnGH_lin, p.s_lin, p.alpha_lin)
    g_sat, x_sat = _sheet_conductance(vg, p.Von_sat, p.lnGH_sat, p.s_sat, p.alpha_sat)
    G_lin = g_lin * wl
    i_lin = G_lin * vd / (1.0 + rsd * G_lin)
    i_sat = g_sat * wl * x_sat

    i_lin = np.maximum(i_lin, I_FLOOR)
    i_sat = np.maximum(i_sat, I_FLOOR)
    # (a^-m + b^-m)^(-1/m) computed as min(a,b) * (1 + (min/max)^m)^(-1/m)
    lo = np.minimum(i_lin, i_sat)
    hi = np.maximum(i_lin, i_sat)
    i_ch = lo * (1.0 + (lo / hi) ** p.m) ** (-1.0 / p.m)
    return i_ch + np.exp(p.lnIoff)
