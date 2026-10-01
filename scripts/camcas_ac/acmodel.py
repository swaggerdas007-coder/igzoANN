"""Candidate AC/charge models and the baseline (existing Verilog-A) model.

All capacitances in F, voltages in V, W and L in um (as in the Verilog-A),
areas converted explicitly: W*L [um^2] * 1e-12 = [m^2].
"""
import numpy as np

UM2 = 1e-12  # m^2 per um^2

# --- baseline: Appendix F charge block exactly as coded in tft_camcas_thesis.va
EPS_OX = 3.45e-11   # F/m  (SiO2: 3.9 * 8.854e-12)
TOX = 0.1e-6        # m


def baseline_as_coded(W, L):
    """Covs, Covd, CCH exactly as Listing F.1 computes them (W, L in um)."""
    Lov = L * 10e-5                      # = L * 1e-4
    Lovch = 0.8 * L * 1e-6
    Cov = EPS_OX * W * 1e-5 * Lov / TOX
    Cch = EPS_OX * W * 1e-5 * Lovch / TOX
    return dict(Covs=Cov, Covd=Cov, Cch=Cch)


def baseline_intended(W, L):
    """What thesis Sec. 4.2 / Table 4.1 states: L_OV = L/10 per side,
    L_CH = 0.8 L, W and L converted um -> m (reproduces Table 4.1)."""
    Wm, Lm = W * 1e-6, L * 1e-6
    Cov = EPS_OX * Wm * 0.1 * Lm / TOX
    Cch = EPS_OX * Wm * 0.8 * Lm / TOX
    return dict(Covs=Cov, Covd=Cov, Cch=Cch)


def baseline_terminal_caps(b):
    """Small-signal caps of the baseline charges Qovs=Covs*Vgs,
    Qovd=Covd*Vgd, Qch=CCH*Vgs (all channel charge on the source):
    bias independent."""
    Cgs = b["Covs"] + b["Cch"]
    Cgd = b["Covd"]
    return dict(Cgs=Cgs, Cgd=Cgd, Cgg=Cgs + Cgd)


# --- channel-charge shape at VDS = 0 -----------------------------------------
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def n_shape(v, p, form):
    """Normalized channel capacitance dQch/dV / Cch_max at VDS = 0."""
    x = v - p["V0"]
    if form == "logistic":
        return sigmoid(x / p["s1"])
    return (1 - p["beta"]) * sigmoid(x / p["s1"]) + p["beta"] * sigmoid((x - p["d2"]) / p["s2"])


def veff(v, p, form):
    """Closed-form channel charge per unit Cch_max (V): integral of n_shape,
    zero deep in the off-state. Used as the effective overdrive."""
    x = v - p["V0"]
    if form == "logistic":
        return p["s1"] * softplus(x / p["s1"])
    return ((1 - p["beta"]) * p["s1"] * softplus(x / p["s1"])
            + p["beta"] * p["s2"] * softplus((x - p["d2"]) / p["s2"]))


# --- proposed charge model ----------------------------------------------------
def charges(vgs, vds, W, L, P, form="mix"):
    """Terminal charges (C) of the proposed model.
      Qov,s = C'ov,s*W*L*Vgs, Qov,d = C'ov,d*W*L*Vgd
      channel: Ward-Dutton long-channel charge with smooth effective
      overdrives Vs = veff(Vgs), Vd = veff(Vgd) (symmetric in S/D):
        Qch = (2/3) Cch (Vs^2 + Vs Vd + Vd^2)/(Vs + Vd)
        Qd  = -Cch (6Vd^3 + 12Vd^2 Vs + 8Vd Vs^2 + 4Vs^3)/(15 (Vs+Vd)^2)
        Qs  = -Qch - Qd   (channel part)
      Qg = Qch + Qov,s + Qov,d, Qd_tot = Qd - Qov,d, Qs_tot = Qs - Qov,s,
      Qg + Qd_tot + Qs_tot = 0 by construction."""
    A = W * L * UM2
    cch, covs, covd = P["Cch_area"] * A, P["Cov_s_area"] * A, P["Cov_d_area"] * A
    vgd = vgs - vds
    vs = veff(vgs, P, form) + 1e-12
    vd = veff(vgd, P, form) + 1e-12
    S = vs + vd
    qch = (2.0 / 3.0) * cch * (vs ** 2 + vs * vd + vd ** 2) / S
    qdc = -cch * (6 * vd ** 3 + 12 * vd ** 2 * vs + 8 * vd * vs ** 2 + 4 * vs ** 3) / (15 * S ** 2)
    qsc = -qch - qdc
    qg = qch + covs * vgs + covd * vgd
    qd = qdc - covd * vgd
    qs = qsc - covs * vgs
    return qg, qd, qs


def terminal_caps(vg, vd, vs, W, L, P, form="mix", h=1e-4):
    """C_ij = dQ_i/dV_j (numerical, central differences) for i, j in g, d, s."""
    names = ("g", "d", "s")
    v0 = dict(g=vg, d=vd, s=vs)
    C = {}
    for j in names:
        up, dn = dict(v0), dict(v0)
        up[j] = v0[j] + h
        dn[j] = v0[j] - h
        qu = charges(up["g"] - up["s"], up["d"] - up["s"], W, L, P, form)
        qd = charges(dn["g"] - dn["s"], dn["d"] - dn["s"], W, L, P, form)
        for i, (a, b) in zip(names, zip(qu, qd)):
            C[i + j] = (a - b) / (2 * h)
    return C


def cg_measured_config(vg, W, L, P, form="mix"):
    """What the 'cg' set-up reads (S and D both on CMU-low, VDS = 0): Cgg."""
    return terminal_caps(vg, 0.0, 0.0, W, L, P, form)["gg"]
