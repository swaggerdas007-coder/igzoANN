"""Transient: 100 mVpp, 10 kHz sine into the recommended a-IGZO diff pair.

Backward Euler, Newton per step, using the model's own Q = C(V)*V charges --
the same (non-charge-conserving) form the .va writes as ddt(C*V).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diffpair import build, solve_rl, OUT1, OUT2, TAIL, IN1, IN2  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")

VDD, CLOAD = 5.0, 1e-12
W1, L1, VCM, VBIAS, VDROP = 160e-6, 15e-6, 1.0, 1.2, 2.5
FREQ, VPP = 10e3, 0.100


def charges(c, v):
    """Total charge at each node from the TFT caps and the explicit load caps."""
    q = np.zeros(c.n + 1)
    for t in c.tfts:
        o = t.op(v)
        for (a, b, cap) in ((t.g, t.d, o["cgd"]), (t.g, t.s, o["cgs"])):
            qq = cap * (v[a] - v[b])
            q[a] += qq
            q[b] -= qq
    for a, b, cap in c.caps:
        qq = cap * (v[a] - v[b])
        q[a] += qq
        q[b] -= qq
    return q


def cap_jac(c, v, idx, pos):
    J = np.zeros((len(idx), len(idx)))

    def stamp(node, wrt, val):
        if node in pos and wrt in pos:
            J[pos[node], pos[wrt]] += val

    for t in c.tfts:
        o = t.op(v)
        for (a, b, cap) in ((t.g, t.d, o["cgd"]), (t.g, t.s, o["cgs"])):
            for node, sgn in ((a, +1), (b, -1)):
                stamp(node, a, sgn * cap)
                stamp(node, b, -sgn * cap)
    for a, b, cap in c.caps:
        for node, sgn in ((a, +1), (b, -1)):
            stamp(node, a, sgn * cap)
            stamp(node, b, -sgn * cap)
    return J


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, rl = solve_rl(W1, L1, 40e-6, 20e-6, VDD, VCM, VBIAS, VDROP, CLOAD)
    v, _ = c.solve_dc(guess={OUT1: VDD - VDROP, OUT2: VDD - VDROP, TAIL: 0.5})
    print(f"R_L = {rl/1e3:.0f} kOhm, quiescent Vout = {v[OUT1]:.3f} V")

    idx = c._unknowns()
    pos = {k: i for i, k in enumerate(idx)}
    npts = 400
    t = np.linspace(0, 3 / FREQ, 3 * npts + 1)
    dt = t[1] - t[0]
    vin = VPP / 2 * np.sin(2 * np.pi * FREQ * t)     # differential, +/- VPP/2

    x = np.array([v[k] for k in idx])
    q_prev = charges(c, v)[idx]
    rec = np.zeros((len(t), 3))
    rec[0] = [v[OUT1], v[OUT2], v[OUT1] - v[OUT2]]

    for n in range(1, len(t)):
        c.fixed[IN1] = VCM + vin[n] / 2
        c.fixed[IN2] = VCM - vin[n] / 2
        for _ in range(60):
            vv = c._full(x)
            f = c.residual(vv)[idx] + (charges(c, vv)[idx] - q_prev) / dt
            if np.max(np.abs(f)) < 1e-13:
                break
            J = c.jacobian(vv) + cap_jac(c, vv, idx, pos) / dt
            x = x + np.clip(np.linalg.solve(J, -f), -0.2, 0.2)
        vv = c._full(x)
        q_prev = charges(c, vv)[idx]
        rec[n] = [vv[OUT1], vv[OUT2], vv[OUT1] - vv[OUT2]]

    # measure on the last cycle, once the start-up transient has died
    last = t >= 2 / FREQ
    vo = rec[last, 2]
    vi = vin[last]
    gain = (vo.max() - vo.min()) / (vi.max() - vi.min())
    # THD from the fundamental vs harmonics of the last cycle
    sp = np.fft.rfft(vo[:-1] - vo[:-1].mean())
    h = np.abs(sp)
    thd = np.sqrt((h[2:10] ** 2).sum()) / h[1]
    print(f"input  {VPP*1e3:.0f} mVpp differential @ {FREQ/1e3:.0f} kHz")
    print(f"output {vo.max()-vo.min():.4f} Vpp differential  ->  gain {gain:.2f} "
          f"({20*np.log10(gain):.1f} dB)")
    print(f"DC gain was 12.50 (21.9 dB); we are at 10 kHz vs the 15.7 kHz pole, "
          f"so {gain/12.5*100:.0f}% of it survives")
    print(f"THD = {thd*100:.2f} %")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t * 1e6, vin * 1e3, "k")
    ax[0].set_ylabel("Vin differential [mV]")
    ax[0].set_title(f"100 mVpp, 10 kHz into the 160/15 um pair (Av_DC = 12.5)")
    ax[1].plot(t * 1e6, rec[:, 0], label="Vout1")
    ax[1].plot(t * 1e6, rec[:, 1], label="Vout2")
    ax[1].plot(t * 1e6, rec[:, 2] + v[OUT1], "r--", lw=1, label="Vout1-Vout2 (offset)")
    ax[1].set_xlabel("time [us]"); ax[1].set_ylabel("V")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diffpair_transient.png"), dpi=130)
    print(f"wrote {OUT}/diffpair_transient.png")


if __name__ == "__main__":
    main()
