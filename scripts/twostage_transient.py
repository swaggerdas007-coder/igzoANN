"""10 kHz sine through the 40 dB two-stage a-IGZO amplifier.

Design from twostage_L5.csv: both pairs 160/5 um, 40.2 dB, f-3dB 17.9 kHz.
40 dB is 100x, so the input has to be small -- 10 mVpp in gives ~1 Vpp out,
which is all the output swing there is.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.circuit import transient  # noqa: E402
from scripts.twostage import build, solve, O1A, O2A, TA, O1B, O2B, TB, IN1, IN2, VDD  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "va_test")
W, L, VCM, VBA, VBB, DA, DB = 160e-6, 5e-6, 0.8, 2.0, 2.0, 3.0, 2.0
FREQ, VPP = 10e3, 0.010


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = solve(W, L, W, L, VCM, VBA, VBB, DA, DB)
    c, v = r["c"], r["v"]
    print(f"R_A = {r['ra']/1e6:.2f} MOhm, R_B = {r['rb']/1e6:.2f} MOhm")
    print(f"Av = {r['av']:.0f} ({r['av_db']:.1f} dB), f-3dB = {r['f3db']/1e3:.1f} kHz, "
          f"P = {r['power']*1e6:.1f} uW")
    print(f"quiescent: stage A out {r['vo_a']:.3f} V, stage B out {r['vo_b']:.3f} V")

    npts = 400
    t = np.linspace(0, 3 / FREQ, 3 * npts + 1)
    vin = VPP / 2 * np.sin(2 * np.pi * FREQ * t)

    def drive(time):
        s = VPP / 2 * np.sin(2 * np.pi * FREQ * time)
        c.fixed[IN1] = VCM + s / 2
        c.fixed[IN2] = VCM - s / 2

    rec = transient(c, v, t, drive)
    vo = rec[:, O1B] - rec[:, O2B]
    va = rec[:, O1A] - rec[:, O2A]

    last = t >= 2 / FREQ
    swing = vo[last].max() - vo[last].min()
    gain = swing / (vin[last].max() - vin[last].min())
    sp = np.abs(np.fft.rfft(vo[last][:-1] - vo[last][:-1].mean()))
    thd = np.sqrt((sp[2:10] ** 2).sum()) / sp[1]
    print(f"\ninput  {VPP*1e3:.0f} mVpp @ {FREQ/1e3:.0f} kHz")
    print(f"stage A out {va[last].max()-va[last].min():.4f} Vpp")
    print(f"stage B out {swing:.4f} Vpp  ->  total gain {gain:.1f} "
          f"({20*np.log10(gain):.1f} dB)")
    print(f"DC gain {r['av_db']:.1f} dB; at 10 kHz vs the {r['f3db']/1e3:.1f} kHz "
          f"corner, {gain/r['av']*100:.0f}% survives")
    print(f"THD = {thd*100:.2f} %")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t * 1e6, vin * 1e3, "k")
    ax[0].set_ylabel("Vin diff [mV]")
    ax[0].set_title(f"{VPP*1e3:.0f} mVpp, 10 kHz through the two-stage "
                    f"({r['av_db']:.1f} dB, 160/5 um pairs)")
    ax[1].plot(t * 1e6, va, "C0")
    ax[1].set_ylabel("stage A out diff [V]")
    ax[2].plot(t * 1e6, vo, "C3")
    ax[2].set_ylabel("stage B out diff [V]")
    ax[2].set_xlabel("time [us]")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twostage_transient.png"), dpi=130)
    print(f"wrote {OUT}/twostage_transient.png")


if __name__ == "__main__":
    main()
