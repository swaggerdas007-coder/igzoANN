"""Full characterisation of the two recommended a-IGZO differential pairs.

Design A ("trusted"): every device sits on a (W,L) the C-V nets were trained
on, so the bandwidth number is backed by measured capacitance.
Design B ("fast"):    L = 5 um input pair -- much faster, but CGD/CGS are
extrapolated there, so treat the bandwidth as indicative only.

Run: python scripts/diffpair_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diffpair import build, analyse, solve_rl, OUT1, OUT2, TAIL, IN1, IN2, VDDN, VB  # noqa: E402
from src.circuit import model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")
os.makedirs(OUT, exist_ok=True)

VDD, CLOAD = 5.0, 1e-12

DESIGNS = {
    "A_trusted_160_15": dict(w1=160e-6, l1=15e-6, vcm=1.0, vb=1.2, vdrop=2.5),
    "B_fast_160_5":     dict(w1=160e-6, l1=5e-6, vcm=2.5, vb=2.0, vdrop=1.0),
}


def dc_transfer(cfg, rl, vid_range):
    """Sweep the differential input, return Vout1-Vout2 -- gives the linear range."""
    vo = []
    for vid in vid_range:
        c = build("res", cfg["w1"], cfg["l1"], 40e-6, 20e-6, VDD,
                  cfg["vcm"], cfg["vb"], rl=rl, cload=CLOAD)
        c.fixed[IN1] = cfg["vcm"] + vid / 2
        c.fixed[IN2] = cfg["vcm"] - vid / 2
        v, ok = c.solve_dc(guess={OUT1: VDD - cfg["vdrop"], OUT2: VDD - cfg["vdrop"], TAIL: 0.5})
        vo.append(v[OUT1] - v[OUT2] if ok else np.nan)
    return np.array(vo)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    summary = []

    for (name, cfg), col in zip(DESIGNS.items(), "rb"):
        c, rl = solve_rl(cfg["w1"], cfg["l1"], 40e-6, 20e-6, VDD,
                         cfg["vcm"], cfg["vb"], cfg["vdrop"], CLOAD)
        r = analyse(c, VDD, CLOAD)
        v, _ = c.solve_dc(guess={OUT1: VDD - cfg["vdrop"], OUT2: VDD - cfg["vdrop"], TAIL: 0.5})

        print(f"\n===== {name}  (W1/L1 = {cfg['w1']*1e6:.0f}/{cfg['l1']*1e6:.0f} um, "
              f"tail M3 = 40/20 um, R_L = {rl/1e3:.0f} kOhm, VDD = {VDD} V) =====")
        print(f"  bias: VCM = {cfg['vcm']} V, V_B(tail gate) = {cfg['vb']} V")
        print(f"  DC op: I_tail = {r['itail']*1e6:.3f} uA, V_out(CM) = {r['vout']:.3f} V, "
              f"V_tail = {r['vtail']:.3f} V, P = {r['power']*1e6:.2f} uW")
        print(f"  M1: gm = {r['gm1']*1e6:.3f} uS, gds = {r['gds1']*1e6:.4f} uS, "
              f"ro = {r['ro1']/1e6:.2f} MOhm, gm/ID = {r['gm_over_id']:.2f} /V, "
              f"gm*ro = {r['av_intrinsic']:.1f}")
        print(f"  Cgs = {r['cgs1']*1e12:.2f} pF, Cgd = {r['cgd1']*1e12:.2f} pF "
              f"(C_load = {CLOAD*1e12:.0f} pF)")
        print(f"  Av = {r['av']:.2f} ({r['av_db']:.1f} dB), f-3dB = {r['f3db']/1e3:.1f} kHz, "
              f"GBW = {r['gbw']/1e3:.0f} kHz")
        print(f"  CMRR = {r['cmrr_db']:.1f} dB, slew = {r['slew_v_per_us']:.2f} V/us, "
              f"Cgd feedthrough zero at {r['f_zero_cgd']/1e3:.0f} kHz")

        f = np.logspace(1, 8, 500)
        vac = c.ac(v, f, {IN1: 0.5, IN2: -0.5, VDDN: 0.0, VB: 0.0})
        add = vac[:, OUT1] - vac[:, OUT2]
        ax[0, 0].semilogx(f, 20 * np.log10(np.abs(add)), col, label=name)
        ax[0, 1].semilogx(f, np.angle(add, deg=True), col, label=name)

        vid = np.linspace(-1.5, 1.5, 61)
        vo = dc_transfer(cfg, rl, vid)
        ax[1, 0].plot(vid, vo, col + "-", label=name)
        g = np.gradient(vo, vid)
        ax[1, 1].plot(vid, g, col + "-", label=name)
        lin = vid[np.abs(g) > 0.9 * np.nanmax(np.abs(g))]
        print(f"  input range for <10% gain compression: "
              f"{lin.min():+.2f} .. {lin.max():+.2f} V differential")

        summary.append(dict(design=name, W1_um=cfg["w1"] * 1e6, L1_um=cfg["l1"] * 1e6,
                            RL_kohm=rl / 1e3, VCM=cfg["vcm"], VB=cfg["vb"],
                            Itail_uA=r["itail"] * 1e6, Vout_V=r["vout"],
                            Av_dB=r["av_db"], f3dB_kHz=r["f3db"] / 1e3,
                            GBW_kHz=r["gbw"] / 1e3, P_uW=r["power"] * 1e6,
                            CMRR_dB=r["cmrr_db"], gm1_uS=r["gm1"] * 1e6,
                            Cgd_pF=r["cgd1"] * 1e12,
                            linear_range_V=lin.max() - lin.min()))

    for a, t, xl, yl in ((ax[0, 0], "differential gain", "f [Hz]", "|Ad| [dB]"),
                         (ax[0, 1], "phase", "f [Hz]", "deg"),
                         (ax[1, 0], "DC transfer", "Vid [V]", "Vout1-Vout2 [V]"),
                         (ax[1, 1], "incremental gain", "Vid [V]", "dVout/dVid")):
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"a-IGZO n-only differential pair, resistive load, VDD={VDD}V, CL={CLOAD*1e12:.0f}pF")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diffpair_report.png"), dpi=130)

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "diffpair_recommended.csv"), index=False)
    print(f"\nwrote {OUT}/diffpair_report.png and diffpair_recommended.csv")


if __name__ == "__main__":
    main()
