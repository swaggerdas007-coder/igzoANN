"""Cross-check src/tran.py against ngspice-42 on a NAND2 (+ FO1 load), and
show that the .va's own ddt(C*V) capacitor form cannot be simulated.

ngspice runs the behavioural transliteration of the .va (src/ngspice_tft.py)
with its own Gear integrator and LTE timestep control -- an independent
solver on the same equations. It is ~1000x slower than src/tran.py, hence
one small gate.

    python scripts/logic_ngspice_check.py   # -> outputs/logic_test/ngspice_check.{png,json}
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logic_circuits import OUT, C, style        # noqa: E402
from src.logic import Netlist, run_ngspice     # noqa: E402
from src.logic_meas import crossings            # noqa: E402

VDD, F = 5.0, 50e3
TSTOP = 30e-6


def bench():
    n = Netlist()
    n.vdc("vdd", VDD)
    n.vpulse("a", 0, VDD, 1 / F, delay=5e-6)      # rises at 5 us, falls at 15 us
    n.vdc("b", VDD)
    n.nand("dut", ["a", "b"], "y")
    n.nand("l0", ["y", "vdd"], "z0")
    return n


def main():
    style()
    res = {}
    n = bench()
    t0 = time.time()
    rp = n.simulate(TSTOP, nodeset={"y": VDD, "dut_x": VDD}, h_max=50e-9)
    res["python_cpu_s"] = time.time() - t0

    cols = ["y", "x", "z"]
    deck = n.ngspice("NAND2 cross-check", TSTOP, 10e-9, ["v(y)", "v(dut_x)", "v(z0)"],
                     nodeset={"y": VDD, "dut_x": VDD})
    t0 = time.time()
    rs = run_ngspice(deck, cols, workdir=os.path.join("/tmp", "ngs_check_cdv"))
    res["ngspice_cpu_s"] = time.time() - t0

    t = rs["time"]
    for name, k in (("y", "y"), ("x", "dut_x"), ("z", "z0")):
        d = np.abs(np.interp(t, rp["time"], rp[k]) - rs[name])
        res[f"max_abs_diff_{name}_V"] = float(d.max())
        res[f"rms_diff_{name}_V"] = float(np.sqrt(np.mean(d ** 2)))
    for lbl, d in (("fall", -1), ("rise", +1)):
        tp = crossings(rp["time"], rp["y"], VDD / 2, d)
        ts = crossings(t, rs["y"], VDD / 2, d)
        res[f"y_{lbl}_cross_python_us"] = float(tp[0] * 1e6)
        res[f"y_{lbl}_cross_ngspice_us"] = float(ts[0] * 1e6)

    # the .va as written: I = ddt(C(V)*V)
    deck = n.ngspice("NAND2, ddt(C*V) form", TSTOP, 10e-9, ["v(y)", "v(dut_x)", "v(z0)"],
                     nodeset={"y": VDD, "dut_x": VDD}, form="qcv")
    try:
        rq = run_ngspice(deck, cols, workdir=os.path.join("/tmp", "ngs_check_qcv"), timeout=3600)
        tend = rq["time"][-1]
        res["qcv_ngspice"] = (f"completed to t = {tend * 1e6:.2f} us" if tend >= 0.999 * TSTOP else
                              f"transient aborted at t = {tend * 1e6:.2f} us of {TSTOP * 1e6:g} us")
    except Exception as e:                       # noqa: BLE001
        msg = [l for l in str(e).splitlines() if "too small" in l or "abort" in l or "singular" in l]
        res["qcv_ngspice"] = "failed: " + " | ".join(dict.fromkeys(msg))[:300]
    try:
        bench().simulate(TSTOP, nodeset={"y": VDD, "dut_x": VDD}, form="qcv")
        res["qcv_python"] = "completed"
    except Exception as e:                       # noqa: BLE001
        res["qcv_python"] = f"failed: {e}"

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, (k, kp, lbl) in zip(axs, (("y", "y", "output Y"), ("x", "dut_x", "internal node X"))):
        ax.plot(rp["time"] * 1e6, rp[kp], color=C["blue"], lw=2.4, label="src/tran.py")
        ax.plot(t * 1e6, rs[k], color=C["orange"], lw=1.2, ls="--", label="ngspice-42")
        ax.set_ylabel("V")
        ax.set_title(f"NAND2 {lbl}: max |diff| {res[f'max_abs_diff_{k}_V'] * 1e3:.0f} mV, "
                     f"rms {res[f'rms_diff_{k}_V'] * 1e3:.1f} mV")
        ax.legend(loc="center right")
    axs[1].set_xlabel("time (us)   [A: 0->5 V at 5 us, 5->0 V at 15 us; B = 5 V]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ngspice_check.png"), dpi=150)
    with open(os.path.join(OUT, "ngspice_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
