"""How much do the results depend on the .va's missing reverse conduction?

ntft_full.va clamps VDS < 0 to 0, so a device whose drain dips below its
source keeps pushing its VDS = 0 current forward instead of conducting
backwards. Re-run the 5 V / 50 kHz NAND2 bench and counter with D/S swapped
for VDS < 0 (src/tran.py symmetric=True) and compare.

    python scripts/logic_sensitivity.py     # -> outputs/logic_test/sensitivity_symmetric.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logic_circuits import (OUT, counter_netlist, counter_measure, nand_measure,   # noqa: E402
                            Netlist)

VDD, F = 5.0, 50e3


def nand2(sym):
    n = Netlist()
    n.vdc("vdd", VDD)
    n.vpulse("a", 0, VDD, 1 / F, delay=0.25 / F)
    n.vpulse("b", 0, VDD, 2 / F, delay=0.25 / F)
    n.nand("dut", ["a", "b"], "y")
    for k in range(3):
        n.nand(f"l{k}", ["y", "vdd"], f"z{k}")
    r = n.simulate(2.5 / F, symmetric=sym)
    r["ins"] = ["a", "b"]
    m = nand_measure(r, VDD, F)
    m["Y_min_V"] = float(r["y"].min())
    return m


def main():
    res = {}
    for sym in (False, True):
        key = "symmetric" if sym else "va_as_written"
        net, ns = counter_netlist(VDD, F)
        r = net.simulate(10 / F, nodeset=ns, symmetric=sym)
        m = counter_measure(r, VDD, F)
        m["Q_min_V"] = float(min(r[f"q{k}"].min() for k in range(3)))
        res[key] = dict(counter=m, nand2=nand2(sym))
        print(key, json.dumps(res[key], indent=1), flush=True)
    with open(os.path.join(OUT, "sensitivity_symmetric.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
