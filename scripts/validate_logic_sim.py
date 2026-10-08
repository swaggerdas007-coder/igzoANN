"""Cross-validate src/logic_sim.py against src/circuit.py and against calculus.

Four checks:
  1. DC agreement with the existing engine on the diode-load differential pair,
     with the physics fixes switched off so the two engines are solving the
     identical set of equations.
  2. Analytic did/dvgs, did/dvds of the *symmetrised* device vs finite
     differences, including the reverse-bias region the raw .va cannot express.
  3. Device symmetry / continuity at VDS = 0.
  4. Transient integrator against the analytic RC step response.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logic_sim import Net, eval_batch, model          # noqa: E402
from src.circuit import Circuit                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "pseudo_cmos")
os.makedirs(OUT, exist_ok=True)
fail = []


def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fail.append(name)


print("1. DC vs src/circuit.py -- diode-load differential pair (fixes OFF)")
W1, L1, W3, L3, WL, LL = 160e-6, 5e-6, 80e-6, 5e-6, 20e-6, 20e-6
VDD, VCM, VB = 3.0, 0.8, 2.0

old = Circuit(7)
O1, O2, TAIL, I1, I2, VD, VBN = 1, 2, 3, 4, 5, 6, 7
old.fixed = {I1: VCM, I2: VCM, VD: VDD, VBN: VB}
old.add_tft("M1", O1, I1, TAIL, W1, L1)
old.add_tft("M2", O2, I2, TAIL, W1, L1)
old.add_tft("M3", TAIL, VBN, 0, W3, L3)
old.add_tft("ML1", VD, VD, O1, WL, LL)
old.add_tft("ML2", VD, VD, O2, WL, LL)
vo, oko = old.solve_dc(guess={O1: VDD * .6, O2: VDD * .6, TAIL: .5})

new = Net(symmetric=False, offset=False)
for nm in ("o1", "o2", "tail", "in1", "in2", "vdd", "vb"):
    new.node(nm)
new.tft("M1", "o1", "in1", "tail", W1, L1)
new.tft("M2", "o2", "in2", "tail", W1, L1)
new.tft("M3", "tail", "vb", "gnd", W3, L3)
new.tft("ML1", "vdd", "vdd", "o1", WL, LL)
new.tft("ML2", "vdd", "vdd", "o2", WL, LL)
for nm, val in (("in1", VCM), ("in2", VCM), ("vdd", VDD), ("vb", VB)):
    new.force(nm, val)
new.build()
g = np.zeros(new.n)
g[new._idx["o1"]] = g[new._idx["o2"]] = VDD * .6
g[new._idx["tail"]] = .5
vn, okn = new.solve_dc(guess=g)

pairs = [("o1", O1), ("o2", O2), ("tail", TAIL)]
err = max(abs(vn[new._idx[a]] - vo[b]) for a, b in pairs)
for a, b in pairs:
    print(f"    {a:5s} old {vo[b]: .6f}  new {vn[new._idx[a]]: .6f}")
chk("both converged", oko and okn)
chk("node voltages agree", err < 1e-6, f"max |dV| = {err:.2e} V")

print("\n2. symmetrised derivatives vs finite differences")
m = model()
rng = np.random.default_rng(0)
vgs = rng.uniform(-3, 5, 4000)
vds = rng.uniform(-4, 5, 4000)
w = rng.choice([5e-6, 20e-6, 160e-6], 4000)
l = rng.choice([5e-6, 20e-6], 4000)
h = 1e-6
d0 = eval_batch(vgs, vds, w, l)
fg = (eval_batch(vgs + h, vds, w, l)["id"] - eval_batch(vgs - h, vds, w, l)["id"]) / (2 * h)
fd = (eval_batch(vgs, vds + h, w, l)["id"] - eval_batch(vgs, vds - h, w, l)["id"]) / (2 * h)
sc = np.maximum(np.abs(d0["id"]) / 0.05, 1e-10)       # currents span 8 decades
eg = np.max(np.abs(fg - d0["didvg"]) / sc)
ed = np.max(np.abs(fd - d0["didvd"]) / sc)
chk("did/dvgs matches FD", eg < 2e-2, f"max rel err {eg:.2e}")
chk("did/dvds matches FD", ed < 2e-2, f"max rel err {ed:.2e}")
rv = d0["rev"]
chk("reverse region exercised", rv.mean() > 0.3, f"{rv.mean()*100:.0f}% of samples")

print("\n3. device symmetry and continuity at VDS = 0")
vg = np.linspace(-2, 5, 29)
z = np.zeros_like(vg)
i0 = eval_batch(vg, z, 160e-6, 5e-6)["id"]
chk("ID(VDS=0) == 0", np.max(np.abs(i0)) < 1e-18, f"max |ID| = {np.max(np.abs(i0)):.1e} A")
e = 1e-4
ip = eval_batch(vg, z + e, 160e-6, 5e-6)["id"]
im = eval_batch(vg - e, z - e, 160e-6, 5e-6)["id"]   # same vg_eff, mirrored vds
chk("odd in VDS (terminal swap)", np.max(np.abs(ip + im) / np.maximum(np.abs(ip), 1e-16)) < 1e-9,
    f"max asym {np.max(np.abs(ip + im) / np.maximum(np.abs(ip), 1e-16)):.1e}")
cd = eval_batch(vg, z + 1e-9, 160e-6, 5e-6)
cs = eval_batch(vg, z - 1e-9, 160e-6, 5e-6)
jump = np.max(np.abs(cd["cgd"] - cs["cgd"]) / cd["cgd"])
chk("caps continuous through VDS=0", jump < 1e-5, f"max jump {jump*100:.2e} %")

print("\n4. transient integrator vs analytic RC")
rc = Net()
rc.node("a")
rc.node("vin")
rc.res("vin", "a", 1e6)
rc.cap("a", "gnd", 10e-12)
rc.force("vin", 1.0)
rc.build()
v0 = np.zeros(rc.n)
v0[rc._idx["vin"]] = 1.0
t, V = rc.transient(v0, 5e-5, dvmax=0.002, dt_max=2e-7)
ana = 1 - np.exp(-t / (1e6 * 10e-12))
err = np.max(np.abs(V[:, rc._idx["a"]] - ana))
chk("RC step response", err < 2e-3, f"max |err| = {err:.2e} V ({rc.stats['steps']} steps)")

print("\n" + ("ALL CHECKS PASSED" if not fail else f"FAILED: {fail}"))
sys.exit(1 if fail else 0)
