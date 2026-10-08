# Pseudo-CMOS bootstrapped NOT gate and ring oscillators in a-IGZO

Everything here is simulated with the repo's own ANN compact model
(`verilogA/ntft_full.va`, read straight out of the `.va` source by
`src/va_model.py`), so the devices are the measured a-IGZO TFTs of
`data_cleaned/`, not textbook transistors.

```
python scripts/validate_logic_sim.py     # engine checks -- run this first
python scripts/crosscheck_engines.py     # logic_sim vs tran (independent engines)
python scripts/pseudo_cmos_screen.py     # topology comparison + 2592-point DC sizing screen
python scripts/pseudo_cmos_best.py       # ring-based sizing + bootstrap-cap sweep (3 phases)
python scripts/pczb_check.py             # why the best-looking DC variant is useless
python scripts/ring_osc5.py              # 5-stage ring: supply / frequency study
python scripts/ring_startup_limit.py     # minimum supply vs N, and startup time
python scripts/ring_osc_stages.py        # how far the ring goes, N = 2 .. 101
```

---

## 0. Results

**The gate.** Pseudo-CMOS with a bootstrap capacitor,
**W1/W2/W3/W4 = 5 / 20 / 160 / 40 um, L = 5 um on every device, Cboot = 1 pF.**

| at VDD = 3 V | |
|---|---|
| DC | VOH 2.901 V, VOL 0.086 V, swing 93.8% of VDD, loop gain at the trip point 2.05, butterfly SNM 0.516 V, static power 7.6 uW |
| 5-stage ring | **194.3 kHz**, swing **93.0% of VDD**, 49.1 uW, 515 ns/stage, 25.3 pJ/transition |
| bootstrap | node X peaks at **3.207 V, i.e. 0.207 V above the supply**; without Cboot it stops 0.317 V *below* it |

**Supply window, 5 stages.** Oscillates **0.8 V to 5.0 V**. Swing >= 80% of VDD
over 1.5-4.75 V, >= 90% over **2.0-3.5 V**. Frequency is **linear in the
supply**, f ~ VDD^1.00, from 66 kHz at 1 V to 307 kHz at 5 V.

**How far it goes.** See sections 7-8.

**What the numbers are worth.** The dominant uncertainty is not the design, the
sizing or the integrator -- it is the capacitance model: +-2x on C moves the
frequency by -48% / +91%. Time-step error is 0.13% and two independent engines
agree to 0.6 ns on a 40 us period.

---

## 1. The device decides the design

Two measured properties drive every choice below.

**These TFTs are normally-on.** Measured turn-on is at **VG ~ -0.25 V**, not at
a positive threshold. At VGS = 0 a W160/L5 device passes 146 nA measured
(ANN: 211 nA) against 148 uA at VGS = 3 V, so "input low" buys a factor of
~1000, not 10^8:

```
W160/L5, VD = 5 V          measured      ANN     (data_cleaned/W160_L5_saturation_clean.csv)
  VG = -0.5 V              3.1e-11   1.7e-11 A
  VG =  0.0 V              1.5e-07   2.1e-07 A   <-- the "off" state of a 0 V input
  VG = +3.0 V              1.5e-04   1.5e-04 A
```

Three consequences:

1. **A negative rail does not help, it hurts.** The textbook pseudo-CMOS drawing
   puts the first-stage driver's source on a negative VSS. Useless here: the
   input low level sits at VSS too, so the driver's VGS is 0 either way. And
   putting *only* M2's source at -1 V makes its VGS = +1 V at input-low -- on
   when it must be off -- which collapses VOH from 2.31 V to 1 mV (measured,
   VDD = 3 V). Everything here is **single supply, all driver sources at 0 V**.
2. **There is no classical low-side noise margin.** The VTC's steepest point is
   at VIN ~ 0, so no VIL exists inside [0, VDD] and the unity-gain NML is zero
   or undefined. That is the technology, not the topology. The number used
   instead is the **butterfly SNM**: two cascaded inverters are the same
   structure as a cross-coupled pair, so the eye between the VTC and its mirror
   is well defined, and it is 0.516 V at VDD = 3 V (17% of VDD).
3. **No unipolar ratioed gate here can have zero static current in both
   states.** VOL >= VSS always, so the next stage's driver can never be driven
   below its own source. Static power is a design variable, not something to
   design away -- and it is ~80% of this ring's total.

**Only L = 5 um is trustworthy for sizing.** The ANN's W scaling is monotonic
only on the L = 5 um row; at L = 10 um it puts W80 *below* W40 (12.3 vs
13.8 uA at VG = VD = 3 V) and at L = 15 um W5, W10 and W20 agree to 2%. L = 5 um
is also the fastest row. So all four devices are L = 5 um and W comes from the
measured grid {5, 10, 20, 40, 80, 160} um. Section 9 revisits the one place this
restriction costs something.

---

## 2. Simulator

`src/circuit.py` evaluates the ANN once per device per stamp -- fine for a
7-node differential pair, hopeless for a 101-stage ring (404 devices, ~10^5 time
steps). `src/logic_sim.py` batches every device into one `TFTModel.op()` call and
assembles the Jacobian through pre-computed flat index arrays, so one Newton
iteration costs a fixed handful of numpy calls at any circuit size.
`TFTModel.op()` was also rewritten to get gm/gds by reverse accumulation (three
small matmuls) instead of the `.va`'s forward Jacobian product: algebraically
identical, 8x faster, still exact against finite differences.

Two physics fixes were needed before unipolar *logic* could be simulated at all:

* **D/S symmetrisation.** The `.va` clamps `V(D,S)` at `vd_lo = 0`, so a
  reverse-biased device returns a meaningless ~1 nA with `gds = 0` exactly. A
  bootstrap node driven above VDD reverse-biases its charging device *on
  purpose*, and every logic node resting at VOL sits on VDS = 0.
* **ID(VDS = 0) = 0.** A log10 output cannot reach zero, so the net sources
  0.3-5 nA into a shorted device. With symmetrisation that becomes a ~10 nA step
  discontinuity exactly where logic nodes come to rest, and Newton stops
  converging. Subtracting ID(VGS, 0) costs 0.01% of the on-state current.

### Checks

`scripts/validate_logic_sim.py`:

```
[PASS] node voltages agree with src/circuit.py   max |dV| = 2.15e-08 V
[PASS] did/dvgs matches finite differences       max rel err 9.15e-10
[PASS] did/dvds matches finite differences       max rel err 1.56e-07   (44% of samples reverse-biased)
[PASS] ID(VDS=0) == 0                            max |ID| = 0.0e+00 A
[PASS] odd in VDS under terminal swap            max asymmetry 0.0e+00
[PASS] caps continuous through VDS=0             max jump 7.6e-07 %
[PASS] transient vs analytic RC step             max |err| = 8.9e-04 V
```

The fast DC screen (section 3) reproduces the full nodal engine to **8e-7 V**.

`scripts/crosscheck_engines.py` runs the same 3-stage chain through
`src/logic_sim.py` and through `src/tran.py`, which was written independently
for the NAND/DFF/counter study on this branch and differs in the integrator
(backward Euler vs variable-step BDF2/Gear-2), the net forward pass, the t = 0
state and the node shunt:

| node | max difference | rms |
|---|---|---|
| stage 1 out | 16.1 mV | 1.7 mV |
| stage 2 out | 13.9 mV | 1.6 mV |
| stage 3 out | 10.1 mV | 1.1 mV |
| bootstrap node X | 16.7 mV | 1.8 mV |

on ~3.0 V swings, with the output's 50% crossings landing within **0.6 ns on a
40 us period (0.002%)**. `src/tran.py` itself matches **ngspice-42** to 16.7 mV
peak on a NAND2 (`outputs/logic_test/ngspice_check.json`), so that validation
carries across to the ring results here.

### The `.va`'s capacitance formulation cannot run this circuit

The `.va` writes `I(G,D) <+ ddt(cgd*V(G,D))`, i.e. Q = C(V)*V, so a simulator
sees dQ/dV = C + V*dC/dV. Because dC/dV is large and negative on the turn-on
knee, **the effective capacitance goes negative**:

| device | C range | effective dQ/dV | negative over |
|---|---|---|---|
| W160/L5, VD = 1.0 V | 0.19 .. 1.63 pF | **-0.47** .. +1.66 pF | 5.4% of VG |
| W160/L5, VD = 2.5 V | 0.16 .. 1.62 pF | **-4.60** .. +1.65 pF | 10.7% of VG |
| W160/L20, VD = 2.5 V | 0.75 .. 6.24 pF | **-17.57** .. +6.40 pF | 12.1% of VG |

Every logic node crosses that knee on every transition. Run this way
(`cap_mode="cv"`), the 5-stage ring stalls after a few tens of time steps at
**every one of the 23 supplies swept** -- the step collapses and Newton fails.
ngspice aborts on the same form (`"transient aborted at t = 15.41 us of 30 us"`),
so it is the model, not the solver.

Results here therefore use the incremental form, C as the calibrated incremental
capacitance. That is exactly what `verilogA/ntft_full_cdv.va` on this branch
implements (`I(G,D) <+ cgd*ddt(V(G,D))`), arrived at independently from the same
diagnosis -- so these numbers are what that `.va` produces in a real simulator.
It extends item 4 of `outputs/va_test/README.md` from "up to 303% off" to "not
simulable for digital work".

---

## 3. Which NOT gate: four unipolar styles

All at W1/W2/W3/W4 = 5/160/160/160 um, L = 5 um, VDD = 3 V
(`topology_comparison.csv`, `topology_comparison.png`):

| style | devices | gain | VOH | VOL | swing | SNM | P_static | usable |
|---|---|---|---|---|---|---|---|---|
| pE  zero-VGS load | 2 | 0.16 | 0.039 | 0.014 | 0.9% | 0.015 | 0.03 uW | no |
| pD  diode load | 2 | 3.47 | 2.321 | 0.071 | 75.0% | 0.661 | 7.3 uW | yes |
| **pC  pseudo-CMOS** | 4 | **4.41** | 2.312 | **0.035** | 75.9% | **0.761** | 8.0 uW | yes |
| pCz pseudo-CMOS, zero-VGS first stage | 4 | 0.46 | 0.121 | 0.031 | 3.0% | 0.027 | 0.37 uW | no |

pE is dead on arrival -- its pull-up is the device's own off-current, so VOH is
39 mV. pD works but its VOL is set purely by the W ratio. pC wins on what
matters: it drives the pull-up's *gate* from the inverted node X, so the pull-up
is switched off while the pull-down pulls down. That decouples VOL from the
ratio (0.035 V against 0.071 V) and roughly doubles the butterfly eye.

**Bootstrapping is invisible in DC** -- a capacitor does nothing at DC, so pC and
pCb share a VTC exactly. It is chosen in section 5, on transient.

### The high-gain trap, and a metric that was wrong

A 1296-point sizing screen of each 4T variant (`sizing_screen.csv`) finds pCz
reaching **max |dVout/dVin| = 118 and a 1.16 V butterfly eye at 65 nW** -- 25x
the gain of pC at 1/100 the power. It never oscillates, in a window 300x longer
than it needs (`scripts/pczb_check.py`), and the reason is not only that a
zero-VGS load drives node X at a fraction of a nA into ~3 pF:

```
pCz 5/10/160/40   VM = 0.2378 V   slope AT VM = -0.53   max |slope| = 220 at VIN = 0.0045 V
pC  5/20/160/40   VM = 1.0326 V   slope AT VM = -2.05   max |slope| =  2.6 at VIN = 0.799  V
```

pCz's headline gain lives at VIN = 4.5 mV, nowhere near its own trip point,
where the slope is **0.53**. Its symmetric ring point is therefore *stable* and
the ring latches. **`max |dVout/dVin|` is the wrong number for a cascadable
gate**; the slope at the trip point is the one that decides. `gain_vm` was added
to `src/pseudo_cmos.py` and is what the sizing in section 4 ranks on.

A second caveat: pCz's gain rests on the output conductance of a device at
VGS ~ 0, the least trustworthy corner of this model (`outputs/va_test/README.md`
already flags gds as 5-20x too high). Nothing below depends on it.

---

## 4. Sizing the bootstrapped gate

A DC screen cannot size a bootstrapped gate, and its delay proxy is actively
misleading: it counts M3's CGS as load when that capacitor *is* the gate's
intrinsic bootstrap, sitting between X and OUT. So sizing was decided on the
real figure of merit -- a 5-stage ring -- in three phases, 296 ring transients
(`ring5_sizing_phaseA/B/C.csv`, combined in `ring5_sizing_all.csv`).

### The capacitance budget at node X is the whole design

Node X carries three capacitors and they do not pull the same way
(W1/W2/W3 = 5/160/160 um, VDD = 3 V):

| cap | between | value | effect |
|---|---|---|---|
| CGS(M3) | X <-> OUT | 1.489 pF | **bootstrap**: couples the rising output into the pull-up's own gate |
| CGD(M2) | X <-> IN | 1.488 pF | **anti-bootstrap**: the input falls while the output rises, dragging X down |
| CGS(M1) | X <-> VDD | 0.049 pF | plain load |

At W2 = W3 = 160 um the two cancel almost exactly, which is why the textbook
sizing bootstraps not at all. Shrinking W2 fixes it: at W2 = 20 um, CGD(M2)
falls to 0.19 pF and the coupling ratio into X goes from 0.49 to 0.91. Measured,
Cboot = 0 throughout:

| W1/W2/W3/W4 | f | swing | max VX (VDD = 3 V) |
|---|---|---|---|
| 5/160/160/160 | 123.4 kHz | 65.9% | 2.38 V |
| 5/40/160/160 | 162.8 kHz | 72.2% | 2.65 V |
| 5/40/160/80 | 156.4 kHz | 85.0% | 2.82 V |
| 5/20/160/40 | 212.0 kHz | 84.4% | 2.80 V |

A weaker pull-down (W4 < W3) helps for a second reason: it lengthens the
half-period relative to the rise, giving M1 time to get X near VDD before the
output rises -- and X must *reach* VDD before a capacitor can push it past.

### Phase C: is W3 = 160 um an artefact of searching at Cboot = 0?

Phase A ran at Cboot = 0, which rewards a large W3 purely because M3's own CGS
is the bootstrap capacitor. Re-sweeping W3 and Cboot jointly (108 runs) says no:

| best at each W3 | f | swing | P |
|---|---|---|---|
| W3 = 40 um | 150.5 kHz | 83.7% | 38.1 uW |
| W3 = 80 um | 159.6 kHz | 82.3% | 46.2 uW |
| **W3 = 160 um** | **218.1 kHz** | **92.8%** | 54.7 uW |

### A ring is not a logic gate

Ranking on ring performance alone (f x swing^3) lands on 5/10/160/40 @ 2 pF:
225.5 kHz at 91.7% swing -- whose loop gain at the trip point is **1.63**,
barely regenerative. The stated rule is therefore: among designs swinging >= 90%
of VDD, take the (frequency, loop-gain-at-VM) Pareto front and choose the
**fastest point whose loop gain clears 2.0**, then the smallest Cboot on that
sizing reaching 92% swing.

| W1/W2/W3/W4 | Cboot | f | swing | loop gain at VM | SNM | P |
|---|---|---|---|---|---|---|
| 10/10/160/40 | 4 pF | 235.2 kHz | 90.3% | 1.45 | 0.362 | 65.7 uW |
| 5/10/160/40 | 2 pF | 225.5 kHz | 91.7% | 1.63 | 0.413 | 54.3 uW |
| 10/20/160/40 | 1 pF | 212.6 kHz | 90.1% | 1.82 | 0.465 | 60.2 uW |
| **5/20/160/40** | **1 pF** | **194.3 kHz** | **93.0%** | **2.05** | **0.516** | **49.1 uW** |
| 10/40/160/40 | 0.5 pF | 170.5 kHz | 91.0% | 2.47 | 0.535 | 60.9 uW |
| 5/40/160/40 | 0.5 pF | 154.6 kHz | 93.9% | 2.73 | 0.585 | 48.9 uW |

Giving up 14% of the frequency buys 26% more loop gain, 25% more noise margin, a
better swing and 10% less power. The front is in `ring5_sizing_all.csv` if the
other trade is wanted.

---

## 5. What the bootstrap capacitor does

`ring5_bootstrap_waveforms.png` shows it directly: with Cboot the pull-up's gate
rides *above* the supply, so M3 -- a source follower, whose VGS collapses as its
own source rises -- keeps drive all the way to the rail instead of stalling a
threshold short of it.

Cboot sweep on the recommended sizing, 5-stage ring, VDD = 3 V:

| Cboot | f | swing | max VX | VX - VDD |
|---|---|---|---|---|
| 0 | 212.0 kHz | 84.4% | 2.799 V | **-0.201 V** |
| 0.5 pF | 200.7 kHz | 90.8% | 3.077 V | +0.077 V |
| **1 pF** | **194.3 kHz** | **93.0%** | **3.207 V** | **+0.207 V** |
| 2 pF | 186.0 kHz | 94.1% | 3.299 V | +0.299 V |
| 4 pF | 175.8 kHz | 94.2% | 3.333 V | +0.333 V |
| 8 pF | 167.2 kHz | 93.7% | 3.317 V | +0.317 V |

Monotonic and saturating: swing climbs 84% -> 94% and stops, while frequency
falls steadily because the capacitor is also load. Past ~2 pF only the cost is
left. **The bootstrap turns an 84%-swing oscillator into a 93%-swing one at a
cost of 8% in frequency** -- it is a swing feature, not a speed feature.

M1 is what makes it work. Its gate and drain are both on VDD, so the moment X
rises above VDD its VGS goes to zero and it stops conducting: a diode that lets
the capacitor push X past the rail. This is also why the engine had to be made
D/S-symmetric -- the raw `.va` would have "blocked" there for the wrong reason
(an input clamp rather than the physics) and reported a boost it had not earned.

---

## 6. Five stages: supply voltage and frequency

`ring5_vdd_sweep.csv`, `ring5_vdd_sweep.png`. 23 supplies from 0.4 V to 5.0 V,
the whole range the model can be held responsible for.

| VDD | f | swing | VOL | VOH | max VX | VX-VDD | P | E/transition | tpd |
|---|---|---|---|---|---|---|---|---|---|
| 0.4-0.7 V | — | latched | | | | | | | |
| 0.80 V | 61.0 kHz | 35.2% | 0.254 | 0.535 | 0.502 | -0.298 | 0.80 uW | 1.31 pJ | 1640 ns |
| 0.90 V | 62.8 kHz | 51.7% | 0.204 | 0.670 | 0.651 | -0.249 | 1.15 uW | 1.83 pJ | 1592 ns |
| 1.00 V | 66.2 kHz | 61.6% | 0.175 | 0.791 | 0.787 | -0.213 | 1.60 uW | 2.42 pJ | 1511 ns |
| 1.50 V | 91.6 kHz | 84.6% | 0.102 | 1.371 | 1.456 | -0.044 | 5.57 uW | 6.09 pJ | 1092 ns |
| 2.00 V | 123.4 kHz | 91.4% | 0.077 | 1.905 | 2.099 | +0.099 | 13.4 uW | 10.9 pJ | 811 ns |
| 2.50 V | 159.0 kHz | 93.5% | 0.071 | 2.408 | 2.697 | +0.197 | 27.2 uW | 17.1 pJ | 629 ns |
| **2.75 V** | 177.0 kHz | **93.6%** | 0.071 | 2.644 | 2.965 | **+0.215** | 37.1 uW | 20.9 pJ | 565 ns |
| **3.00 V** | **194.3 kHz** | 93.0% | 0.073 | 2.864 | 3.207 | +0.207 | 49.1 uW | 25.3 pJ | 515 ns |
| 3.50 V | 226.0 kHz | 90.1% | 0.083 | 3.237 | 3.614 | +0.114 | 79.9 uW | 35.3 pJ | 442 ns |
| 4.00 V | 255.1 kHz | 86.0% | 0.098 | 3.536 | 3.949 | -0.051 | 120 uW | 47.2 pJ | 392 ns |
| 4.50 V | 281.8 kHz | 82.2% | 0.117 | 3.814 | 4.265 | -0.235 | 172 uW | 60.9 pJ | 355 ns |
| 5.00 V | 307.3 kHz | 78.9% | 0.141 | 4.086 | 4.577 | -0.423 | 237 uW | 77.0 pJ | 325 ns |

### Best ranges of operation

| criterion | window |
|---|---|
| oscillates at all | **0.80 - 5.00 V** |
| swing >= 50% of VDD | 0.90 - 5.00 V (62.8 - 307.3 kHz) |
| swing >= 80% of VDD | 1.50 - 4.75 V (91.6 - 294.6 kHz) |
| **swing >= 90% of VDD** | **2.00 - 3.50 V (123.4 - 226.0 kHz)** |
| bootstrap actually boosting (VX > VDD) | 1.75 - 3.75 V, peaking +0.215 V at 2.75 V |
| best swing | 93.6% at 2.75 V |
| lowest energy/transition | 1.31 pJ at 0.80 V, but at 35% swing — the lowest *usable* point is 10.9 pJ at 2.0 V |
| highest frequency | 307 kHz at 5.0 V, but at 79% swing — the fastest point with full swing is **226 kHz at 3.5 V** |
| compact model inside its trained box | everywhere below 4.75 V (0.8% / 1.6% of time steps out of box at 4.75 / 5.0 V) |

**Recommended operating point: 2.5-3.0 V.** That is where the swing (93.5%), the
bootstrap boost (+0.20 V) and the model's validity all sit comfortably, at
159-194 kHz for 27-49 uW.

### Why the window closes at each end

**Below 0.8 V** the stage gain falls under the Barkhausen threshold -- see
section 7, where the floor is traced to |A| >= sec(pi/N) and depends on the
stage count.

**Above 3.5 V** the gate becomes rise-time limited. The bootstrap boost shrinks
and then reverses (+0.215 V at 2.75 V, -0.423 V at 5 V): the ring is now fast
enough that the half-period no longer gives M1 time to charge X to VDD before
the output rises, and a capacitor cannot boost a node that never got to the rail.
VOH/VDD falls from 97% to 82% and the swing follows. This is a *ring* limit, not
a gate limit, and section 8 shows it lifting as soon as the ring is longer.

### Scaling

**f ~ VDD^1.00** (fit residual 0.023 in ln f over 1-5 V). Frequency is linear in
the supply, which is what `f = I/(C*V)` gives when the drive current is roughly
quadratic in VDD. Total power goes as VDD^3.14 and energy per transition as
VDD^2.21 -- the latter is not CV^2: this logic is static-power dominated
(7.6 uW/gate static against 49.1 uW total for five gates), so energy per
transition is static current integrated over a period, and it has **no interior
minimum**. The cheapest transition is always at the lowest supply that still
swings.

### What the frequency is actually worth

| source of error | effect on f |
|---|---|
| time step (backward Euler, Richardson-extrapolated) | **+0.13%** |
| two independent engines (BE vs Gear-2) | 0.6 ns on a 40 us period, **0.002%** |
| charge formulation (`ddt(C*V)` vs `C*ddt(V)`) | the `.va`'s own form does not run at all |
| **capacitance magnitude, x0.5 / x2** | **+91% / -48%** |

The capacitance model dominates everything else by two orders of magnitude, and
it is the part with the weakest evidence: CGD/CGS were trained at four
geometries, none shorter than L = 15 um, so L = 5 um logic runs on an
area-scaling extrapolation. The extrapolation is at least self-consistent --
1.63 pF on W160/L5 against 6.28 pF on W160/L20 is 2.04 vs 1.96 fF/um^2 -- but
**every frequency here should be read as a factor-of-two band, not a number.**
Measuring C-V on a short-channel device is the single highest-value experiment
for this model.
