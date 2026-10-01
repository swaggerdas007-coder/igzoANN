# AC / charge model for the CAMCAS a-IGZO TFT -- analysis of the measured C-V data

This folder holds an analysis and a proposal only. **No `.va` file was modified.** The
validated DC model (`verilogA/tft_camcas_thesis.va`, DC equations and
parameters) is untouched. The proposed Verilog-A change is
[`proposed_ac_patch.diff`](proposed_ac_patch.diff). It was verified on a scratch
copy: the DC current is bit-identical, and the charges match the Python
reference to 2e-12.

Reproduce: `scripts/camcas_ac/run_all.sh` (about 1 min).

## Summary

| question | answer from the data |
|---|---|
| What do the files measure? | Small-signal Cp (Cp-G model) at **10 kHz, 50 mV AC**, VG = -3 to 5 V in 0.2 V steps. "cg" has S and D on the CMU low terminal. In **"cgd"/"cgs" the other electrode was floating**: off-state they read one overlap capacitance, on-state they read the *total* gate capacitance. They are **not** intrinsic Cgd/Cgs. |
| Quasi-static at 10 kHz? | On and off plateaus: yes. C(f) is flat to ±0.7% from 1 to 200 kHz at VG = 5 V, and G/ω is ≤1% of C. The turn-on transition (about ±0.3 V) is not: there the channel resistance limits charging, and G/ω reaches 20% of C. |
| Geometry scaling | **Every plateau scales with W·L**, including the off-state "overlap". Channel: 1.478 fF/µm² (leave-one-geometry-out error ≤2.4%). Off-state: 0.478 fF/µm² (≤6.6%). Scaling with W alone is rejected (about 30% errors). |
| Is Cgd overlap-only? | The measured "cgd" is strongly bias dependent, but only because of the floating source. At VDS = 0 a symmetric device must split the channel charge equally (verified for the proposed model). The existing model gives all channel charge to the source and Cgd = overlap only, which is wrong at VDS = 0 by Cch/2. |
| VDS dependence | **Not determinable from these data.** The "Vd=x" family shows none of the expected saturation drop. Its apparent VD dependence tracks the measurement time (bias-stress drift), not VD. |
| Existing model | As coded: overlaps **1200-1600x** too large and Cgg **344-355x** too large, because of unit errors including `Lov = L*10e-5`. As the thesis intended (Table 4.1): Cgg is **5.7-5.9x too small**, because SiO₂ 100 nm is assumed while the measured value corresponds to about 23 nm EOT. Both versions are bias independent. |
| Proposed model | Charge-based, charge-conserving and S/D-symmetric. Overlap charges are linear and ∝ W·L. Channel charge is Ward-Dutton with an effective overdrive fitted to the measured C(VG) shape. The turn-on is tied to the DC model's Von_eff. On-state Cgg error is **1.4% mean (max 6.7%)**, off-state **2.4% (max 7.8%)**, over all 4 geometries. |

---

## 1. The four AC datasets

Source: the raw B1500 exports in `data/` (`scripts/camcas_ac/cvdata.py` parses
them; every number is used in SI units as exported: C in F, G in S). Summary:
`ac_step1_summary.csv`. Plots: `plots/ac_step1_*.png`.

| family | files | instruments (from file header) | sweep | notes |
|---|---|---|---|---|
| basic | `C-V [W-L cg / cgd / cgs]` | CMU only (`Channel.Unit CMU1:MF/SC`), no SMU | VG -3 to 5 V, 0.2 V | Cp-G, 10 kHz, 50 mV, `Bias.Source 0` |
| Vd | `C-V Vd=x [W-L cg]`, x = 0.1, 0.2, 0.3, 0.4, 4, 4.5, 5 | CMU on the gate + **SMU1 on the drain** (VD constant) | VG -3 to 5 V | taken sequentially, about 37 s apart, in VD order |
| fine | `C-V fc [W-L cg]` | CMU only | VG -1 to 1 V, 0.05 V | |
| C-f | `Generic C-f [160-20 cg]` | CMU | 1 kHz to 4.5 MHz at VG = 5.007 V | |

Geometries are 20x20, 40x20, 160x15 and 160x20 um (W x L), with 41 points
per sweep. The plateau noise is 0.3-1.3 fF rms.

**What "cg", "cgd" and "cgs" actually measure.** The CMU is the only
instrument in these files, so the third terminal is not AC-grounded. The data
show that it was **floating**:
- In the on-state, all three agree: 160x20 at VG = 5 V gives 6.250 / 6.251 /
  6.258 pF. With the other electrode AC-grounded, each of "cgd" and "cgs"
  would be about half of cg, and they would sum to cg.
- In the off-state, "cgd" + "cgs" = cg to within 13-20 fF
  (160x20: 0.846 + 0.687 = 1.533 vs 1.514 pF). The small excess is the series
  path through the floating electrode.

So the off plateau of "cgd" ("cgs") is the gate-drain (gate-source) overlap
capacitance, measured separately. Once the channel conducts, the floating
electrode is tied to the measured one and the sweep reads the full Cgg. The
earlier `data_cv_cleaned/README.md` describes the non-swept terminal as held at
0 V, which the data contradict for the AC signal.

**Measurement artefacts and limits.**
- **Bias-stress drift of the turn-on.** The mid-transition voltage rises
  monotonically with measurement time on every device. Within the Vd family,
  20x20 goes from -0.43 to +0.22 V over 3.7 min. The *next* sweep, at
  VD = 0, continues to +0.27 V, so the shift follows time, not VD. 160x20
  partially recovers after a 55 min rest (0.62 → 0.48 V). The plateaus do
  not drift: VG = 5 V values agree within 0.5% across all sweeps of a device.
- **Non-quasi-static transition.** G/ω peaks at the turn-on (up to 20% of C)
  and is ≤1% of C beyond VG ≈ V0 + 1.5 V (`plots/ac_step1_loss.png`). The
  floating-electrode sweeps charge the channel from one end only, so they turn
  on later than cg.
- **The Vd family cannot be used for VDS dependence.**
  - At VG = 2-5 V with VD = 4-5 V, the device should be in saturation, where
    any charge model puts Cgg at about Cov + ⅔Cch: a ~1.6 pF (25%) drop for
    160x20.
  - The data change by ≤1.2% at VG ≥ 2 V (6.246 → 6.238 pF at VG = 5 V).
  - In the transition region the "VD dependence" is non-physical. For 160x15
    at VG = 0.4 V, C collapses between VD = 0.1 and 0.3 V, then does not
    change from 0.4 to 4 V. That is the time-ordered drift.
  - Most likely the drain bias was not effective in this set-up (for example,
    the drain also held on the CMU low terminal). The files don't record the
    drain current, so this cannot be confirmed.
- **Frequency.** The C-f sweep exists at one bias only (VG = 5 V, 160x20): it
  is flat to ±0.7% from 1 kHz to 200 kHz. The rise above about 700 kHz is the
  cable/instrument resonance.

## 2. Geometry scaling

See `ac_scaling.csv` and `plots/ac_step2_scaling.png`. The plateau values are
listed in the table below. "C_ch" is the fitted asymptotic channel amplitude.

| W x L | C_g,off | C_ov,d ("cgd" off) | C_ov,s ("cgs" off) | C_g at VG = 5 | C_ch |
|---|---|---|---|---|---|
| 20 x 20 | 0.190 pF | 0.110 | 0.093 | 0.807 | 0.606 |
| 40 x 20 | 0.408 | 0.230 | 0.195 | 1.613 | 1.19 |
| 160 x 15 | 1.159 | 0.657 | 0.522 | 4.767 | 3.58 |
| 160 x 20 | 1.514 | 0.846 | 0.687 | 6.250 | 4.70 |

The requested pairwise comparisons:

| pair | W ratio | L ratio | C_off | C_ov,d | C_ov,s | C_ch | C_on |
|---|---|---|---|---|---|---|---|
| 40x20 / 20x20 | 2 | 1 | 2.15 | 2.10 | 2.09 | 1.95 | 2.00 |
| 160x20 / 40x20 | 4 | 1 | 3.71 | 3.67 | 3.53 | 3.93 | 3.88 |
| 160x20 / 160x15 | 1 | 1.33 | 1.31 | 1.29 | 1.32 | 1.31 | 1.31 |

Leave-one-geometry-out errors of candidate laws (max over the 4 geometries):

| quantity | ∝ W | **∝ W·L** | a·W + b·W·L |
|---|---|---|---|
| C_ch | 31% | **2.4%** | 2.9% |
| C_g,off | 31% | **6.6%** | 7.2% |
| C_ov,d | 30% | **7.0%** | 8.1% |
| C_ov,s | 33% | **11.3%** | 12.4% |

**Channel.** C_ch = 1.478 fF/µm² × W·L. With the thesis's ε_ox = 3.45e-11 F/m,
this corresponds to **EOT ≈ 23 nm**, not the 100 nm assumed in Appendix F.
The 160x15/160x20 pair alone would give L_eff = L + 1.0 µm. That value is
close to the DC TLM ΔL (≈ -0.8 µm at W = 160), but it rests on two devices
and is not used.

**Overlap.** The off-state capacitance grows with L at fixed W (ratio 1.31 for
an L ratio of 1.33). So the S/D overlap length scales with L in this layout,
the same convention the thesis assumed (L_OV = L/10). For the same dielectric
it corresponds to **L_ov,d ≈ 0.18·L and L_ov,s ≈ 0.14·L**. The drain overlap
is consistently 18-26% larger than the source overlap on all 4 geometries.
This could be layout asymmetry or probe parasitics. It is kept as measured.

## 3. Overlap vs channel, and the existing model

**Separation.** The off plateau gives overlap only: the channel is depleted
and G/ω ≈ 0. C_ov,d and C_ov,s come from "cgd"/"cgs" off. They are rescaled
so they sum to the directly measured cg off plateau, which removes the
13-20 fF floating-electrode series path while keeping the measured d/s ratio.
The channel contribution is Cgg(VG) - C_g,off.

**Cgd bias dependence.** The measured "cgd" changes about 7x with VG. As shown
above, that comes from the floating source, so it does not show what part of
the channel charge belongs to the drain. What the data *do* show is the total
channel charge at VDS = 0. A symmetric device must split it equally between S
and D at VDS = 0, so the intrinsic Cgd at VDS = 0 is Cch/2 + C_ov,d, not
C_ov,d. The existing model (`Qch = CCH*V(g,s)`, all to the source) is
therefore wrong at VDS = 0. Partition away from VDS = 0 cannot be determined
from these data.

**The existing charge block**, `ac_baseline_comparison.csv` and
`plots/ac_step3_baseline.png`:

The thesis (Sec. 4.2) states that *"direct C-V measurements were not
available"*. Its capacitances are geometric assumptions: L_OV = L/10 per side,
L_CH = 0.8·L, SiO₂ 100 nm. Table 4.1 (W = 100, L = 20 µm: C_OV = 0.069 pF,
C_CH = 0.552 pF) is reproduced exactly by Eqs. 4.10-4.12 with W and L in
metres.

The code in Listing F.1, with W and L in µm, does something else:
- `Lov = L*10e-5` = L·1e-4. Intended was 0.1·L µm = **L·1e-7 m**, so it is
  1000x too large.
- `W*1e-5` in all three capacitances should be `W*1e-6` (µm → m), 10x too
  large.
- Together: Covs = Covd = 3.45e-13·W·L F (1.1 nF for 160x20, **10⁴x the
  thesis's own Table 4.1**), and CCH = 2.76e-15·W·L F (10x Table 4.1).

`Lov = L*10e-5` is therefore a unit-conversion error, not a convention. No
consistent unit choice makes both Covs and CCH match Table 4.1.

| ratio model/measured | Cov,d | Cov,s | Cg,off | Cch | Cg,on |
|---|---|---|---|---|---|
| as coded | 1200-1310 | 1420-1610 | 1360-1460 | 1.8 | 344-355 |
| thesis intent | 0.12-0.13 | 0.14-0.16 | 0.68-0.73 | 0.18-0.19 | 0.17-0.18 |
| **proposed** | 0.92-1.00 | 0.88-0.99 | 0.94-1.01 | 0.96-1.00 | 0.97-1.00 |

Also note thesis Eq. 4.14/4.15, C_High = C_OV + ⅔C_CH. This is the
*saturation* partition. A C-V sweep at VDS = 0 contains the full C_CH, so
Eq. 4.15 would overestimate C_CH by 1.5x if it were applied to these data.

## 4. Which model the data support

- **Constant capacitances** (options 1 and 2) are rejected. Cgg changes about
  4x between the off and on states. Even the on-plateau alone rises 9-12%
  from VG = 1 V to 5 V, and that rise is quasi-static (G/ω ≤ 1%).
- **Bias-dependent C at VDS = 0** is required, and the data pin it down well.
  Normalized by its own plateaus and aligned at its own turn-on V0, every
  curve (4 geometries x 2 stress states) has **the same shape**: 84-87%
  charged at V0 + 0.5 V, 90-93% at +1 V, 96-97% at +2 V
  (`plots/ac_step4_shape.png`).
  - A single logistic step cannot follow the slow approach to the plateau
    (3.1% rms, 11% max).
  - A sharp step plus a broad step does (**1.27% rms**, 8% max at the
    non-quasi-static edge).
  - The sharp width is fixed from the DC model's subthreshold swing,
    s₁ = SS/ln10 = 68.5 mV/dec / 2.303 = 30 mV. The non-quasi-static
    transition points cannot constrain it; left free it collapsed to 15 mV,
    below kT/q.
- **A charge-based model** (option 4) is needed for the implementation, not
  because of extra data. Independent bias-dependent Cgs/Cgd expressions do not
  conserve charge. The simplest conserving form that is S/D symmetric and
  reduces exactly to the measured VDS = 0 curves is the Ward-Dutton
  long-channel charge with smooth effective overdrives. It adds **no
  parameters** beyond the VDS = 0 fit.
  - Its VDS behaviour (Cgd → C_ov,d and Cgs → C_ov,s + ⅔Cch in saturation;
    `plots/ac_step5_vds.png`) is standard long-channel physics. **It is an
    assumption, not validated by these data.**
  - A "two halves" split (½Cch on each side) is equally consistent with the
    data and gives ½ instead of ⅔ in saturation. Only a VDS-resolved split
    C-V can discriminate between them.

## 5. Proposed charge model

All lengths are in µm and areas are converted explicitly: A = W·L·1e-12 m².

```
Overlap:   Qov,s = C'ov,s·A·VGS        C'ov,s = 2.135e-4 F/m²
           Qov,d = C'ov,d·A·VGD        C'ov,d = 2.643e-4 F/m²
Channel:   Cch = C'ch·A                C'ch   = 1.478e-3 F/m²   (EOT ≈ 23 nm)
  effective overdrive (V), the integral of the normalized measured C(VG):
     veff(v) = (1-β)·s1·ln(1+e^{v/s1}) + β·s2·ln(1+e^{(v-d2)/s2})
     s1 = 0.0297 V (DC SS/ln10), β = 0.318, s2 = 0.475 V, d2 = 0.469 V
  V0 = Von_eff(L, W) [validated DC model] + dV_C,   dV_C = +0.144 V
  Vs = veff(VGS - V0),  Vd = veff(VGD - V0)
  Qch = (2/3)·Cch·(Vs² + Vs·Vd + Vd²)/(Vs + Vd)
  Qd,ch = -Cch·(6Vd³ + 12Vd²Vs + 8VdVs² + 4Vs³)/(15·(Vs+Vd)²)
Terminals: Qg = Qch + Qov,s + Qov,d,  Qd = Qd,ch - Qov,d,  Qs = -Qg - Qd
Currents:  i_g = dQg/dt, i_d = dQd/dt, i_s = dQs/dt
```

At VDS = 0 this gives Cgg = (C'ov,s + C'ov,d)·A + Cch·n(VGS − V0), where n is
the fitted normalized shape. Each of the two channel sides then carries
Cch·n/2.

Checks (`ac_conservation_check.csv`, 200 random biases): Qg + Qd + Qs = 0 to
3e-16; column sums of C_ij to 1e-11 relative; row sums to 4e-9; channel
Cgs = Cgd at VDS = 0 to 2e-12.

## 6. Validation

See `ac_validation_metrics.csv`, `ac_floating_check.csv` and
`plots/ac_step6_*.png`. Errors are relative errors of Cgg against every
VDS = 0 total-gate-capacitance curve (basic cg and the VD = 0.1 V curve).
They come from one global parameter set: area coefficients ∝ W·L plus the
shape.

**V0 fitted per curve** (absorbs the bias-stress drift; tests geometry scaling
and shape):

| geometry | off-state mean / max | on-state mean / max | transition (non-quasi-static) mean |
|---|---|---|---|
| 20 x 20 | 1.4% / 5.1% | 2.0% / 6.7% | 22% |
| 40 x 20 | 5.2% / 7.8% | 1.8% / 4.3% | 45% |
| 160 x 15 | 1.4% / 5.4% | 0.9% / 2.7% | 25% |
| 160 x 20 | 1.6% / 3.3% | 1.1% / 2.1% | 23% |
| **all** | **2.4% / 7.8%** | **1.4% / 6.7%** | 31% (12 points) |

**V0 = DC Von_eff + dV_C (predictive):**
- On-state: 3.2% mean.
- Points near the turn-on are off by up to 60-200%, because the measured
  turn-on itself moves by up to 0.8 V between sweeps of the same device.
- Per-device offsets of the least-stressed sweeps are -0.18, -0.01, +0.30 and
  +0.76 V.

**Floating-electrode sweeps** ("cgd"/"cgs"): the off plateaus match C_ov,d and
C_ov,s to within 0-12%. The model sits low here by the 13-20 fF series path
that was removed. The on plateaus match Cgg to within 0.7-4%.

**One common model structure works for all four geometries.** The same
equations, three area coefficients ∝ W·L, and one shape fit every curve. The
largest geometry residual is C_ov,s at 40x20 (−12%).

## What the four geometries establish, and what they don't

**Established with confidence:**
- The measurement configurations, including the floating electrode in
  "cgd"/"cgs".
- The plateaus are quasi-static at 10 kHz.
- Channel capacitance per area (1.48 fF/µm², ±2%) and its W·L scaling.
- The off-state capacitance scales with W·L, not W. Overlap ∝ W alone is
  rejected at about 30% error.
- The d/s overlap asymmetry of about 1.2.
- A universal VDS = 0 bias shape with a slow, quasi-static approach to the
  plateau.
- The existing block has unit errors in the code and an assumed dielectric
  4.3x too thin in capacitance.

**Underdetermined:**
- **VDS dependence / charge partition.** There is no valid VDS-resolved split
  C-V. Ward-Dutton is assumed.
- **Turn-on position.** It drifts ±0.5 V with bias-stress history. The model
  ties it to the DC Von with a median offset; device-to-device spread is
  -0.2 to +0.8 V.
- **L scaling outside L = 15-20 µm.** The DC model covers L = 5-20 µm. W·L
  scaling at L = 5 and 10 µm is an extrapolation, and the L-dependence rests
  on one pair, 160x15 vs 160x20.
- **The origin of the W·L-scaled off-state capacitance**: an L-scaled overlap
  layout or an area parasitic. The model is the same either way; layout data
  would decide. A separate W-only overlap term is not resolvable (it fits to
  about 0).
- The transition-region dynamics (distributed channel RC), frequency
  dispersion away from VG = 5 V, and temperature. Each was either measured at
  one condition only or not at all.

**Measurements that would close these gaps:**
1. A split C-V with the third terminal AC-grounded (Cgs and Cgd separately),
   swept over VDS from 0 to 5 V at several VGS. This would test the partition.
2. The same at an L = 5 or 10 µm device.
3. Sweeps after a rest, or interleaved, to separate stress drift from bias.
4. C-f at 2-3 biases near the turn-on.

## 7. Proposed Verilog-A change (not applied)

[`proposed_ac_patch.diff`](proposed_ac_patch.diff), generated by
`scripts/camcas_ac/propose_va_patch.py` against the current
`verilogA/tft_camcas_thesis.va`:
- **Removed:** `eps_ox`, `tox`, `Lov`, `Lovch`, the old Covs/Covd/CCH/Qov*/Qch
  lines, and `I(g,s) <+ ddt(Qovs+Qch); I(g,d) <+ ddt(Qovd);`.
- **Added parameters:** `Cch_area`, `Cov_d_area`, `Cov_s_area` (F/m²), and
  `dV_C`, `sC1`, `betaC`, `sC2`, `dC2` (V or dimensionless).
- **Added functions:** an overflow-safe `softplus()` and `veff_c()`.
- **New charge block**, placed *after* the DC current so it can use
  `Von_eff`, with the area conversion `Aox = W*L*1e-12`:
  `I(g,s) <+ ddt(Qg); I(d,s) <+ ddt(Qd);`. Qs = −Qg − Qd follows from KCL,
  so charge is conserved.
- **DC equations are untouched.** On a scratch copy the patched file compiles
  under OpenVAF, and `Ids_total` is **bit-identical** (max difference 0 A) at
  every measured DC bias of all 19 devices. The Verilog-A charges equal the
  Python reference to 1.7e-12.

`tft_camcas_thesis.va` is generated by
`scripts/camcas_thesis/step4_export_va.py`. If the change is approved, it
should be made in that exporter's template so that regenerating the DC model
keeps the AC block.

## Files

| file | content |
|---|---|
| `ac_step1_summary.csv` | per-sweep plateaus, noise, turn-on, timestamps |
| `ac_scaling.csv` | geometry-law fits with leave-one-geometry-out errors |
| `ac_shape_fit_per_curve.csv`, `ac_model_params.json` | shape fit, fitted V0 per curve |
| `ac_model_params_final.json` | final parameters, including dV_C |
| `ac_validation_metrics.csv`, `ac_validation_points.csv` | Cgg errors by geometry, region and V0 mode |
| `ac_floating_check.csv` | "cgd"/"cgs" plateaus vs model |
| `ac_baseline_comparison.csv` | existing block (as coded / intended) vs measured vs proposed |
| `ac_conservation_check.csv` | charge conservation and symmetry checks |
| `proposed_ac_patch.diff` | the proposed Verilog-A change |
| `plots/` | `ac_step1_*` inspection, `ac_step2_scaling`, `ac_step3_baseline`, `ac_step4_shape`, `ac_step5_vds`, `ac_step6_*` validation |
