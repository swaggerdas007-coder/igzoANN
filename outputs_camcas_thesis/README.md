# CAMCAS compact model, thesis methodology, re-parameterized on our data

`verilogA/tft_camcas_thesis.va` is the Verilog-A model of Appendix F
(Listing F.1) of C. de Almeida, *Development of IGZO thin-film transistors
(TFTs) compact models* (NOVA FCT, 2025). The equations, module structure
and W/L-scaling functions are the thesis's. The coefficients were extracted
from our 19 measured geometries with the thesis's own procedure (Ch. 3).
Every model parameter is a function of the instance geometry:

```
W, L (um) -> *_func(L, W) scaling relations -> Von, alpha, kappa, G0, Ioff (lin, sat), DeltaL, RSD, m
          -> Appendix F current equations -> Id
```

The model contains no per-device values, lookup tables or ANN.

**Headline result.** One set of CAMCAS equations plus one set of
thesis-form scaling relations reproduces all 16 fit geometries to
**0.065 decades** mean |log10 error| (median relative error 8.6%). It
predicts the 3 never-used geometries to **0.111 decades** (median 10.8%).
Leave-one-geometry-out cross-validation inside the fit set gives
0.176 decades. The per-device extraction floor, meaning the same
equations with each device's *own* extracted parameters, is
0.02-0.12 decades. Details and caveats are in [Assessment](#step-6--assessment).

Reproduce everything: `scripts/camcas_thesis/run_all.sh` (about 8 minutes).

| step | script (`scripts/camcas_thesis/`) | main outputs (this folder) |
|---|---|---|
| 0 select curves | `step0_select_curves.py` | `step0_selection.csv`, `step0_candidates.csv`, `selected_curves/`, `plots/step0_*` |
| 1 extract parameters | `step1_extract_params.py` | **`step1_parameter_table.csv`** (master table), `step1_tlm.csv`, `plots/step1_*` |
| 2 inspect vs geometry | `step2_plot_params.py` | `plots/step2_*`, `step2_summary.csv` |
| 3 scaling fit | `step3_fit_scaling.py` | `step3_coefficients_<variant>.json`, `step3_per_width_quadratics_<variant>.csv` |
| 4 Verilog-A | `step4_export_va.py` | `verilogA/tft_camcas_thesis.va` |
| 5 simulate & compare | `step5_validate.py` | `step5_<variant>_*.csv`, `plots/step5_<variant>_*` |
| 6 one model for all? | `step6_improve.py`, `refine.py` | `step6_variant_comparison.csv`, `step6_logo_cv.csv`, `final_variant.txt` |
| 7 report | `step7_report.py` | `scaling_equations.md`, `final_metrics.csv`, `plots/step7_scaling_fits.png` |

The unseen geometries are **W5_L5, W10_L5 and W10_L10**. The 16 fit
geometries form the full grid W = 20, 40, 80, 160 um by L = 5, 10, 15, 20 um.
The unseen devices never enter extraction-to-scaling, the choice of
variant, or any coefficient. Note that they lie *below* the fitted width
range, so they test extrapolation in W, which is harder than interpolation.
The leave-one-geometry-out CV is the interpolation test.

---

## Step 0 -- curve selection

For every geometry, **one physical device** supplies both the transfer
curves (VD = 0.1 V and 5 V) and the output family. Parameters extracted
from the transfer curves therefore describe the same device whose output
curve is used for validation. The candidates are all QC-passing device
replicates in `data_cleaned/` (die sites bot1, bot2, top1, top2).

Hard rejects (`step0_candidates.csv`, column `reject`) are applied for:
- a noisy or non-monotonic on-state (a dip of more than 5%)
- off-state leakage
- compliance clipping
- a non-monotonic output trace or a crossing output family
- disagreement between the output and the same device's transfer curve at
  the same bias (drift between sweeps)

Among the survivors, the thesis rule (Sec. 3.3.2, "most closely aligned
with ideal IGZO TFT behavior ... minimal dispersion") picks the device
with the most typical turn-on, the best output/transfer agreement and the
smoothest output.

| W/L | device | | W/L | device | | W/L | device |
|---|---|---|---|---|---|---|---|
| 5/5 | top2 | | 40/5 | top1 | | 80/15 | bot1 |
| 10/5 | top1 | | 40/10 | bot1 | | 80/20 | bot1 |
| 10/10 | top2 | | 40/15 | bot1 | | 160/5 | bot1 |
| 20/5 | bot1 | | 40/20 | bot1 | | 160/10 | bot1 |
| 20/10 | top1 | | 80/5 | bot1 | | 160/15 | bot2 |
| 20/15 | bot1 | | 80/10 | top2 | | 160/20 | bot2 |
| 20/20 | bot1 | | | | | | |

Notes:
- Every chosen device turns on near the typical -0.35 V. Its output
  family agrees with its own transfer curves to within about 3%.
- `data_cleaned_2/` (used by the earlier CAMCAS work) chose the **bot2** die
  site almost everywhere. bot2 turns on late, which is a wafer-position
  effect rather than a geometry effect, and it would contaminate the W/L
  trends.
- Rejected, for example: W5_L5 top1 and W10 rep runs (±10-30% jumps in the
  on-state); W20_L10 bot1 (noisy linear curve); W160_L15 top2 (compliance
  clipped); W160_L20 bot1 (a decaying off-state leakage transient plus a
  ±20% glitch at VGS = 4.6-4.8 V in its linear sweep).
- **W160_L20** has only two live devices, and bot2 is the only acceptable
  one. Its curves are proper, but its normalized current I·L/W is 0.57x the
  typical value (bot1's on-state is typical), and its VGS = 4-5 V output
  traces have small (<2%) wiggles. It is a low-mobility die. See Steps 2
  and 6.

## Step 1 -- parameter extraction (thesis Sec. 3.2)

**Master table: `step1_parameter_table.csv`** (all 19 devices). The key
columns:

| device | Von_lin | Von_sat | α_lin | α_sat | κ_lin | κ_sat | G0_lin (S) | G0_sat (S) | Ioff_lin (A) | Ioff_sat (A) | ΔL (um) | RSD (kΩ) | m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W5_L5_top2 *(unseen)* | -0.5 | -0.5 | -0.255 | -0.197 | -8.47 | -9.93 | 0.00139 | 0.003 | 3.3e-12 | 2.8e-12 | 0.10 | 12.78 | 2.09 |
| W10_L5_top1 *(unseen)* | -0.4 | -0.4 | -0.340 | -0.341 | -7.21 | -6.53 | 0.000341 | 9.18e-05 | 2.9e-12 | 2.4e-12 | 0.10 | 12.78 | 2.06 |
| W10_L10_top2 *(unseen)* | -0.4 | -0.4 | -0.249 | -0.269 | -8.98 | -8.14 | 0.00214 | 0.00042 | 2.7e-12 | 3.1e-12 | 0.10 | 12.78 | 2.11 |
| W20_L5_bot1 | -0.4 | -0.6 | -0.392 | -0.370 | -6.51 | -6.36 | 0.000171 | 7.22e-05 | 3.1e-12 | 5.8e-12 | 0.42 | 5.52 | 2.08 |
| W20_L10_top1 | -0.5 | -0.4 | -0.319 | -0.328 | -7.49 | -7.07 | 0.000422 | 0.000126 | 3.2e-12 | 3.2e-12 | 0.42 | 5.52 | 2.03 |
| W20_L15_bot1 | -0.4 | -0.3 | -0.409 | -0.584 | -6.19 | -5.45 | 0.000129 | 1.8e-05 | 3.3e-12 | 2.9e-12 | 0.42 | 5.52 | 2.01 |
| W20_L20_bot1 | -0.2 | -0.2 | -0.282 | -0.403 | -7.29 | -5.83 | 0.000543 | 4.62e-05 | 3.1e-12 | 2.8e-12 | 0.42 | 5.52 | 2.04 |
| W40_L5_top1 | -0.4 | -0.4 | -0.461 | -0.386 | -6.00 | -6.75 | 0.000102 | 0.0001 | 3.2e-12 | 2.6e-12 | -1.29 | 0.00 | 2.02 |
| W40_L10_bot1 | -0.4 | -0.4 | -0.681 | -0.839 | -5.69 | -6.56 | 3.55e-05 | 1.15e-05 | 3.1e-12 | 3e-12 | -1.29 | 0.00 | 1.92 |
| W40_L15_bot1 | -0.6 | -0.3 | -0.425 | -0.575 | -5.77 | -5.40 | 0.000103 | 2.03e-05 | 3.5e-12 | 3.2e-12 | -1.29 | 0.00 | 1.99 |
| W40_L20_bot1 | -0.2 | -0.3 | -0.282 | -0.438 | -7.03 | -5.62 | 0.000486 | 3.85e-05 | 3.3e-12 | 3.1e-12 | -1.29 | 0.00 | 2.02 |
| W80_L5_bot1 | -0.6 | -0.8 | -0.502 | -0.420 | -6.22 | -7.66 | 8.49e-05 | 0.000118 | 3.1e-12 | 4.1e-12 | -0.58 | 0.00 | 2.13 |
| W80_L10_top2 | -0.4 | -0.5 | -0.500 | -0.439 | -5.50 | -5.87 | 5.96e-05 | 3.93e-05 | 3e-12 | 3.3e-12 | -0.58 | 0.00 | 1.89 |
| W80_L15_bot1 | -0.4 | -0.3 | -0.326 | -0.430 | -6.41 | -5.64 | 0.000246 | 4.14e-05 | 6.2e-12 | 2.8e-12 | -0.58 | 0.00 | 2.05 |
| W80_L20_bot1 | -0.5 | -0.5 | -0.546 | -0.749 | -5.54 | -5.50 | 5.14e-05 | 1.12e-05 | 2.8e-12 | 3e-12 | -0.58 | 0.00 | 2.10 |
| W160_L5_bot1 | -0.6 | -0.7 | -0.513 | -0.505 | -5.95 | -6.70 | 7.1e-05 | 4.69e-05 | 3.2e-12 | 3.5e-12 | -0.82 | 0.00 | 2.13 |
| W160_L10_bot1 | -0.5 | -0.4 | -0.475 | -0.543 | -5.85 | -6.12 | 7.8e-05 | 2.81e-05 | 3.4e-12 | 3.2e-12 | -0.82 | 0.00 | 2.00 |
| W160_L15_bot2 | -0.3 | -0.3 | -0.423 | -0.540 | -5.46 | -5.39 | 8.63e-05 | 2.22e-05 | 3.5e-12 | 3.1e-12 | -0.82 | 0.00 | 2.03 |
| W160_L20_bot2 | -0.1 | -0.1 | -0.572 | -0.527 | -4.06 | -5.94 | 1.6e-05 | 2.2e-05 | 2.8e-12 | 2.6e-12 | -0.82 | 0.00 | 2.72 |

Holdout rows (W5, W10) are extracted the same way for reference only. W5
has one length, so no TLM is possible there, and it borrows W10's
ΔL/RSD. Other columns in the table:
- the device's reference Von (`Von_eff`)
- the thesis-literal single-point Ioff
- R² of each U-function regression (0.95-0.995)
- normalized on-currents
- `own_mae_dec_*`: the error of the Appendix F equations with the
  device's own parameters, i.e. the floor for any scaling

How each quantity is obtained, following the thesis unless stated otherwise:

- **Von, Ioff**: the derivative-onset point, i.e. the first point from
  which dIDS/dVGS stays positive up to VH, on the linear and the saturation
  curve separately. On-state dips below 5% are treated as instrument noise.
  The larger of the two Von values is the device's reference Von for the
  α, κ and G0 extraction (Sec. 3.3.2-3.3.3). Ioff_lin and Ioff_sat are the
  currents at that Von (Table 3.10), *averaged over the 5 points ending at
  Von*. A single point is one sample of the ±5 pA instrument noise and
  scattered from 8e-14 to 2e-11 A; the average is 2.6-6e-12 A.
- **ΔL, RSD**: TLM per width (Fig. 3.1). Rtot = VDS/IDS at VDS = 0.1 V is
  plotted against L, with one line per VGS = 2...5 V. The least-squares
  common intersection gives (ΔL, RSD) (`plots/step1_tlm.png`,
  `step1_tlm.csv`).
  - The unconstrained intersection gives RSD = -6.6, -1.1 and -0.7 kΩ at
    W = 40, 80 and 160 um. The thesis's Dataset 2 shows the same
    (Table 3.7: -9.2 kΩ).
  - The reason: RSD is below this dataset's resolution. I·L/W at VGS = 5 V
    is flat across L = 5-20 um to within about 5%, which bounds RSD at
    about <1 kΩ at W = 20 um. The device-to-device scatter between the four
    L-devices then decides the sign.
  - The intersection is therefore constrained to RSD ≥ 0; both values are
    in the table. The lines visibly meet near R ≈ 0 with ΔL ≈ -0.6 to
    +0.4 um.
  - W160_L20 is excluded from the W = 160 TLM: its low mobility breaks the
    TLM assumption of equal sheet resistance.
- **α, κ**: U-functions (Eqs. 3.3/3.4) on I' = IDS − Ioff, with
  V' = VDS − RSD·IDS in the linear regime. ln U against ln(VGS − Von) is a
  straight line for every device (`plots/step1_ufunction.png`):
  α = 1 − slope and κ = e^(−intercept)/α. The points used are those with
  U > 0 and a device that is on (|ID| > 1e-10 A). Dropping the low-(VGS−Von)
  points makes α noisier (tested), because those points are what pin α.
- **G0**: Eqs. 2.32/2.33 at VH = 5 V.
- **m**: Eq. 2.35, m = 1/log2(Isat/Is), per on-state output trace. Isat is
  the saturation-branch current and Is is the measured current where the
  linear and saturation branches cross. The per-device median is
  1.9-2.1, close to the thesis's 2.2; W160_L20 gives 2.7.

Extraction quality: with its **own** parameters, each device's on-state
transfer and output curves are reproduced to **0.02-0.12 decades**
(roughly 5-30%). The CAMCAS equations describe these TFTs well.

## Step 2 -- parameters vs geometry

See `plots/step2_params_vs_L.png`, `plots/step2_params_vs_W.png`,
`plots/step2_correlations.png` and `step2_summary.csv`.

- **Von** rises with L, from about -0.6 V at L = 5 um to about -0.2 V at
  L = 20 um: a plausible short-channel trend. It is quantized by the 0.1 V
  sweep step. There is no W trend.
- **α, κ, G0 show no clean W/L trend individually, and they trade off
  against each other.** Across the 16 fit devices, G0 against α has
  r = +0.93 and κ against α has r = -0.75. G0 alone spans 30x. Their
  *product*, the channel conductance G(VH) = G0·exp(κ·(VH−Von)^α), is
  uniform to **±7%**. So W/L scaling itself holds very well on this
  process, but *how* a given curve is split between α, κ and G0 is noisy.
  The thesis reports the same situation for its Datasets 2 and 3 (Sec. 3.3:
  "no clear or systematic trends").
- **Ioff** is flat at the instrument floor (about 3e-12 A).
- **ΔL** has no trend (-1.3 to +0.4 um). **RSD** is zero within
  resolution, except for the W = 20 TLM (5.5 kΩ, from a mixed-die-site
  series).
- **W160_L20** is an outlier in every conductance-related quantity:
  G(VH) is 0.55x typical, m = 2.7 and κ_lin = -4.1.
- The unseen W5/W10 devices have a typical G(VH) but a shifted
  (α, κ, G0) triplet: α about -0.25 and κ about -8.5 to -10.

## Step 3 -- scaling relations (thesis Sec. 3.4)

These are the forms of Appendix F, unchanged:

- Von, α, κ, G0 and Ioff (lin and sat):
  **X(L, W) = (aW·W + a0)·L² + (bW·W + b0)·L + (cW·W + c0)**, quadratic in
  L with every coefficient linear in W (Eqs. 3.5-3.16).
- **ΔL(W)** and **RSD(W)**: straight lines in W (Eqs. 3.17/3.18).
- **m**: one global constant.

The thesis two-step procedure is used: a quadratic in L per width (the
analogue of Table 3.13, in `step3_per_width_quadratics_*.csv`), then a line
in W for each coefficient. We have 4 lengths and 4 widths instead of 3 and
2, so both steps are least-squares rather than exact. On the full 4x4 grid
this is identical to a single least-squares fit of the six coefficients;
the script asserts the identity. The single fit is used because it stays
defined when a grid point is missing.

**Final equations and coefficients: [`scaling_equations.md`](scaling_equations.md)**,
with the extracted values and fitted surfaces overlaid in
`plots/step7_scaling_fits.png`.

## Step 4 -- the Verilog-A

`verilogA/tft_camcas_thesis.va`, module `tft_camcas_thesis(d, g, s)`,
follows Listing F.1 line for line:
- instance parameters `W` and `L` **in um** (as in the thesis)
- ten `*_func(L, W)` functions, plus `DeltaL_func(W)`, `RSD_func(W)` and a
  global `m`
- Von_eff = max(Von_lin, Von_sat) and the off-state branch
- Ids_lin = (G·Vds + Ioff_lin)/(1 + RSD·G) and Ids_sat
- the harmonic-mean blend
- the thesis's overlap/channel charge block, copied verbatim; only DC data
  were fitted, so it was not re-extracted

The two deliberate differences are both on the RSD line:
1. `RSD_func` is in kΩ, as in Eq. 3.18 and Table 3.11, and is converted
   `*1e3` to the Ω that RSD·G needs. Listing F.1 divides by 100, which is
   not a unit conversion, and our G0 values were extracted with RSD in Ω.
2. RSD is clamped at 0, because the linear-in-W fit crosses zero inside the
   width range (see Step 1).

Inherited from Listing F.1 and kept as-is:
- a ~pA current at VDS = 0 (Ioff_lin in Ids_lin)
- no source/drain symmetry, so VDS < 0 gives a negative base for `**(-m)`

Add a source/drain swap before using the model in circuits that reverse
VDS.

The file compiles with OpenVAF, so it runs in ngspice/Xyce through OSDI,
and it uses only standard Verilog-A for Spectre.

## Step 5 -- simulation vs measurement

The `.va` is compiled by OpenVAF (through its Python front end
`verilogae`) and evaluated at every measured (VGS, VDS) of all 19 devices.
Every terminal is source-driven in the measurement, so this is exactly the
DC operating point a circuit simulator computes. A Python mirror of the
equations used for CV agrees with the compiled `.va` to below 1e-5.

Metrics are computed on on-state points (|ID| > 1e-10 A). The off-state is
the ±5 pA instrument floor: the model sits at about 3 pA there, and the
floor is reported separately in `step5_*_summary.csv`.
- `mae_dec` is the mean |log10(ID_va/ID_meas)|; 0.1 decade ≈ 26%.
- `bias` is the signed mean.
- `med_rel` is the median |ΔI|/I.

**Final model (`thesis_g`):**

| | sweep | points | MAE (dec) | RMSE (dec) | bias (dec) | median rel. err |
|---|---|---|---|---|---|---|
| **16 fit geometries** | linear transfer (VD = 0.1 V) | 839 | 0.055 | 0.117 | -0.006 | 4.9% |
| | saturation transfer (VD = 5 V) | 832 | 0.077 | 0.154 | +0.020 | 10.4% |
| | output (VG = -5...5, VD = 0.1...5) | 4700 | 0.065 | 0.102 | -0.016 | 9.1% |
| | **all** | 6371 | **0.065** | 0.112 | -0.010 | **8.6%** |
| **3 unseen geometries** | linear transfer | 160 | 0.103 | 0.245 | -0.082 | 8.5% |
| | saturation transfer | 160 | 0.111 | 0.231 | -0.109 | 13.5% |
| | output | 900 | 0.112 | 0.201 | -0.105 | 10.8% |
| | **all** | 1220 | **0.111** | 0.211 | -0.102 | **10.8%** |

Per geometry and sweep (`step5_thesis_g_per_geometry.csv`):
- W5_L5 (unseen) 0.17-0.21 dec
- W10_L5 (unseen) 0.06-0.07
- W10_L10 (unseen) 0.06-0.08
- fit geometries mostly 0.02-0.09, with these exceptions:
  - W40_L10: 0.10-0.26. This device turns on about 0.2 V later in
    saturation than its neighbours.
  - W80_L5 saturation: 0.12
  - W80_L15 output: 0.14
  - W80_L20 linear: 0.14
- W160_L20, the low-mobility die kept in the fit: 0.03-0.08

The plots are:
- `plots/step5_thesis_g_transfer_log.png` and `_transfer_lin.png`: Id-Vg,
  all 19 devices, log and linear scale
- `plots/step5_thesis_g_output.png`: Id-Vd, all 19
- `plots/step5_thesis_g_errors.png`: the systematic-error analysis
- the same four plots for the thesis-literal baseline, under
  `plots/step5_thesis_*`

Systematic errors (`step5_thesis_g_systematic.csv`, fit geometries):
- No trend with W or L: per-W bias is within ±0.05 dec and per-L bias
  within ±0.04 dec.
- By VGS: 0.20 dec at VGS ≤ 0 V (turn-on), 0.09 dec at 0-1 V and
  0.03-0.045 dec above 1 V. By current: 0.17-0.25 dec between 1e-10 and
  1e-7 A, and 0.03-0.05 dec above 1e-6 A. Turn-on is the hard region: the
  exponential is steep, so a 0.1 V Von error is worth about 1 decade, and
  Von is quantized to the 0.1 V sweep step.
- A mild VDS trend: 0.055 dec at 0.1 V rising to 0.073 dec at 4-5 V.
- On the output curves the model saturates slightly early and low at
  VGS = 4-5 V on some devices (W20_L5, W80_L5, W160_L5). This is the single
  global m of Appendix F; the per-device m spans 1.9-2.1.
- On the unseen devices the bias is -0.10 dec, i.e. about 20% low: the
  W<20 um devices sit slightly above the extrapolated W trend.

## Step 6 -- assessment

**Does one model work for all 19 devices? Yes, after one change in how
the thesis-form scaling coefficients are obtained.** The variants below
all use the same equations and scaling forms
(`step6_variant_comparison.csv`); "undefined" means the model returns NaN
at a geometry in the grid.

| variant | how coefficients are obtained | fit-16 MAE | leave-one-geometry-out CV | unseen-3 MAE |
|---|---|---|---|---|
| `thesis` | thesis procedure, all 16 | undefined at W160_L20 (0.161 on the other 15) | undefined | 0.203 |
| `thesis_x` | same, W160_L20 excluded | undefined at W160_L20 | undefined | 0.208 |
| `sequential` | thesis extraction chain re-run on scaled values | undefined at W160_L20 (0.147 on 15) | undefined | 0.147 |
| `sequential_x`, `sequential_k(_x)` | variants of the chain | undefined or not buildable | undefined | -- |
| **`thesis_g`** | **thesis procedure + global coefficient refinement** | **0.065** | **0.176** | **0.111** |
| `thesis_x_g` | same, W160_L20 excluded | 0.073 | 0.278 | 0.134 |
| `sequential_g` | sequential + refinement | 0.068 | 0.276 | 0.117 |

The final variant was chosen on leave-one-geometry-out CV over the fit
geometries alone. The unseen devices were not consulted.

Diagnosis against the six causes the task asked about:
1. **Bad curve selection**: addressed in Step 0. Using the late-turn-on
   bot2 site would have mixed a wafer-position effect into the W/L trends.
2. **Incorrect extraction**: no. The per-device floor is 0.02-0.12 dec.
   Two thesis-literal steps needed robustness fixes:
   - Ioff: averaging instead of a single noise sample
   - TLM: the RSD ≥ 0 constraint
3. **Outlier device**: W160_L20, a low-mobility die. Excluding it from the
   coefficient fit made every variant *worse* in CV (the `_x` rows): the
   W = 160 row then has only 3 lengths and the quadratic extrapolates
   badly. In the final model it is kept and reproduced to 0.03-0.08 dec.
4. **Incorrect scaling coefficient** and
5. **inappropriate fitting of the prescribed relation**: **this is the
   cause.** Step 2 showed α, κ and G0 trade off, so G0 alone spans 30x in a
   non-quadratic way. Fitted parameter by parameter in linear space (the
   thesis form for G0), its polynomial turns **negative** at the W160_L20
   corner and, in CV, elsewhere. A negative G0 makes
   `Ids_lin**(-m)` undefined. The thesis-literal model is therefore not
   usable over the full design space, although away from that corner it
   reaches 0.16 dec on the fit set and 0.20 on the unseen devices.
   - Re-running the thesis's own extraction chain on the scaled values
     (Von → α → κ → G0, the order of Sec. 3.2) removes much of the
     α/κ/G0 inconsistency (unseen error 0.203 → 0.147). It does not fix the
     sign problem.
   - The fix that works keeps every functional form and starts from the
     thesis-procedure coefficients. The Von/α/κ/G0 polynomial coefficients
     are then refined by least squares against the fit devices' measured
     currents (`refine.py`). Each parameter is kept near its per-device
     extracted value (penalty of 0.1 per standard deviation). The
     refinement also enforces G0 > 0, κ < 0 and α < 0 over the whole range
     W = 20-160 um, L = 5-20 um.
   - ΔL, RSD, Ioff and m keep their extracted values. This is the usual
     local-extraction-then-global-optimization flow for compact models.
     `plots/step7_scaling_fits.png` shows the refined surfaces running
     through the extracted parameter clouds.
6. **A genuine limitation of the CAMCAS scaling assumption**: partly.
   - The scaling forms themselves hold: channel conductance scales with W/L
     to ±7%.
   - The weak points: (i) splitting that conductance into (α, κ, G0) per
     device is ill-conditioned, so independent per-parameter polynomials
     need the refinement above; (ii) Appendix F's single m cannot follow
     the per-device m (1.9-2.1); (iii) Von is quantized to the 0.1 V sweep
     step.
   - More devices per geometry, and a finer VGS step near turn-on, would
     help more than any change of method.

**Unseen geometries.** W10_L5 and W10_L10 are predicted to
0.06-0.08 dec, about as well as the fit devices. W5_L5 is predicted to
0.18-0.19 dec: it sits at W = 5 um, 4x below the smallest fitted width, and
its subthreshold slope differs from the fitted trend. All three are
extrapolations in W. The leave-one-geometry-out CV (0.176 dec; worst folds
are the grid corners 160/5, 160/15 and 80/20) is the better estimate of
error at an unmeasured geometry *inside* the fitted range.

**On the held-out protocol, honestly.** The three unseen devices never
entered extraction-to-scaling or any coefficient. The final variant is
selected by an explicit rule on the fit set (leave-one-geometry-out CV).
During development, however, the unseen scores were printed next to the
fit-set scores, so they were *visible*. None of the following was tuned on
them: the variant list, the outlier handling, and the refinement penalty
(0.1, set once). The CV rule and the unseen scores happen to rank
`thesis_g` first.
