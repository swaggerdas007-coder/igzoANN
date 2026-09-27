# CAMCAS-style semi-empirical compact model

An alternative to this repo's ANN-based models: the unified analytical
drain-current expression from Carolina de Almeida's thesis "Development of
IGZO thin-film transistors (TFTs) compact models" (NOVA FCT, 2025), Sec.
2.3.4 and Chapter 3, whose Verilog-A implementation is the thesis's Appendix
F. The thesis calls it CAMCAS ("Cambridge Compact Analytical Semiconductor")
and traces it to an earlier a-IGZO TFT modeling paper; it's semi-empirical,
not derived from first-principles device physics -- a small set of fitted
parameters (`Von`, `alpha`, `kappa`, `G0`, `Ioff`, described as trap-state
related) plug into one analytical expression that covers both linear and
saturation regimes at once, referenced to a turn-on voltage `Von` rather than
a classical threshold voltage:

```
IDS = G0 * (W/L') * exp(kappa * (VGS - Von)^alpha) * VDS' + Ioff
```

fit separately in the linear (`VDS' = VDS - 2*Rc*IDS`) and saturation
(`VDS' = VGS - Von`, Vds-independent) regimes and blended with a harmonic-mean
smoothing function so a single expression transitions between them at any
bias. See the thesis's Eqs. 2.27-2.35 and 3.1-3.18 for the full derivation.

## Recommended: the unified model (`verilogA/tft_camcas_unified.va`)

**One fixed parameter set for every geometry.** (W, L) enter only through
the scaling the CAMCAS equation already has: W/L' with L' = L - DeltaL, and
a width-normalized series resistance RSD = Rsw/W. There are no per-device
tables and no (L, W) polynomial surfaces.

### How it was built (`scripts/fit_camcas_unified.py`)

1. **Pick the golden devices.** On W/L-normalized curves (so geometry itself
   doesn't count against a device), keep the devices whose turn-on voltage
   is within +-0.75 V of the population median and whose on-current at
   VG = 5 V is within 2x of the median, in both the linear and saturation
   sweeps. 13 of 19 devices pass: W/L = 20/10, 20/15, 20/20, 40/5, 40/10,
   40/15, 80/5, 80/10, 80/15, 80/20, 160/5, 160/10, 160/20 um
   (`unified_device_selection.csv`).
   The 6 that fail are atypical die sites: W5/L5, W10/L5, W10/L10 and W20/L5
   turn on late (1.6-3.4 V vs ~0.6 V typical). W40/L20 turns on late *and*
   carries ~7x less normalized linear current than its neighbours, so it is
   the worst device in the set, not the best. W160/L15 is the one
   negative-VT fallback device (see `data_cleaned_2/README.md`). No W = 5 or
   10 um device is typical, so the model's predictions at those widths are
   "typical process" predictions that can't be checked against a typical
   measured device.
2. **DeltaL from a width-normalized TLM** over all golden devices at once:
   Rtot*W vs L at VG = 5 V gives DeltaL = -3.6 um, i.e. the effective
   channel is ~3.6 um *longer* than drawn. The TLM lines at different VG
   don't share one intersection, because their intercept falls as VG rises.
   So the access region is gate-controlled and acts like extra channel
   length rather than a fixed contact resistor. DeltaL, Rsw and Von trade
   off almost perfectly in the fit, so pinning DeltaL from an independent
   extraction keeps them physical.
3. **Reparametrize G0 and kappa.** In the thesis form
   G0*exp(kappa*x^alpha), G0 and kappa are nearly perfectly correlated: as
   alpha -> 0 only G0*e^kappa is determined. That degeneracy is what sent
   G0 to its search bound in ~40% of the per-device fits. The fit uses the
   identical curve written around the reference point VGS = VH = 5 V,
   G = GH*(W/L')*exp(s*((x/xH)^alpha - 1)/alpha). GH (conductance at 5 V)
   and s (log-log slope there) are each well determined on their own. The
   thesis-notation values are recovered exactly as
   kappa = s/(alpha*xH^alpha) and G0 = GH*exp(-kappa*xH^alpha).
4. **Fit end-to-end, jointly.** The full model (both branches plus the
   harmonic-mean blend) is fit at every measured bias point of the linear
   transfer, saturation transfer and output-curve sweeps of all golden
   devices at once, in log10|ID| (the ANN's target space). Each
   (device, sweep) gets equal total weight, the loss is soft-L1, and the
   fit uses 12 multi-start initial guesses. A channel-length-modulation
   term was tried and fit to exactly zero, so it was dropped.

### Parameters

| | linear branch | saturation branch |
|---|---|---|
| Von (V) | -0.485 | -2.203 |
| alpha | -0.643 | -3.130 |
| kappa (thesis form) | -6.96 | -229.1 |
| G0 (thesis form, S) | 6.59e-5 | 2.32e-6 |
| GH = G/(W/L') at VG = 5 V (S) | 6.4e-6 | 1.45e-6 |
| s (log-log slope at 5 V) | 1.50 | 1.48 |

Shared: DeltaL = -3.60 um, Rsw = 4.07e5 Ohm*um (RSD = 10 kOhm at
W = 40 um), m = 2.85, Ioff = 2.7e-12 A. The saturation branch's Von sits
just below the measured VG range. It is a curve-shape parameter there, not
a physical turn-on voltage; the device's real turn-on is set by where the
steep alpha = -3.1 exponential lifts off, at ~0 V.

### Accuracy vs. this repo's ANN

Same data points, same metrics (`scripts/evaluate_camcas_unified.py` ->
`unified_vs_ann_summary.csv`, `unified_vs_ann_per_geometry.csv`). The ANN is
`outputs/model_weights.pt`, trained on all 19 devices.

| log10\|ID\| MAE (decades) / R^2 | golden 13 devices | atypical 6 devices | all 19 | ANN's own test split |
|---|---|---|---|---|
| **CAMCAS unified** | **0.277 / 0.975** | 1.084 / 0.538 | 0.532 / 0.864 | 0.539 / 0.865 |
| ANN | 0.404 / 0.966 | **0.614 / 0.898** | **0.470 / 0.949** | **0.466 / 0.951** |
| CAMCAS, thesis-style (L,W) surface | 2.040 / 0.144 | 1.482 / 0.238 | 1.864 / 0.188 | 1.812 / 0.230 |

On-state median relative error on the golden devices: CAMCAS unified 24%,
ANN 49%.

- **On typical devices the unified CAMCAS model beats the ANN**: lower
  error on 11 of 13 golden devices, a tie on W80/L20, and the ANN ahead only
  on W20/L10 (`plots/unified_vs_ann_per_geometry.png`).
- **It generalizes to unseen geometries.** Leave-one-device-out CV (refit
  without device k, predict device k) gives a mean held-out error of 0.290
  decades vs 0.277 in-sample (`unified_loo_cv.csv`). That is still better
  than the ANN's 0.404 on the same devices, and the ANN had seen them in
  training. This is the "one formula for any (L, W)" property the
  thesis-style surface couldn't deliver on this data.
- **Output curves are physically well behaved**: smooth knee, flat
  saturation, correct VG ordering (`plots/unified_output_curves.png`). The
  ANN's output curves show a non-physical kink near VD ~ 0.2 V and can
  cross each other.
- **The ANN wins overall only because of the 6 atypical devices.** It was
  trained on them and partly memorizes each one. The unified model
  deliberately doesn't chase them: one parameter set can't reproduce a
  device that turns on 2 V late without breaking every typical device.
  If your circuits need those specific devices, use the ANN. If they need a
  typical transistor of arbitrary (W, L), use this.
- The remaining ~20-30% misses at VG = 5 V go in different directions on
  different devices (W80/L5 under, W160/L20 over), which is ordinary
  device-to-device scatter rather than a model-shape error.

### Verilog-A

`scripts/export_verilog_a_camcas_unified.py` writes
`verilogA/tft_camcas_unified.va`: instance parameters `w`, `l` (m), plus the
fitted values as overridable module parameters. It compiles with OpenVAF
(ngspice/Xyce via OSDI) as well as Spectre; it has no `initial_step`
block, which OpenVAF rejects. `--check` compiles it with `verilogae`
(OpenVAF) and compares it against the Python model on all 13,148 measured
bias points with VD >= 0.1 V: max relative difference 9e-5, all of it from
the zero-bias Ioff term below. Two robustness additions don't touch any
measured bias point: a source/drain swap for VDS < 0, and Ioff scaled by
tanh(VDS/20 mV) so no current flows at VDS = 0. Checked: continuous
through VDS = 0 with the correct sign under reverse bias.

Regenerate: `fit_camcas_unified.py` -> `evaluate_camcas_unified.py` ->
`export_verilog_a_camcas_unified.py --check`.

---

## Earlier approaches (kept for reference)

The sections below document the first adaptation: per-device extraction,
the thesis's (L, W) polynomial surfaces, and the nearest-match
`verilogA/tft_camcas_model.va` that the unified model supersedes.

### Files

- `scripts/fit_camcas_model.py` -- extracts (Von, Ioff) per geometry
  (derivative-onset method), (DeltaL, RSD) per width (bounded TLM), and
  (alpha, kappa, G0) per geometry per regime (direct nonlinear fit to Eq.
  2.27 in log-current space), then fits each of the 10 regime parameters as
  a ridge-regularized bivariate polynomial surface over (L, W).
  -> `extracted_params.csv`, `dl_rsd_per_width.csv`, `camcas_coefficients.json`
- `scripts/plot_camcas_fit.py` -- validates both the scalable surface and
  the raw per-device ("oracle") fits against the measured curves.
  -> `plots/camcas_vs_measured_all_geometries.png`, `camcas_relative_error.csv`
- `scripts/export_verilog_a_camcas.py` -- writes `verilogA/tft_camcas_model.va`.

Regenerate in that order: `fit_camcas_model.py` -> `plot_camcas_fit.py` ->
`export_verilog_a_camcas.py`.

### Data

`data_cleaned_2/` -- one QC-passed, positive-turn-on-voltage device per (W,
L) geometry, 19 geometries (W in {5,10,20,40,80,160} um, L in {5,10,15,20}
um). This is a much richer geometry grid than the thesis's own best dataset
(Dataset 3: 2 widths x 3 lengths = 6 points), but each geometry here is a
*different physical device* (a different die site on the wafer), not a
repeat measurement of the same device -- important below.

### What works: the per-device fit

Fit directly to one device's measured curve (no cross-geometry
generalization), the CAMCAS expression tracks the data to a median relative
error of **~33% (linear regime) / ~18% (saturation regime)** across the 11
geometries where the fit converges to a physically sane optimum (see
below) -- comparable to what a semi-empirical model like this should
achieve, and consistent with the thesis's own finding that error is largest
near turn-on and shrinks at higher VGS.

3 of the per-geometry curve-fits (in either regime) land on a degenerate
optimum instead: `kappa` and `G0` run away together (G0 hitting its search
bound, ~5e8) because that one noisy curve doesn't have enough usable
on-state samples/curvature to pin down all three of (alpha, kappa, G0)
independently. These are excluded (via a plain sanity bound: `0 < G0 < 1`
in the fit's own natural units) from both the "oracle" accuracy numbers
above and the scalable surface fit below -- the thesis hits the identical
issue on its own noisier datasets and drops individual devices by hand
(Sec. 3.3.2).

### What doesn't: the scalable (L, W) surface

The thesis's whole point is a *compact* model -- one formula usable at any
(L, W), not just the measured points -- built by re-fitting each of the 10
regime parameters as a smooth function of (L, W) (thesis Eqs. 3.5-3.18,
literally the six-coefficient polynomials baked into Appendix F's
`von_lin_func` etc.).

That generalization does **not** hold up on our data. Leave-one-out
cross-validated ridge regression picks heavy-to-total shrinkage for 8 of
the 10 parameters (`alpha`, `kappa`, `G0`, `Ioff` in both regimes) --
meaning the CV procedure itself concludes there is no reliable (L, W) trend
to fit, only noise. Only `Von_lin(L, W)` shows real geometric structure
(R^2 = 0.68). Plugging the fitted surface back in reproduces even its own
training geometries very poorly (~100% median error) -- this isn't a
modeling nicety, it's a sign the surface is not usable.

**Why**: unlike the thesis's Dataset 3, our 19 geometries are 19 different
physical devices (die sites), not repeat measurements of a controlled set.
CAMCAS's exponential term is extremely sensitive to small (alpha, kappa)
differences, so ordinary device-to-device threshold/mobility variation
(which `data_cleaned_2`'s own README already documents as a wafer-level
effect motivating its one-device-per-geometry selection) dominates over any
clean channel-geometry trend for most of these parameters. The thesis
reports the same failure mode for its own noisier Dataset 2 (Sec. 3.3.2:
"the extracted model parameters do not exhibit a consistent or reliable
dependence on the geometrical dimensions").

### Nearest-match per-device model (superseded by the unified model)

`verilogA/tft_camcas_model.va` therefore uses **nearest-match per-device
parameters** -- the 11 geometries with a converged fit, selected by (w, l)
distance once at `initial_step` (zero runtime cost) -- instead of the
unreliable smooth surface. This is the same pattern `tft_ann_full_model.va`
already uses for CGD/CGS for the identical reason (see
`scripts/export_verilog_a_full.py`'s docstring).

Accuracy (median relative error, on-state):

| | linear | saturation |
|---|---|---|
| exact geometry match (11/19) | ~33% | ~18% |
| nearest-neighbor snap (8/19) | ~74%, occasionally 10-30x off | ~76%, occasionally 3-5x off |

The snap case is worst for W=10 um and W=5 um, which have no converged
reference device at all and snap to a fairly distant W=20 um neighbor. If
you need those geometries to be reliable, the fix is more/better-behaved
raw measurements at those widths, not a different fitting method -- see
"What doesn't" above.

### Comparison to the ANN (per-device / nearest-match version)

Not a fully apples-to-apples comparison (different error metrics), but for
context: the unified ANN (`outputs/metrics.json`) reaches test R^2 = 0.951
and MAE = 0.47 decades in log10(ID) *across all 19 geometries directly*,
with no separate geometry-scaling step to fail. CAMCAS's per-device fit is
competitive in absolute accuracy at the geometries it fits well, but the
ANN sidesteps the scalability problem entirely by learning geometry
dependence directly from all the data at once, rather than fitting 10
independent parameters per device and then trying to re-fit *those* against
(L, W) afterward.

### Two real bugs worth knowing about if you extend this

1. **Vds hardcoded in the linear branch.** The linear-regime branch must use
   the *actual* instance Vds (thesis/Appendix F: `Ids_lin = (G*Vds +
   Ioff)/(1+RSD*G)` with `Vds = V(d,s)`), not the 0.1V reference bias the
   linear-regime curves happen to be measured at. Get this wrong and the
   harmonic-mean blend can never hand off to the (Vds-independent)
   saturation branch as Vds grows, so the whole model stays pinned near its
   low-Vds value and never saturates -- this alone was worth ~75 percentage
   points of median relative error in the saturation regime during
   development.
2. **Don't fit decade-spanning currents (or G0/Ioff surfaces) in linear
   space.** IDS spans up to ~6 decades over the on-state samples of a single
   curve; an ordinary-scale nonlinear or linear least-squares fit is
   dominated entirely by the largest values and leaves the optimizer with
   essentially no gradient at low current -- confirmed to get every
   geometry's fit stuck at its initial guess. Fit `ln(I)` / `ln(G0)` /
   `ln(Ioff)` instead, exactly like this repo's ANN regresses
   `log10(|ID|)` rather than raw `ID` (`src/dataset.py`).
