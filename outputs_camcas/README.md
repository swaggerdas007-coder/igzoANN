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

## Files

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

## Data

`data_cleaned_2/` -- one QC-passed, positive-turn-on-voltage device per (W,
L) geometry, 19 geometries (W in {5,10,20,40,80,160} um, L in {5,10,15,20}
um). This is a much richer geometry grid than the thesis's own best dataset
(Dataset 3: 2 widths x 3 lengths = 6 points), but each geometry here is a
*different physical device* (a different die site on the wafer), not a
repeat measurement of the same device -- important below.

## What works: the per-device fit

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

## What doesn't: the scalable (L, W) surface

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

## What's shipped: nearest-match, like this repo's own CGD/CGS precedent

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

## Comparison to this repo's ANN model

Not a fully apples-to-apples comparison (different error metrics), but for
context: the unified ANN (`outputs/metrics.json`) reaches test R^2 = 0.951
and MAE = 0.47 decades in log10(ID) *across all 19 geometries directly*,
with no separate geometry-scaling step to fail. CAMCAS's per-device fit is
competitive in absolute accuracy at the geometries it fits well, but the
ANN sidesteps the scalability problem entirely by learning geometry
dependence directly from all the data at once, rather than fitting 10
independent parameters per device and then trying to re-fit *those* against
(L, W) afterward.

## Two real bugs worth knowing about if you extend this

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
