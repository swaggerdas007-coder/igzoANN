# Parasitic-capacitance ANNs (Cgd, Cgs) — baseline

The behavioral model of Bahubalindruni et al. (2016) represents the a-IGZO TFT
as an equivalent circuit built from **three** ANNs joined in Verilog-A: one for
the drain current `I_D`, and one each for the gate–drain and gate–source
parasitic capacitances `C_GD` and `C_GS`. This repo already trained the `I_D`
ANN (`src/train.py`). This note covers the two capacitance ANNs added here.

## Is there enough data? — Yes, for a baseline

| Component | Rows | Geometries (W–L µm)           | Inputs varied | Held fixed        |
|-----------|------|-------------------------------|---------------|-------------------|
| C_GD      | 164  | 20-20, 40-20, 160-15, 160-20  | VG (−3…5 V)   | VD = 0 V, 10 kHz  |
| C_GS      | 164  | same                          | VG (−3…5 V)   | VD = 0 V, 10 kHz  |

The 8 C–V files (`data/C-V [W-L cg{d,s}(..)].csv`) are clean, monotonic
accumulation sweeps with pF-scale capacitance and no noise-floor issues. That
is enough to fit a first `C(VG, W, L)` model per component — a baseline to
iterate on, exactly as requested.

**What it is not enough for (yet):**

- **VD dependence.** Every individual C_GD/C_GS sweep is at VD = 0 V, so the
  model cannot learn the C-vs-VDS dependence that the paper's Fig. 6 shows. The
  VD-swept files in `data/` measure *total* Cg only and can't be split into
  components without per-VD Fig. 3(c) subtraction data.
- **Geometry extrapolation.** Only 4 devices (3 widths, 2 lengths), one
  instance each. Fits interpolate within the measured geometries.

## Pipeline

```
data/  ──clean_cv_curves.py──►  data_cleaned_cv/merged_cap_dataset.csv
                                          │
                                   src/train_cap.py  (one ANN per component)
                                          │
                    outputs/cap_cgd/ , outputs/cap_cgs/
                      ├─ model_weights.pt      (PyTorch state dict)
                      ├─ weights.json          (wh,bh,wo,bo + scaling, for Verilog-A)
                      ├─ metrics.json
                      ├─ test_predictions.npz
                      └─ plots/                (src/evaluate_cap.py)
```

Reproduce:

```
python scripts/clean_cv_curves.py
python -m src.train_cap --cap both
python -m src.evaluate_cap
```

## Model

Same architecture as the `I_D` ANN (`src/model.py::TFTNet`) so all three
elements are pin-compatible for the equivalent circuit:

- inputs `(VG, VD, W, L)`, min-max scaled to `[0,1]` with the **same** bounds
  as the `I_D` model (`src/dataset.py::FEATURE_BOUNDS`);
- one hidden layer, 32 `tanh` neurons; linear output;
- target = capacitance in **pF** (smooth, positive, < 2 decades — regressed
  directly, unlike `log10|ID|` for the current), standardized with train-set
  mean/std.

Trained for a fixed epoch budget with cosine-annealed LR. Early stopping on the
25-point validation set was tried and **removed**: the set is too small to be a
reliable signal and best-val models are badly under-trained (they predict a
near-linear ramp instead of the sigmoidal floor→rise→plateau shape).

## Baseline results (held-out test set)

| Component | R²    | RMSE (pF) | MAE (pF) | MARE on-state (C ≥ 0.5 pF) |
|-----------|-------|-----------|----------|-----------------------------|
| C_GD      | 0.73  | 0.97      | 0.57     | 57 %                        |
| C_GS      | 0.66  | 1.08      | 0.65     | 75 %                        |

See `outputs/cap_{cgd,cgs}/plots/`:
`C_vg_curves.png` (ANN vs measured, per geometry) and `scatter_C.png`.

**Reading the numbers.** The ANN captures the overall magnitude and its scaling
with geometry (on-state capacitance scales ≈ linearly with W·L in the data),
but the measured C–VG curve is a near **step** — a flat sub-threshold floor, an
abrupt jump at threshold (VG ≈ 0.3–0.7 V), then a flat plateau — and a smooth
MLP averaged over 4 geometries rounds that step into a ramp. That mismatch
around threshold, plus a small unphysical dip below 0 pF at very negative VG,
is what caps R² near 0.7 and inflates the relative error (the on-state MARE is
dominated by the steep-transition points where a small VG offset is a large
capacitance error). The full-range MARE is higher still only because of the
~0.1 pF sub-threshold points and is not a meaningful accuracy figure here.

## Deeper variant: two-layer ANNs (`trained_Cg_ANN/`)

`src/train_cg_ann.py` trains a deeper `4 → 10 → 10 → 1` MLP
(`src/model.py::TFTNet2`) per component. The extra hidden layer lets the
network bend into the sharp threshold step, lifting the test metrics over the
single-layer baseline:

| Component | R² (2-layer) | R² (1-layer 32) |
|-----------|--------------|------------------|
| Cgd       | 0.855        | 0.73             |
| Cgs       | 0.819        | 0.66             |

Per-geometry, that fit is excellent on the two W=160 devices (R² ≈ 0.94–0.99)
but only fair on W=20–40.

## What did and did not fix the small-device fit

Four things were tried against the W=20/W=40 overshoot, in order:

| Attempt | Result |
|---------|--------|
| Third hidden layer (10→5→5), `src/train_cg_ann_3layer.py` | **Worse** — Cgd R² 0.855 → 0.839, Cgs 0.819 → 0.810 |
| Log-scaled W and L inputs | **Worse** — Cgd 0.855 → 0.822, Cgs 0.819 → 0.760 |
| Hyperparameter sweep, 432 configs (8 architectures ≤ 20/10 neurons × {MSE, MAE, Huber} × 3 LRs × 3 batch sizes), `src/tune_cg_ann.py` | **Better** — Cgd 0.966 (18→9, Huber, lr 5e-3, bs 32), Cgs 0.946 (20→10, MSE, lr 5e-3, bs 16); small-device overshoot reduced but not removed |
| **Area-normalised target C/(W·L)**, `src/cg_loss_experiments.py` | **Fixes it** — Cgd 0.9946, Cgs 0.9905; see below |

Depth and input scaling could not help because the problem was never the model
or the input geometry — it was the loss. An unweighted loss on absolute pF is
dominated by the W=160 curves, which are ~8× larger in magnitude, so the fit is
spent on them.

The measured capacitance is very nearly proportional to gate area
(C/(W·L) ≈ 2.0e-3 pF/µm² for all four devices), so regressing C/(W·L) puts
every curve on the same scale and removes the imbalance at the source. It also
eliminates the negative-capacitance dip at very negative VG, so no softplus
output head is needed. Per-curve loss weighting on the un-normalised target was
also tried and merely trades the imbalance rather than removing it.

`src/cg_loss_experiments.py` compares all four options over 3 seeds; results in
`cg_experiments/variant_summary.json` and `trained_Cg_ANN/README.md`.

Final models (`src/train_cg_final.py`):

| Component | Architecture | Loss | R²     | RMSE (pF) | MARE on-state |
|-----------|--------------|------|--------|-----------|----------------|
| Cgd       | 4→18→9→1     | Huber| 0.9946 | 0.138     | 4.9 %          |
| Cgs       | 4→20→10→1    | MSE  | 0.9905 | 0.181     | 5.2 %          |

## Verilog-A equivalent circuit

All three ANNs are merged into `verilogA/tft_ann_full_model.va` by
`scripts/export_verilog_a_full.py`, following Fig. 1(b) of the paper — ID as a
current source across D–S, CGD and CGS as bias-dependent capacitors across G–D
and G–S, each driven by its own network. The ID block is the two-hidden-layer
`4 → 32 → 16 → 1` model from `trained_ANN/deep/`. All three networks share the
same 4-input interface and min-max scaling, so the merge needs no adapters.
`scripts/verify_verilog_a_full.py` checks the generated file numerically
against the PyTorch models. See `verilogA/README.md`.

## Next steps

1. **Measure VD dependence** of C_GD/C_GS (component sweeps at several VDS), so
   the ANN can reproduce Fig. 6 — the single biggest remaining gap. Until then
   VD is an input in name only.
2. **More geometries / replicates** to support W/L generalisation. Area
   normalisation gives a physically sensible extrapolation law, but any (W,L)
   outside the 4 measured points is unvalidated.
3. Optionally regress the **intrinsic** C_GDi/C_GSi (measured minus overlap) if
   the EC needs the intrinsic components separately.
