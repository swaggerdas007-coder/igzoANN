# trained_Cg_ANN — two-layer Cgd / Cgs capacitance ANNs

Separate two-hidden-layer ANNs for the gate–drain (`cgd/`) and gate–source
(`cgs/`) parasitic capacitances, trained by `src/train_cg_final.py`.

Reproduce:
```
python scripts/clean_cv_curves.py       # if data_cleaned_cv/ not present
python -m src.cg_loss_experiments       # loss/target variant comparison
python -m src.train_cg_final            # trains the selected variant
```

## Architecture

```
CGD: (VG, VD, W, L) ─► Linear(4→18) ─tanh─► Linear(18→9)  ─tanh─► Linear(9→1)  ─► C/(W·L)
CGS: (VG, VD, W, L) ─► Linear(4→20) ─tanh─► Linear(20→10) ─tanh─► Linear(10→1) ─► C/(W·L)
```

- inputs min-max scaled with the same bounds as the `I_D` ANN, so all three
  equivalent-circuit ANNs are pin-compatible;
- **target = C / (W·L) in pF/µm², ×10³**, standardised with train-set mean/std;
  the capacitance is recovered as `C_pF = (out·y_std + y_mean)·W·L/1e3`;
- tanh hidden layers, linear output — directly portable to Verilog-A;
- fixed 2500-epoch budget, Adam + cosine-annealed LR, seed 42
  (`src/model.py::TFTNet2`).

Hyperparameters came from a 432-config sweep (8 architectures × {MSE, MAE,
Huber} × 3 learning rates × 3 batch sizes, capped at 20 neurons in layer 1 and
10 in layer 2): CGD = Huber, lr 5e-3, batch 32; CGS = MSE, lr 5e-3, batch 16.

Each component folder holds `model_weights.pt`, `weights.json`
(`w1,b1,w2,b2,wo,bo` + scaling + target transform, consumed by
`scripts/export_verilog_a_full.py`), `metrics.json`, `test_predictions.npz`
and `plots/`.

## Test-set metrics (held-out operating points)

| Component | R²     | RMSE (pF) | MAE (pF) | MARE on-state (C ≥ 0.5 pF) |
|-----------|--------|-----------|----------|-----------------------------|
| Cgd       | 0.9946 | 0.138     | 0.054    | 4.9 %                       |
| Cgs       | 0.9905 | 0.181     | 0.064    | 5.2 %                       |

## Why the target is normalised by gate area

The earlier models regressed absolute capacitance in pF. Their pooled metrics
looked acceptable (Cgd R²=0.855, Cgs R²=0.819; after hyperparameter tuning
0.966 / 0.946) but the C–VG curves overshot the turn-on knee of the small
devices and dipped slightly below 0 pF at very negative VG. Neither a third
hidden layer nor log-scaled geometry inputs helped, because the cause is the
loss itself: the W=160 curves are ~8× larger in magnitude, so an unweighted
loss on absolute pF spends its capacity on them.

The measured capacitance is very nearly proportional to gate area —
C/(W·L) ≈ 2.0e-3 pF/µm² for all four devices — so dividing the target by W·L
puts every curve on the same scale and removes the imbalance at the source.

`src/cg_loss_experiments.py` compares four options over 3 seeds each
(`cg_experiments/variant_summary.json`). NRMSE is per geometry over all
measured points, normalised by that geometry's capacitance range:

**CGD**

| variant     | test R² | test RMSE (pF) | W20-L20 | W40-L20 | W160-L15 | W160-L20 |
|-------------|---------|----------------|---------|---------|----------|----------|
| baseline    | 0.9485  | 0.418          | 9.31 %  | 10.41 % | 1.84 %   | 5.69 %   |
| weighted    | 0.8972  | 0.592          | 6.73 %  | 7.54 %  | 5.78 %   | 9.47 %   |
| **areanorm**| **0.9927** | **0.159**   | **3.28 %** | **7.54 %** | **0.61 %** | **2.14 %** |
| areanorm_w  | 0.9834  | 0.238          | 4.16 %  | 6.36 %  | 1.51 %   | 4.30 %   |

**CGS**

| variant     | test R² | test RMSE (pF) | W20-L20 | W40-L20 | W160-L15 | W160-L20 |
|-------------|---------|----------------|---------|---------|----------|----------|
| baseline    | 0.8900  | 0.604          | 10.26 % | 8.06 %  | 2.98 %   | 8.29 %   |
| weighted    | 0.8406  | 0.741          | 6.89 %  | 5.81 %  | 4.63 %   | 10.62 %  |
| **areanorm**| **0.9893** | **0.191**   | **1.60 %** | **2.70 %** | **1.13 %** | **2.53 %** |
| areanorm_w  | 0.9731  | 0.300          | 2.88 %  | 2.64 %  | 1.96 %   | 4.52 %   |

Area normalisation wins on every metric for both components, and it also
removes the negative-capacitance excursions (minimum prediction goes from
−0.105 pF to +0.093 pF for CGS), so no positivity head is needed.

Per-curve loss weighting on the un-normalised target (`weighted`) helps the
small devices but costs more on the large ones than it gains — it trades the
imbalance rather than removing it.

## Per-geometry fit of the final models (all measured points)

| Geometry (W–L µm) | Cgd NRMSE | Cgd RMSE (pF) | Cgs NRMSE | Cgs RMSE (pF) |
|-------------------|-----------|---------------|-----------|---------------|
| 20-20             | 3.60 %    | 0.025         | 1.57 %    | 0.011         |
| 40-20             | 7.44 %    | 0.103         | 2.72 %    | 0.039         |
| 160-15            | 0.70 %    | 0.029         | 1.11 %    | 0.047         |
| 160-20            | 1.79 %    | 0.097         | 2.40 %    | 0.134         |

The remaining weak spot is Cgd on W=40, L=20, where the model still lags the
measured rise just above VG = 0.

## Known limits

- **No VDS dependence.** Every C–V measurement is at VDS = 0 V, so VD is a
  constant across the dataset; the networks accept VD only for pin-compatibility
  with the I_D ANN. The paper's CGSi/CGDi model does depend on VDS and captures
  the saturation-region asymmetry (CGDi → 0). Closing this gap needs
  VD-resolved C–V data.
- **Four geometries.** (20,20), (40,20), (160,15), (160,20) µm. The area
  normalisation gives a physically sensible extrapolation law, but any (W,L)
  outside that hull is still unvalidated.

See `docs/CAPACITANCE_ANN.md` for the full design history.
