# trained_Cg_ANN — two-layer Cgd / Cgs capacitance ANNs

Separate two-hidden-layer ANNs for the gate–drain (`cgd/`) and gate–source
(`cgs/`) parasitic capacitances, trained by `src/train_cg_ann.py`.

Reproduce:
```
python scripts/clean_cv_curves.py      # if data_cleaned_cv/ not present
python -m src.train_cg_ann
```

## Architecture (both components)

```
(VG, VD, W, L) ──► Linear(4→10) ─tanh─► Linear(10→10) ─tanh─► Linear(10→1) ──► C (pF)
```

- inputs min-max scaled with the same bounds as the `I_D` ANN;
- target = capacitance in pF, standardized with train-set mean/std;
- fixed 2500-epoch budget, Adam + cosine-annealed LR, batch 32, seed 42
  (`src/model.py::TFTNet2`).

Each component folder holds `model_weights.pt`, `weights.json`
(`w1,b1,w2,b2,wo,bo` + scaling, for Verilog-A), `metrics.json`,
`test_predictions.npz` and `plots/`.

## Test-set metrics (held-out operating points)

| Component | R²    | RMSE (pF) | MAE (pF) | MARE on-state (C ≥ 0.5 pF) |
|-----------|-------|-----------|----------|-----------------------------|
| Cgd       | 0.855 | 0.713     | 0.439    | 45.9 %                      |
| Cgs       | 0.819 | 0.791     | 0.436    | 48.4 %                      |

This is a clear gain over the single-layer 32-neuron baseline
(`outputs/cap_*`: Cgd R²=0.73, Cgs R²=0.66) — the extra layer lets the network
bend into the sharp threshold step of the C–VG curves.

## Do the models follow the measured data? (see `*/plots/C_vg_curves.png`)

**Yes for the large devices, only partly for the small ones.** Per-geometry fit
over all measured points:

| Geometry (W–L µm) | Cgd R² | Cgs R² |
|-------------------|--------|--------|
| 160-15            | 0.986  | 0.981  |
| 160-20            | 0.948  | 0.940  |
| 40-20             | 0.688  | 0.843  |
| 20-20             | 0.378  | 0.803  |

The ANN tracks the sharp floor→step→plateau shape almost perfectly on the two
W=160 devices, but overshoots / wiggles on W=20–40 (and dips slightly below
0 pF at very negative VG). The reason is the unweighted MSE on absolute pF: the
W=160 curves are ~8× larger, so they dominate the loss and the fit is spent on
them. The pooled R² above is therefore driven mostly by the W=160 curves.

To fix this (next iteration): weight the loss per curve / normalise capacitance
by device area (C/(W·L)) so all geometries contribute equally, add a positive
(softplus) output head to kill the negative dip, and — the biggest gap — obtain
VD-resolved measurements so the model can also learn C-vs-VDS. See
`docs/CAPACITANCE_ANN.md`.
