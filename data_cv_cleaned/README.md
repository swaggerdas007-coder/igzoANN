# data_cv_cleaned — CGD/CGS training dataset (VDS = 0 slice)

Built by `scripts/build_cv_dataset.py` from the 12 raw
`data/C-V [W-L kind(n) ; ...].csv` files (4 device geometries x
{cg, cgd, cgs}). Regenerate with:

```
python scripts/build_cv_dataset.py
```

## What's here

- `W{W}_L{L}_cv_clean.csv` — one file per geometry: `VG, VDS, W, L, CGD, CGS, CG`.
- `merged_cv_dataset.csv` — all 4 geometries concatenated, ready to train on.
- `cv_dataset_manifest.json` — geometry list, sweep range/step, frequency,
  AC level.

Measurement conditions (all 12 files identical): VG swept -3 to 5 V in
0.2 V steps, f = 10 kHz, AC level = 50 mV, single fixed bias of 0 V on the
non-swept terminal (source, for `cgd`; drain, for `cgs`; both, for `cg`).

## Important scope limitation: this is a VDS=0 model, not the paper's full surface

Bahubalindruni et al. characterize **CGSi(VGS, VDS)** and **CGDi(VGS, VDS)**
over a full VDS sweep (0-10 V) at several fixed VGS (their Fig. 5), which is
what lets their model capture the saturation-region asymmetry (CGDi -> 0,
CGSi -> 2/3 CCH). Our raw data does not contain that: the `cgd`/`cgs` files
here are single VG sweeps at a *fixed* VDS = 0 V bias (paper's Fig. 3(b)/(c)
setup, but only the one bias point). We do separately have a `C-V Vd=*`
family that sweeps VDS for the *combined* total Cg — see `data/` and the
main C-V comparison — but it isn't split into CGD/CGS, so it can't fill
this gap.

Consequently, any CGD/CGS ANN trained on this dataset is a function of
`(VG, W, L)` only — it has no VDS dependence and will not reproduce the
paper's saturation-region CGD-to-CGS asymmetry. It reproduces the paper's
*measurement scheme* (Fig. 3b/c, subtraction not even needed here since
CGD and CGS were each measured directly against a grounded reference
terminal) at the one bias point we have data for.

## Geometry coverage

Only 4 (W, L) points: (20,20), (40,20), (160,20), (160,15) um — versus 19
geometries for the ID dataset (`data_cleaned_2/`). Any W/L generalization
from the resulting ANN is far less supported than the ID model's. In fact
a single combined (VG, W, L) ANN could not even interpolate reliably
between these 4 points (see `outputs_cap/README.md`) — the shipped model
is 8 independent per-device 1-input (VG only) networks instead.

## Modeling outcome

See `outputs_cap/README.md`, `src/train_cap_per_device.py`, and
`verilogA/tft_ann_full_model.va` for how this dataset was actually used.
