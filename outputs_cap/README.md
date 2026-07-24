# outputs_cap — CGD/CGS ANN artifacts

Two training approaches were tried on `data_cv_cleaned/merged_cv_dataset.csv`;
only the second is used by `verilogA/tft_ann_full_model.va`.

## Combined (VG, W, L) model — `weights_cgd.json`, `weights_cgs.json`, `plots/cgd_cgs_ann_fit.png`

Trained by `src/train_cap.py`: one MLP per target, inputs `(VG, W, L)`,
mirroring the ID model's architecture. With only 4 measured (W, L) points,
the network cannot learn where each device's turn-on knee sits and visibly
misplaces it (`plots/cgd_cgs_ann_fit.png`, test R^2 ~0.63-0.73). Kept here
as the documented reason for the approach below, not as a usable model.

## Per-device model — `per_device/` — **this is the one the Verilog-A file uses**

Trained by `src/train_cap_per_device.py`: one tiny MLP per (W, L) geometry
per target (8 total), input VG only, fit to all of that device's points
(no train/val/test split -- see that script's docstring for why: there is
exactly one measured device per geometry, so no cross-geometry
generalization claim is possible or attempted either way). Fit quality
(`plots/cgd_cgs_ann_fit_per_device.png`): R^2 = 0.99-1.00, MARE 4-7% on
all 4 devices, both targets (`per_device/metrics.json`).

`verilogA/tft_ann_full_model.va` embeds all 8 per-device subnetworks and
selects the nearest-(W,L)-match one at `initial_step`, the same pattern
`tft_ann_static_per_L.va` uses for L. See `data_cv_cleaned/README.md` for
what this model does and does not capture (no VDS dependence; only the 4
measured geometries, by nearest match, not continuous interpolation).
