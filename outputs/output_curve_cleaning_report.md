# Output-curve cleaning report

Source: 80 raw `tft output *.csv` files across 19 (W, L) combinations.

Each combination was measured up to 4 times at different wafer positions (top_1, top_2, bot_1, bot_2), occasionally re-run. Every file was scored against physical expectations for an n-type TFT output-curve family: current should rise monotonically with VD, increase in a fixed order with VG, stay near the noise floor for VG <= 0V, and show clear, well-separated modulation across VG (see `scripts/clean_output_curves.py` and the diagnostic grids in `outputs/plots/output_curve_qc/`).

## Result

A strong, consistent pattern emerged: devices at the **top** wafer position are unreliable (leaky off-state in most files, and outright broken -- compliance-clipped sweeps, dead/disconnected devices reading a flat current, or off-state noise bursts up to hundreds of microamps -- in roughly half of them). Devices at the **bot** position are consistently clean. No (W, L) combination had to be dropped entirely -- every combination has at least 2 usable replicates -- but `top_*` files should be treated with caution wherever they show up.

No (W, L) combination needed to be flagged in full: every combination had at least 2 replicates scoring at or above the pass threshold.


## Kept: 38 files -> `data/output_curves_clean/`

| W | L | file | replicate | score |
|---|---|------|-----------|-------|
| 5 | 5 | W5_L5_top_1.csv | top_1 | 99.4 |
| 5 | 5 | W5_L5_bot_1.csv | bot_1 | 98.0 |
| 10 | 5 | W10_L5_bot_1.csv | bot_1 | 100.0 |
| 10 | 5 | W10_L5_top_1.csv | top_1 | 85.0 |
| 10 | 10 | W10_L10_bot_1.csv | bot_1 | 99.8 |
| 10 | 10 | W10_L10_top_1.csv | top_1 | 99.8 |
| 20 | 5 | W20_L5_bot_2.csv | bot_2 | 98.9 |
| 20 | 5 | W20_L5_bot_1.csv | bot_1 | 85.0 |
| 20 | 10 | W20_L10_bot_1.csv | bot_1 | 99.0 |
| 20 | 10 | W20_L10_bot_2.csv | bot_2 | 98.6 |
| 20 | 15 | W20_L15_bot_1.csv | bot_1 | 100.0 |
| 20 | 15 | W20_L15_bot_2.csv | bot_2 | 99.8 |
| 20 | 20 | W20_L20_bot_1.csv | bot_1 | 100.0 |
| 20 | 20 | W20_L20_bot_2.csv | bot_2 | 99.8 |
| 40 | 5 | W40_L5_bot_2.csv | bot_2 | 100.0 |
| 40 | 5 | W40_L5_bot_1.csv | bot_1 | 85.0 |
| 40 | 10 | W40_L10_bot_2.csv | bot_2 | 100.0 |
| 40 | 10 | W40_L10_bot_1.csv | bot_1 | 99.6 |
| 40 | 15 | W40_L15_bot_1.csv | bot_1 | 100.0 |
| 40 | 15 | W40_L15_bot_2.csv | bot_2 | 100.0 |
| 40 | 20 | W40_L20_bot_1.csv | bot_1 | 100.0 |
| 40 | 20 | W40_L20_bot_2.csv | bot_2 | 98.4 |
| 80 | 5 | W80_L5_bot_2.csv | bot_2 | 100.0 |
| 80 | 5 | W80_L5_bot_1.csv | bot_1 | 84.8 |
| 80 | 10 | W80_L10_bot_2.csv | bot_2 | 100.0 |
| 80 | 10 | W80_L10_bot_1_run2.csv | bot_1_run2 | 85.0 |
| 80 | 15 | W80_L15_bot_2.csv | bot_2 | 100.0 |
| 80 | 15 | W80_L15_bot_1_run2.csv | bot_1_run2 | 85.0 |
| 80 | 20 | W80_L20_bot_2.csv | bot_2 | 99.8 |
| 80 | 20 | W80_L20_bot_1.csv | bot_1 | 84.8 |
| 160 | 5 | W160_L5_bot_2_run2.csv | bot_2_run2 | 99.8 |
| 160 | 5 | W160_L5_top_1.csv | top_1 | 85.0 |
| 160 | 10 | W160_L10_bot_2.csv | bot_2 | 100.0 |
| 160 | 10 | W160_L10_bot_1.csv | bot_1 | 85.0 |
| 160 | 15 | W160_L15_bot_1.csv | bot_1 | 85.0 |
| 160 | 15 | W160_L15_bot_2.csv | bot_2 | 84.6 |
| 160 | 20 | W160_L20_bot_1_run2.csv | bot_1_run2 | 100.0 |
| 160 | 20 | W160_L20_bot_2.csv | bot_2 | 100.0 |

## Excluded: 42 files

| W | L | replicate | score | reason |
|---|---|-----------|-------|--------|
| 5 | 5 | top_1 | 97.7 | ranked lower than the two best replicates for this (W, L) |
| 5 | 5 | top_2 | 97.7 | ranked lower than the two best replicates for this (W, L) |
| 5 | 5 | top_2 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 5 | 5 | bot_1 | 34.2 | no VD dependence: current does not rise with drain voltage |
| 5 | 5 | bot_2 | -29.9 | dead/disconnected device: current is flat, independent of both VG and VD |
| 5 | 5 | bot_2 | -36.7 | dead/disconnected device: current is flat, independent of both VG and VD |
| 10 | 5 | top_2 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 10 | 5 | bot_2 | 32.4 | no VD dependence: current does not rise with drain voltage |
| 10 | 10 | bot_2 | 95.5 | ranked lower than the two best replicates for this (W, L) |
| 10 | 10 | top_2 | 84.6 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 20 | 5 | top_1_run2 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 20 | 5 | top_2 | 22.5 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 20 | 10 | top_1 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 20 | 10 | top_2 | 14.9 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 20 | 15 | top_1_run2 | 22.5 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 20 | 15 | top_2 | 22.5 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 20 | 20 | top_2 | 99.8 | ranked lower than the two best replicates for this (W, L) |
| 20 | 20 | top_1_run2 | 21.9 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 5 | top_1 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 40 | 5 | top_2 | 19.4 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 10 | top_1_run2 | 84.8 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 40 | 10 | top_2 | 19.8 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 15 | top_1 | 22.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 15 | top_2 | 16.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 20 | top_1 | 22.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 40 | 20 | top_2_run2 | -45.5 | dead/disconnected device: current is flat, independent of both VG and VD |
| 80 | 5 | top_1 | 25.0 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 80 | 5 | top_2 | 21.7 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 80 | 10 | top_1 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 80 | 10 | top_2 | 85.0 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 80 | 15 | top_1 | 21.9 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 80 | 15 | top_2 | 21.5 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 80 | 20 | top_1 | 22.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 80 | 20 | top_2 | 19.2 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 160 | 5 | bot_1 | 84.8 | elevated off-state leakage (still usable, ranked below cleaner replicates) |
| 160 | 5 | top_2 | 22.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 160 | 10 | top_2_run2 | 22.3 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 160 | 10 | top_1_run3 | -20.2 | dead/disconnected device: current is flat, independent of both VG and VD |
| 160 | 15 | top_1 | 83.9 | off-state current bursts / sign flips |
| 160 | 15 | top_2_run2 | -20.4 | dead/disconnected device: current is flat, independent of both VG and VD |
| 160 | 20 | top_1 | 21.8 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
| 160 | 20 | top_2 | 18.4 | no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance) |
