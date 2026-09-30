#!/usr/bin/env bash
# Rebuild the thesis-faithful CAMCAS model end to end (~8 min, step 6 dominates).
set -euo pipefail
cd "$(dirname "$0")"
python step0_select_curves.py
python step1_extract_params.py
python step2_plot_params.py
python step3_fit_scaling.py thesis
python step6_improve.py          # builds, validates and CV-scores all variants; writes the final .va
python step7_report.py
