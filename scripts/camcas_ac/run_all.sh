#!/usr/bin/env bash
# AC/charge-model analysis end to end (~1 min). Does not modify any .va file.
set -euo pipefail
cd "$(dirname "$0")"
python step1_inspect.py
python step2_4_analyse.py
python step5_6_validate.py
python propose_va_patch.py
