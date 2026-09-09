#!/usr/bin/env bash
# Second queue (2026-09-08 07:30), chained after the Phase 3 (qatf x3 seeds) driver:
#   A  qat as-is 20 ep + qatf 8 ep, seed 1 (h 2,4)   quantify the cooldown-freeze effect
#   B  qatf12 20 ep, seeds 1 2 3                      scales frozen from epoch 12 (see driver docstring)
#   C  float  LR 0.001 / 0.005, 20 ep, seed 1         LR sensitivity
#   D  qatf12 LR 0.001 / 0.005, 20 ep, seed 1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
COMMON="--hs 2 4 --datadir data/toptag20 --jobs ${JOBS:-3} --threads ${THREADS:-2}"
echo "queue2 start $(date)"
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qat qatf --lrs 0.0025 --epochs 20 8 --seeds 1
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qatf12   --lrs 0.0025 --epochs 20 --seeds 1 2 3
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes float    --lrs 0.001 0.005 --epochs 20 --seeds 1
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qatf12   --lrs 0.001 0.005 --epochs 20 --seeds 1
echo "queue2 done $(date)"
$PY scripts/training_curves.py --group-by h && $PY scripts/capacity_report.py
