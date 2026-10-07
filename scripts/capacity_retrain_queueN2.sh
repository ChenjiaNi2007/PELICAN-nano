#!/usr/bin/env bash
# Re-ordered tail of the N sweep (replaces the running queueN.sh + queueNc.sh orchestrators on
# 2026-09-28 07:20 EDT; the in-flight N=24 driver is left alone and waited for):
#   1. wait for the N=24 driver, evaluate N=24 on data/toptag24/test.h5
#   2. N=8 plain arm (cheap)
#   3. clip-floored h2 QAT arm at N = 12 8 16 20 24  <- the headline curve, before the 30 h N=32
#   4. N=32 plain arm, then N=32 clip-floored arm     <- optional tail; kill any time, resumable
# Why: at ~8 min/epoch for N=24 and ~2x that for N=32, the original order put the clip-floored
# curve ~2 days out behind the least essential point.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
echo "queueN2 start $(date)"
while pgrep -f "capacity_retrain.py --tag cap24" >/dev/null; do sleep 60; done
echo "=== N=24 training done $(date); evaluating on data/toptag24/test.h5 ==="
OMP_NUM_THREADS=2 $PY scripts/capacity_eval.py --glob "cap24_*" --testfile data/toptag24/test.h5
echo "=== N=24 eval done $(date) ==="
NLIST="8" bash scripts/capacity_retrain_queueN.sh
NLIST="12 8 16 20 24" bash scripts/capacity_retrain_queueNc.sh
NLIST="32" bash scripts/capacity_retrain_queueN.sh
NLIST="32" bash scripts/capacity_retrain_queueNc.sh
echo "queueN2 done $(date)"
