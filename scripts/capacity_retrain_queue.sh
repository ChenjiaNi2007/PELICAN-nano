#!/usr/bin/env bash
# Full h=2 vs h=4 retrain queue (see scripts/capacity_retrain.py for the rationale).
# Resumable: finished runs (log says "Inference phase complete") are skipped, so
# re-running this script after a kill/reboot continues where it stopped. Never start it
# while another driver is still training the same prefixes (they would be relaunched).
#
# Phase 1a  production LR 0.0025, 8 epochs (the old budget, fixed cooldown), seed 1  [done]
# Phase 1b  float, 20 epochs, seeds 1 2 3                       <- headline float comparison
# Phase 3   qatf (QAT + scales frozen for the cooldown), 20 ep, seeds 1 2 3
#           <- headline QAT comparison; motivated by the late po2 scale flips seen in 1a
# Phase 1b' qat as-is (no freeze), 20 ep, seed 1                <- quantifies the freeze effect
# Phase 3a  qatf 8 ep seed 1                                    <- direct pair to the 1a qat runs
# Phase 2   LR 0.001 / 0.005, 20 ep, seed 1: float first, then qatf   <- LR sensitivity
# all phases: h in {2, 4}
#
# CPU only (Brevitas needs named tensors, which MPS lacks; float is slower on MPS).
# 3 concurrent jobs x 2 threads gives ~1.4x the throughput of one 4-thread job on an M1.
# caffeinate keeps the laptop from idling to sleep; progress:
#   .venv/bin/python scripts/training_curves.py --watch
#   tail -f log/capacity_retrain_queue.out
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-3}
THREADS=${THREADS:-2}
COMMON="--hs 2 4 --datadir data/toptag20 --jobs $JOBS --threads $THREADS"
echo "queue start $(date)"
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes float qat --lrs 0.0025 --epochs 8  --seeds 1
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes float      --lrs 0.0025 --epochs 20 --seeds 1 2 3
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qatf       --lrs 0.0025 --epochs 20 --seeds 1 2 3
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qat qatf   --lrs 0.0025 --epochs 20 8 --seeds 1
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes float      --lrs 0.001 0.005 --epochs 20 --seeds 1
caffeinate -i -s $PY scripts/capacity_retrain.py $COMMON --modes qatf       --lrs 0.001 0.005 --epochs 20 --seeds 1
echo "queue done $(date)"
$PY scripts/training_curves.py --group-by h && $PY scripts/capacity_report.py
