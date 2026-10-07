#!/usr/bin/env bash
# N=32 tail of the particle-count sweep, run ONE job at a time (4 threads).
# Why: at N=32 each trainer holds a ~1.3 GB dataset; two of them on the 8 GB laptop (with the
# user's apps) thrashed the page compressor (2026-09-29 12:30: 3.6 GB in compressor, 116 MB
# free, per-batch time 2-3x slower). One job at a time is faster in total and leaves headroom.
# Resumable: unfinished runs continue from model/<prefix>.pt (--load); finished ones are skipped.
# Same recipe as the rest of the sweep (see capacity_retrain_queueN.sh). Eval on toptag32/test.h5.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
echo "queue32 start $(date)"
caffeinate -i -s $PY scripts/capacity_retrain.py --tag cap32 --hs 2 4 --datadir data/toptag32 \
    --jobs 1 --threads 4 --modes float qatf12 --lrs 0.0025 --epochs 20 --seeds 1 2 3 --extra="--nobj 32"
echo "=== N=32 training done $(date); evaluating on data/toptag32/test.h5 ==="
OMP_NUM_THREADS=2 $PY scripts/capacity_eval.py --glob "cap32_*" --testfile data/toptag32/test.h5
echo "=== N=32 eval done $(date) ==="
echo "queue32 done $(date)"
