#!/usr/bin/env bash
# Follow-up to capacity_retrain_queue32.sh: a 4th plain-recipe h=2 QAT seed at N=32, as was done at
# N=16 and N=24 when one of the three seeds fell into a known bad basin (here seed 2: 0.9392).
# Waits for queue32 to finish, then trains seed 4 (1 job, 4 threads) and evaluates on toptag32.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
while /usr/bin/pgrep -f capacity_retrain_queue32.sh >/dev/null; do sleep 120; done
echo "queue32b start $(date)"
caffeinate -i -s $PY scripts/capacity_retrain.py --tag cap32 --hs 2 --datadir data/toptag32 \
    --jobs 1 --threads 4 --modes qatf12 --lrs 0.0025 --epochs 20 --seeds 4 --extra="--nobj 32"
OMP_NUM_THREADS=2 $PY scripts/capacity_eval.py --glob "cap32_*" --testfile data/toptag32/test.h5
echo "queue32b done $(date)"
