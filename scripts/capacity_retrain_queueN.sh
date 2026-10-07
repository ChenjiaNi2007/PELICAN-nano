#!/usr/bin/env bash
# Accuracy-vs-particle-count sweep: the N=16 companion (scripts/capacity_retrain_queue16.sh)
# extended to N in {12, 24, 8, 32}. Same recipe as the headline N=20 rows in
# results/roc_summary.csv (20 ep cos, peak LR 0.0025, 3 seeds; float and 6/6/6-pmu12 QAT with
# --freeze-scales-epoch 12; h=2 and h=4), ONLY --nobj changes. --nobj-avg stays 49 (dataset
# multiplicity / firmware invnave, not the truncation).
#
# Why these N: 8/16/32 are the constituent counts of the DeepSet comparison paper
# (arXiv:2402.01876) and the inventory's planned resource-vs-N points; 20 is the deployed
# operating point; 12 and 24 fill the curve either side of it. Order = value per hour:
# 12 (cheap, requested) -> 24 (requested) -> 8 (cheap) -> 32 (the expensive tail, ~3.6x N=16).
#
# Data: N<=20 read data/toptag20 (first 20 constituents), N=24 data/toptag24, N=32 data/toptag32
# (built by scripts/make_toptag20.py --nobj 24|32); collate then keeps p[:N]. N=32 runs 2-way
# instead of 3-way because each process holds the whole file in RAM (~1.3 GB at 32).
#
# Prefixes cap{N}_h{2,4}_{float,qatf12}_lr0p0025_e20_s{1,2,3}. Resumable (finished runs are
# skipped, unfinished ones continue from model/<prefix>.pt). Each N ends with the full-test-set
# evaluation on ITS OWN test file (a 20-constituent file would silently truncate a 24/32 model).
#   progress: .venv/bin/python scripts/training_curves.py --glob 'cap12_*' --watch
#             tail -f log/capacity_retrain_queueN.out
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
NLIST=${NLIST:-"12 24 8 32"}
echo "queueN start $(date)  N list: $NLIST"
for N in $NLIST; do
    if   [ "$N" -le 20 ]; then DD=data/toptag20; J=3; T=2
    elif [ "$N" -le 24 ]; then DD=data/toptag24; J=3; T=2
    else                       DD=data/toptag32; J=2; T=3; fi
    while [ ! -f "$DD/train.h5" ]; do echo "waiting for $DD/train.h5 ($(date))"; sleep 60; done
    echo "=== N=$N  datadir=$DD  jobs=$J threads=$T  $(date) ==="
    caffeinate -i -s $PY scripts/capacity_retrain.py --tag "cap$N" --hs 2 4 --datadir "$DD" \
        --jobs "$J" --threads "$T" --modes float qatf12 --lrs 0.0025 --epochs 20 \
        --seeds 1 2 3 --extra="--nobj $N"
    echo "=== N=$N training done $(date); evaluating on $DD/test.h5 ==="
    OMP_NUM_THREADS=2 $PY scripts/capacity_eval.py --glob "cap${N}_*" --testfile "$DD/test.h5"
    echo "=== N=$N eval done $(date) ==="
done
echo "queueN done $(date)"
