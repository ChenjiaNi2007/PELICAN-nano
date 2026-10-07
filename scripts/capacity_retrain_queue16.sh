#!/usr/bin/env bash
# N=16 constituents companion to scripts/capacity_retrain_queue.sh: the SAME recipe as the
# headline n=20 rows in results/roc_summary.csv (20 ep cos, peak LR 0.0025, 3 seeds; float
# and 6/6/6-pmu12 QAT with --freeze-scales-epoch 12) with ONLY the constituent truncation
# changed to --nobj 16. Fills the paper's missing "accuracy at N=16" point, which until now
# was only ever synthesized (a 20-particle checkpoint at NPARTICLES=16) and never trained.
#
# --nobj-avg stays 49 on purpose: it is the dataset's true average multiplicity (the 1/Nbar
# normalization, invnave in the firmware), not the truncation (see scripts/sweep_nobj.sh).
# --extra="--nobj 16" wins over the driver's pinned --nobj 20 (argparse last-occurrence).
#
# Prefixes: cap16_h{2,4}_{float,qatf12}_lr0p0025_e20_s{1,2,3}. Resumable: re-run after a
# kill/reboot (finished runs are skipped, unfinished ones continue from model/<prefix>.pt).
# Progress:  .venv/bin/python scripts/training_curves.py --glob 'cap16_*' --watch
#            tail -f log/capacity_retrain_queue16.out
# Afterwards: .venv/bin/python scripts/capacity_eval.py --glob 'cap16_*'   (full 404k test set)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-3}
THREADS=${THREADS:-2}
echo "queue16 start $(date)"
caffeinate -i -s $PY scripts/capacity_retrain.py --tag cap16 --hs 2 4 --datadir data/toptag20 \
    --jobs "$JOBS" --threads "$THREADS" --modes float qatf12 --lrs 0.0025 --epochs 20 \
    --seeds 1 2 3 --extra="--nobj 16"
echo "queue16 done $(date)"
$PY scripts/capacity_eval.py --glob 'cap16_*'
echo "queue16 eval done $(date)"
