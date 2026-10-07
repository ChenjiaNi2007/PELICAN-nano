#!/usr/bin/env bash
# Clip-floored companion arm to scripts/capacity_retrain_queueN.sh: h=2, 6/6/6-pmu12 QAT,
# scales frozen at epoch 12, 3 seeds, at EVERY N (12 8 16 20 24 32), with --input-clip-min 512.
#
# Why: under the plain N=20 recipe the learned 6-bit d_ij clip is a basin lottery (7h: ~17% of
# runs collapse to clip ~128, AUC ~0.93). Truncation makes it worse -- the surviving
# constituents are the hardest, so pair dots grow (p90 89 -> 189 GeV^2 from N=20 to N=12) and a
# 124 clip saturates 15% of pairs instead of 7%; cap12_h2_qatf12 seed 1 sat at 0.935 from epoch
# 4. The inventory's adopted fix for the operating point is --input-clip-min 512 (a floor: the
# scale may still learn a larger clip), so this arm gives the headline h=2 QAT accuracy-vs-N
# curve without the lottery, and 16/20 are re-run here so the curve is one recipe end to end.
# Runs after queueN finishes (waits on its process). Prefixes capc{N}_h2_qatf12_lr0p0025_e20_s{1,2,3}.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
NLIST=${NLIST:-"12 8 16 20 24"}
# 2026-09-29: the clip floor never beat the plain recipe at N=8/12/16/20/24 (neutral at 20,
# worse elsewhere: early-peak instability) -> the N=32 arm is skipped; see docs / memory.
case " $NLIST " in *" 32 "*) echo "queueNc: clip-floored arm at N=32 skipped (negative result at N=8..24)"; exit 0;; esac
while pgrep -f capacity_retrain_queueN.sh >/dev/null; do sleep 120; done
echo "queueNc start $(date)  N list: $NLIST"
for N in $NLIST; do
    if   [ "$N" -le 20 ]; then DD=data/toptag20; J=3; T=2
    elif [ "$N" -le 24 ]; then DD=data/toptag24; J=3; T=2
    else                       DD=data/toptag32; J=2; T=3; fi
    echo "=== clip512 N=$N  datadir=$DD  $(date) ==="
    caffeinate -i -s $PY scripts/capacity_retrain.py --tag "capc$N" --hs 2 --datadir "$DD" \
        --jobs "$J" --threads "$T" --modes qatf12 --lrs 0.0025 --epochs 20 \
        --seeds 1 2 3 --extra="--nobj $N --input-clip-min 512"
    echo "=== clip512 N=$N training done $(date); evaluating on $DD/test.h5 ==="
    OMP_NUM_THREADS=2 $PY scripts/capacity_eval.py --glob "capc${N}_*" --testfile "$DD/test.h5"
    echo "=== clip512 N=$N eval done $(date) ==="
done
echo "queueNc done $(date)"
