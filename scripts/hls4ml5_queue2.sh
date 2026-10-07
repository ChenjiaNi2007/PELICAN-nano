#!/usr/bin/env bash
# hls4ml 5-class nanoPELICAN -- phase 1b: IMPROVEMENTS retrain (2026-10-07). See docs/HLS4ML_5CLASS.md.
# Motivation (measured): N=32 h=4 plain reaches 60.5% / mAUC 0.854, while a histogram classifier on
# (mass of kept constituents, FULL-jet mass) reaches 73.6% -> the model is starved of full-jet info.
#   --add-jet         full-jet 4-momentum (Pjet) as a 3rd spurion at slot 2  (needs data/hls4ml5j_*)
#   --head-hidden 16  ReLU hidden layer between the 2->0 pooling and the 5 logits
#   40 epochs         best epochs were 18-20 of 20 in every phase-1 run
# Matrix (N=32, h=4 = best phase-1 point): jet+head x3 seeds FIRST, then single-seed ablations
# (jet only, head only, plain@40ep), then one N=16 jet+head run (firmware-sized point).
# Each run is one line: comment in/out. Launch:
#   caffeinate -i -s scripts/hls4ml5_queue2.sh > log/hls4ml5_queue2.out 2>&1 &
# It WAITS for the phase-1 queue (hls4ml5_queue.sh) to finish before starting.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p log

PY=${PY:-.venv/bin/python}
MAX_PARALLEL=${MAX_PARALLEL:-3}
THREADS=${THREADS:-2}
EPOCHS=${EPOCHS:-40}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then HAVE_WAIT_N=1; else HAVE_WAIT_N=0; fi

# Wait for phase 1 (any train_pelican_nano.py with a h5n16_/h5n32_ prefix, or its queue script).
echo "queue2: waiting for phase-1 runs to finish ($(date))"
while pgrep -f "train_pelican_nano.py --prefix h5n(16|32)_h[0-9]+_float" >/dev/null 2>&1 \
   || pgrep -f "scripts/hls4ml5_queue.sh" >/dev/null 2>&1; do sleep 60; done
echo "queue2: phase 1 finished ($(date))"

J16=data/hls4ml5j_n16
J32=data/hls4ml5j_n32
for d in $J16 $J32; do
  [[ -f $d/train.h5 && -f $d/valid.h5 && -f $d/test.h5 ]] || { echo "missing $d (build with scripts/make_hls4ml5.py)"; exit 1; }
done

# run PREFIX DATADIR NOBJ N_HIDDEN SEED [extra flags...]
run() {
  local prefix=$1 datadir=$2 nobj=$3 h=$4 seed=$5
  shift 5
  local extra=("$@")
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    if (( HAVE_WAIT_N )); then wait -n || true; else sleep 20; fi
  done
  echo "$(date '+%F %T') start $prefix"
  (
    "$PY" -u train_pelican_nano.py --prefix "$prefix" --datadir "$datadir" --target label --n-out 5 \
      --n-hidden "$h" --nobj "$nobj" --nobj-avg 49 --num-epoch "$EPOCHS" --lr-init 0.0025 --lr-decay-type cos \
      --batch-size 256 --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 --activation relu --batchnorm b \
      --cpu --no-predict --summarize-csv all --seed "$seed" ${extra[@]+"${extra[@]}"} > "log/${prefix}.stdout" 2>&1 \
      && echo "$(date '+%F %T') done  $prefix" || echo "$(date '+%F %T') FAIL  $prefix (see log/${prefix}.stdout)"
  ) &
}

echo "queue2 start $(date)"
# ---- main: N=32 h=4, jet spurion + head, 40 epochs, 3 seeds ----
run h5j32_h4_jh_e40_s1 $J32 32 4 1 --add-jet --head-hidden 16
run h5j32_h4_jh_e40_s2 $J32 32 4 2 --add-jet --head-hidden 16
run h5j32_h4_jh_e40_s3 $J32 32 4 3 --add-jet --head-hidden 16
# ---- ablations (1 seed each): which improvement carries the gain ----
run h5j32_h4_jet_e40_s1  $J32 32 4 1 --add-jet
run h5j32_h4_head_e40_s1 $J32 32 4 1 --head-hidden 16
run h5j32_h4_plain_e40_s1 $J32 32 4 1
# ---- firmware-sized point ----
run h5j16_h4_jh_e40_s1 $J16 16 4 1 --add-jet --head-hidden 16
wait
echo "queue2 end $(date)"
