#!/usr/bin/env bash
# Phase 1c: 6-bit QAT of the jet-spurion models with the SPLIT jet quantizers at WIDE widths
# (jet-row dots 10 bits, m_jet^2 16 bits, jet momentum 20 bits; particle grids stay 6/6/6 + pmu12).
# Why: at 6 bits the m_jet^2 LSB (~1000 GeV^2) is half the W/Z mass^2 gap and the 12-bit jet momentum
# grid puts ~1e4 GeV^2 of cancellation error into m^2 = E^2 - p^2. One row + one scalar per jet -> free.
# Waits for (a) the phase-1c flags to exist in train_pelican_nano.py and (b) the running qat6sj* runs.
#   caffeinate -i -s scripts/hls4ml5_queue3.sh > log/hls4ml5_queue3.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}; THREADS=${THREADS:-2}; export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "queue3: waiting for flags + running qat6sj runs ($(date))"
until "$PY" train_pelican_nano.py --help 2>/dev/null | grep -q -- "--jet-pmu-bit-width"; do sleep 60; done
# (phase-1b split runs s2/jet-only were stopped at epoch 5 on 2026-10-07 14:50 to free CPU; s1 continues as the 6-bit-split data point)
echo "queue3: start ($(date))"
COMMON=(--datadir data/hls4ml5j_n16 --target label --n-out 5 --n-hidden 4 --add-jet --jet-quant-split
        --jet-input-bit-width 10 --mjet-input-bit-width 16 --jet-pmu-bit-width 20
        --nobj 16 --nobj-avg 49 --num-epoch 40 --lr-init 0.0025 --lr-decay-type cos --batch-size 256
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 --activation relu --batchnorm b
        --cpu --no-predict --summarize-csv all --quant --po2-scales --weight-bit-width 6 --act-bit-width 6
        --input-bit-width 6 --pmu-bit-width 12 --freeze-scales-epoch 32)
run() { local prefix=$1; shift; echo "$(date '+%F %T') start $prefix";
  ( "$PY" -u train_pelican_nano.py --prefix "$prefix" "${COMMON[@]}" "$@" > "log/${prefix}.stdout" 2>&1 \
      && echo "$(date '+%F %T') done  $prefix" || echo "$(date '+%F %T') FAIL  $prefix" ) & }
run qat6wj16_h4_jh_e40_s1 --head-hidden 16 --seed 1
run qat6wj16_h4_jh_e40_s2 --head-hidden 16 --seed 2
run qat6wj16_h4_jet_e40_s1 --seed 1
wait; echo "queue3 end $(date)"
