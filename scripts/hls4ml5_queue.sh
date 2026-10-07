#!/usr/bin/env bash
# hls4ml 5-class (g/q/W/Z/t) nanoPELICAN run matrix -- comparison with the MLP-Mixer paper
# arXiv:2503.03103 Table 2 (N=16, N=32). See docs/HLS4ML_5CLASS.md. NOT run automatically:
# review, comment lines in/out (one run per line), then launch from anywhere:
#   caffeinate -i -s scripts/hls4ml5_queue.sh > log/hls4ml5_queue.out 2>&1 &
#
# Recipe = production toptag recipe (scripts/capacity_retrain.py::RECIPE) with a 5-logit
# head: --target label --n-out 5, nobj-avg 49, batch 256, wd 0.005, dropout 0.05/0.05,
# ReLU, BN b, 20 epochs, lr 0.0025 cos (4 warmup + cosine + 3 cooldown), CPU only
# (Brevitas breaks on MPS), --no-predict, per-epoch CSV via --summarize-csv all.
# QAT (phase 2) adds the firmware operating point w6 a6 i6 pmu12 + --freeze-scales-epoch 12.
#   NB: QAT clip ranges were tuned on toptag; hls4ml jets carry ~2x the energy -> re-derive
#   (scripts/check_scales.py) after the first QAT run before trusting phase 2.
#
# --nobj-avg: 49 is the firmware constant invnave=1/49 and is kept for comparability with
#   the toptag models. The true mean multiplicity after truncation is ~16 at N=16 and ~29 at
#   N=32, so sums are normalized ~3x/1.7x below unity. A variant worth one seed each:
#   replace "--nobj-avg 49" via extra flags, e.g.  run h5n16_h2_float_navg16_s1 ... --nobj-avg 16
#   (argparse takes the last occurrence, so an extra --nobj-avg overrides the default).
#
# Cost (8 GB M1, CPU): toptag N=20 float h=2 ~5-6 min/epoch on 1.2M jets. Here 558k train
# jets (~0.46x) and cost ~N^2: N=16 ~1.7 min/epoch -> ~35 min/run; N=32 ~7 min/epoch ->
# ~2.3 h/run. h=4/8 add some LUT-free FLOPs (~+20-50%); QAT ~1.5-2x float.
# Phase 1 = 18 runs ~ 9 N=16 x 0.6 h + 9 N=32 x 2.8 h ~ 30 CPU-h -> ~10-12 h wall at
#   MAX_PARALLEL=3 x 2 threads.
#
# Evaluate each finished run on the hls4ml test split (260k jets), e.g.:
#   .venv/bin/python -u scripts/eval_hls4ml5.py --checkpoint model/h5n16_h2_float_s1_best.pt \
#       --print-paper --tag h5n16_h2_float_s1
# Rows accumulate in ../results/hls4ml5/summary.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p log

PY=${PY:-.venv/bin/python}
MAX_PARALLEL=${MAX_PARALLEL:-3}
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
# `wait -n` needs bash >= 4.3; macOS ships /bin/bash 3.2 -> fall back to polling.
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
  HAVE_WAIT_N=1
else
  HAVE_WAIT_N=0
fi

D16=data/hls4ml5_n16
D32=data/hls4ml5_n32
QAT=(--quant --po2-scales --weight-bit-width 6 --act-bit-width 6 --input-bit-width 6
     --pmu-bit-width 12 --freeze-scales-epoch 12)

# run PREFIX DATADIR NOBJ N_HIDDEN SEED [extra flags...]  -- launches in the background,
# throttled to MAX_PARALLEL concurrent jobs.
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
      --n-hidden "$h" --nobj "$nobj" --nobj-avg 49 --num-epoch 20 --lr-init 0.0025 --lr-decay-type cos \
      --batch-size 256 --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 --activation relu --batchnorm b \
      --cpu --no-predict --summarize-csv all --seed "$seed" ${extra[@]+"${extra[@]}"} > "log/${prefix}.stdout" 2>&1 \
      && echo "$(date '+%F %T') done  $prefix" || echo "$(date '+%F %T') FAIL  $prefix (see log/${prefix}.stdout)"
  ) &
}

echo "queue start $(date)"

# ---- Phase 1: float, N in {16,32} x h in {2,4,8} x seeds {1,2,3} ----
run h5n16_h2_float_s1 $D16 16 2 1
run h5n16_h2_float_s2 $D16 16 2 2
run h5n16_h2_float_s3 $D16 16 2 3
run h5n16_h4_float_s1 $D16 16 4 1
run h5n16_h4_float_s2 $D16 16 4 2
run h5n16_h4_float_s3 $D16 16 4 3
run h5n16_h8_float_s1 $D16 16 8 1
run h5n16_h8_float_s2 $D16 16 8 2
run h5n16_h8_float_s3 $D16 16 8 3
run h5n32_h2_float_s1 $D32 32 2 1
run h5n32_h2_float_s2 $D32 32 2 2
run h5n32_h2_float_s3 $D32 32 2 3
run h5n32_h4_float_s1 $D32 32 4 1
run h5n32_h4_float_s2 $D32 32 4 2
run h5n32_h4_float_s3 $D32 32 4 3
run h5n32_h8_float_s1 $D32 32 8 1
run h5n32_h8_float_s2 $D32 32 8 2
run h5n32_h8_float_s3 $D32 32 8 3

# ---- Phase 2: QAT (w6a6i6 pmu12, scales frozen from epoch 12) -- off by default ----
# run h5n16_h2_qatf12_s1 $D16 16 2 1 "${QAT[@]}"
# run h5n16_h2_qatf12_s2 $D16 16 2 2 "${QAT[@]}"
# run h5n16_h2_qatf12_s3 $D16 16 2 3 "${QAT[@]}"
# run h5n16_h4_qatf12_s1 $D16 16 4 1 "${QAT[@]}"
# run h5n16_h4_qatf12_s2 $D16 16 4 2 "${QAT[@]}"
# run h5n16_h4_qatf12_s3 $D16 16 4 3 "${QAT[@]}"
# run h5n16_h8_qatf12_s1 $D16 16 8 1 "${QAT[@]}"
# run h5n16_h8_qatf12_s2 $D16 16 8 2 "${QAT[@]}"
# run h5n16_h8_qatf12_s3 $D16 16 8 3 "${QAT[@]}"
# run h5n32_h2_qatf12_s1 $D32 32 2 1 "${QAT[@]}"
# run h5n32_h2_qatf12_s2 $D32 32 2 2 "${QAT[@]}"
# run h5n32_h2_qatf12_s3 $D32 32 2 3 "${QAT[@]}"
# run h5n32_h4_qatf12_s1 $D32 32 4 1 "${QAT[@]}"
# run h5n32_h4_qatf12_s2 $D32 32 4 2 "${QAT[@]}"
# run h5n32_h4_qatf12_s3 $D32 32 4 3 "${QAT[@]}"
# run h5n32_h8_qatf12_s1 $D32 32 8 1 "${QAT[@]}"
# run h5n32_h8_qatf12_s2 $D32 32 8 2 "${QAT[@]}"
# run h5n32_h8_qatf12_s3 $D32 32 8 3 "${QAT[@]}"

wait
echo "queue done $(date)"
