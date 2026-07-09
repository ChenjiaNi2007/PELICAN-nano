#!/usr/bin/env bash
# Phase A* of nPELICAN-fpga/docs/INPUT_WIDTH_RETRAIN_PLAN.md:
# retrain QAT with a MOMENTUM quantizer (--pmu-bit-width, the firmware input_t
# grid) swept, at the production dot/weight/act widths (default 6/6/6).
# 18 first: sanity anchor against the current input_t=18 operating point.
#
# Smoke test (sample_data, CPU):
#   bash scripts/sweep_pmu_width.sh
# Full run (GPU/cluster):
#   DATADIR=<full-dataset-dir> EPOCHS=35 DEVICE="--cuda --no-reproducible" \
#       bash scripts/sweep_pmu_width.sh
#
# Overridable: PMU_WIDTHS, WBITS/ABITS/IBITS, DATADIR, EPOCHS, DEVICE, SEED, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

PMU_WIDTHS=${PMU_WIDTHS:-"18 16 14 12 10"}
WBITS=${WBITS:-6}
ABITS=${ABITS:-6}
IBITS=${IBITS:-6}
DATADIR=${DATADIR:-./data/sample_data}
EPOCHS=${EPOCHS:-8}
DEVICE=${DEVICE:---cpu}      # GPU: DEVICE="--cuda --no-reproducible"
SEED=${SEED:-42}             # fixed across widths for comparability
PY=${PY:-python3}

for PW in $PMU_WIDTHS; do
    PREFIX="fpga_model_qat_w${WBITS}a${ABITS}i${IBITS}p${PW}"
    echo "=== training pmu width ${PW} -> model/${PREFIX}_best.pt ==="
    $PY train_pelican_nano.py \
        --datadir "$DATADIR" --target is_signal \
        --nobj 20 --nobj-avg 49 --n-hidden 2 \
        --num-epoch "$EPOCHS" --batch-size 256 \
        --quant --po2-scales \
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" \
        --input-bit-width "$IBITS" --pmu-bit-width "$PW" \
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 \
        --seed "$SEED" --prefix "$PREFIX" $DEVICE
done

echo
echo "=== sweep summary (fill Phase A* table in INPUT_WIDTH_RETRAIN_PLAN.md) ==="
for PW in $PMU_WIDTHS; do
    PREFIX="fpga_model_qat_w${WBITS}a${ABITS}i${IBITS}p${PW}"
    BEST="model/${PREFIX}_best.pt"
    echo
    echo "--- pmu width ${PW} ---"
    if [[ -f "$BEST" ]]; then
        # learned momentum scale 2^-k -> input_t = ap_fixed<PW, PW-k>, clip 2^(PW-1-k)
        $PY scripts/check_scales.py --checkpoint "$BEST" \
            --weight-bit-width "$WBITS" --act-bit-width "$ABITS" \
            --input-bit-width "$IBITS" --pmu-bit-width "$PW" \
            | grep -E "pmu_quant|input_quant" || true
    else
        echo "  (no checkpoint: $BEST)"
    fi
    grep -E "AUC" "log/${PREFIX}.log" | tail -3 || true
done
