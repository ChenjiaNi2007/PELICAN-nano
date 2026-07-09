#!/usr/bin/env bash
# Phase A of nPELICAN-fpga/docs/INPUT_WIDTH_RETRAIN_PLAN.md:
# retrain QAT with the input (d_ij) quantizer width swept, weight/act fixed at 24,
# all other flags identical to the current firmware checkpoint (fpga_model_qat).
#
# Smoke test (sample_data, CPU):
#   bash scripts/sweep_input_width.sh
# Full run (GPU/cluster):
#   DATADIR=<full-dataset-dir> EPOCHS=35 DEVICE="--cuda --no-reproducible" \
#       bash scripts/sweep_input_width.sh
#
# Overridable: WIDTHS, DATADIR, EPOCHS, DEVICE, SEED, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

WIDTHS=${WIDTHS:-"24 20 18 16 14 12 10"}
DATADIR=${DATADIR:-./data/sample_data}
EPOCHS=${EPOCHS:-8}
DEVICE=${DEVICE:---cpu}      # GPU: DEVICE="--cuda --no-reproducible"
SEED=${SEED:-42}             # fixed across widths for comparability
PY=${PY:-python3}

for W in $WIDTHS; do
    PREFIX="fpga_model_qat_w24a24i${W}"
    echo "=== training input width ${W} -> model/${PREFIX}_best.pt ==="
    $PY train_pelican_nano.py \
        --datadir "$DATADIR" --target is_signal \
        --nobj 20 --nobj-avg 49 --n-hidden 2 \
        --num-epoch "$EPOCHS" --batch-size 256 \
        --quant --po2-scales \
        --weight-bit-width 24 --act-bit-width 24 --input-bit-width "$W" \
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 \
        --seed "$SEED" --prefix "$PREFIX" $DEVICE
done

echo
echo "=== sweep summary (fill the table in INPUT_WIDTH_RETRAIN_PLAN.md) ==="
for W in $WIDTHS; do
    PREFIX="fpga_model_qat_w24a24i${W}"
    BEST="model/${PREFIX}_best.pt"
    echo
    echo "--- input width ${W} ---"
    if [[ -f "$BEST" ]]; then
        # learned input_quant scale 2^-k; clip point is 2^(W-1-k)
        $PY scripts/check_scales.py --checkpoint "$BEST" \
            --weight-bit-width 24 --act-bit-width 24 --input-bit-width "$W" \
            | grep -E "input_quant" || true
    else
        echo "  (no checkpoint: $BEST)"
    fi
    grep -E "AUC" "log/${PREFIX}.log" | tail -3 || true
done
