#!/usr/bin/env bash
# Lever 7 of nPELICAN-fpga/docs/RESOURCE_REDUCTION_LEVERS.md:
# retrain QAT with a PER-PARTICLE BLOCK-FLOATING-POINT momentum grid
# (--pmu-block-fp) and sweep the mantissa width, at the production
# dot/weight/act widths (default 6/6/6).
#
# What this is testing
# --------------------
# The dot-front-end analysis (nPELICAN-fpga/analysis/blockfp_dots.py) shows
# block-FP mantissas at W=8 land dots on the right dot_t cell more often
# (12.20% mismatch) than the CURRENT UNIFORM 12-bit production grid (17.61%).
# This sweep answers the only question that analysis cannot: does AUC hold?
#
#   baseline to beat: fpga_model_qat_w6a6i6p12 (uniform, 12-bit).
#   Its checkpoint best_metrics AUC is 0.9515; the full-dataset TEST AUC recorded
#   in INPUT_WIDTH_RETRAIN_PLAN.md is 0.9519. The summary below reports
#   best_metrics for every row, so rows are comparable to each other and to
#   0.9515 -- but only BASELINE=1 (same seed/epochs/data) is a true controlled
#   comparison. Do that before concluding anything from a small delta.
#
# Read the summary as: the largest W where AUC still matches baseline is the
# operating point; the interesting region is W = 10 down to 7. If AUC holds at
# W<=8 the DSP48 shared-operand packing opens up (2 mults/DSP, ~506 dot DSP).
#
# Smoke test (sample_data, CPU, ~minutes):
#   bash scripts/sweep_pmu_blockfp.sh
# Full run (GPU/cluster):
#   DATADIR=<full-dataset-dir> EPOCHS=35 DEVICE="--cuda --no-reproducible" \
#       bash scripts/sweep_pmu_blockfp.sh
# Re-run the uniform baseline alongside for an apples-to-apples comparison
# (same seed/epochs/data):
#   BASELINE=1 DATADIR=... EPOCHS=35 DEVICE="--cuda --no-reproducible" \
#       bash scripts/sweep_pmu_blockfp.sh
#
# Overridable: MANTISSA_WIDTHS, EXP_MIN, EXP_MAX, WBITS/ABITS/IBITS, DATADIR,
#              EPOCHS, DEVICE, SEED, PY, BASELINE.
set -euo pipefail
cd "$(dirname "$0")/.."

MANTISSA_WIDTHS=${MANTISSA_WIDTHS:-"12 10 9 8 7"}
EXP_MIN=${EXP_MIN:-0}
EXP_MAX=${EXP_MAX:-10}
WBITS=${WBITS:-6}
ABITS=${ABITS:-6}
IBITS=${IBITS:-6}
DATADIR=${DATADIR:-./data/sample_data}
EPOCHS=${EPOCHS:-8}
DEVICE=${DEVICE:---cpu}      # GPU: DEVICE="--cuda --no-reproducible"
SEED=${SEED:-42}             # fixed across widths for comparability
PY=${PY:-python3}
BASELINE=${BASELINE:-0}      # also retrain the uniform p12 grid for comparison

COMMON=(--datadir "$DATADIR" --target is_signal
        --nobj 20 --nobj-avg 49 --n-hidden 2
        --num-epoch "$EPOCHS" --batch-size 256
        --quant --po2-scales
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS"
        --input-bit-width "$IBITS"
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005
        --seed "$SEED")

PREFIXES=()

if [[ "$BASELINE" == "1" ]]; then
    PREFIX="fpga_model_qat_w${WBITS}a${ABITS}i${IBITS}p12"
    echo "=== training UNIFORM baseline pmu=12 -> model/${PREFIX}_best.pt ==="
    # shellcheck disable=SC2086
    $PY train_pelican_nano.py "${COMMON[@]}" \
        --pmu-bit-width 12 --no-pmu-block-fp \
        --prefix "$PREFIX" $DEVICE
    PREFIXES+=("$PREFIX:uniform:12")
fi

for MW in $MANTISSA_WIDTHS; do
    PREFIX="fpga_model_qat_w${WBITS}a${ABITS}i${IBITS}bfp${MW}"
    echo "=== training block-FP mantissa ${MW} (exp [${EXP_MIN},${EXP_MAX}]) -> model/${PREFIX}_best.pt ==="
    # shellcheck disable=SC2086
    $PY train_pelican_nano.py "${COMMON[@]}" \
        --pmu-bit-width "$MW" --pmu-block-fp \
        --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX" \
        --prefix "$PREFIX" $DEVICE
    PREFIXES+=("$PREFIX:blockfp:$MW")
done

echo
echo "=== sweep summary (fill the Lever 7 table in RESOURCE_REDUCTION_LEVERS.md) ==="
echo "baseline to beat: uniform pmu=12, AUC 0.9515"
echo
printf '%-38s %-9s %-4s %-8s %-8s %s\n' "checkpoint" "grid" "W" "AUC" "acc" "bgRej@0.5"
for ENTRY in "${PREFIXES[@]}"; do
    PREFIX="${ENTRY%%:*}"; REST="${ENTRY#*:}"
    KIND="${REST%%:*}"; MW="${REST#*:}"
    BEST="model/${PREFIX}_best.pt"
    if [[ -f "$BEST" ]]; then
        # best_metrics on the checkpoint is authoritative; the training log's
        # per-batch AUC lines are noisy and its last line is NOT the best epoch.
        METRICS=$($PY - "$BEST" <<'PYEOF'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("best_metrics", {})
print("%-8.4f %-8.4f %s" % (m.get("AUC", float("nan")),
                            m.get("accuracy", float("nan")),
                            round(m.get("BgRejectionAt0.5", float("nan")), 1)))
PYEOF
)
    else
        METRICS="(no checkpoint)"
    fi
    printf '%-38s %-9s %-4s %s\n' "$PREFIX" "$KIND" "$MW" "$METRICS"
done

echo
echo "Per-checkpoint quantizer detail:"
for ENTRY in "${PREFIXES[@]}"; do
    PREFIX="${ENTRY%%:*}"; REST="${ENTRY#*:}"
    KIND="${REST%%:*}"; MW="${REST#*:}"
    BEST="model/${PREFIX}_best.pt"
    [[ -f "$BEST" ]] || continue
    echo
    echo "--- ${PREFIX} ---"
    BFP_FLAG=()
    [[ "$KIND" == "blockfp" ]] && BFP_FLAG=(--pmu-block-fp --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX")
    $PY scripts/check_scales.py --checkpoint "$BEST" \
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" \
        --input-bit-width "$IBITS" --pmu-bit-width "$MW" "${BFP_FLAG[@]}" \
        | grep -E "pmu_quant|input_quant" || true
done

echo
echo "NEXT: if AUC holds at W<=8, csynth the 253 realignment shifters"
echo "      (Lever 7e) before doing any firmware work."
