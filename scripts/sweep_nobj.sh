#!/usr/bin/env bash
# Sequence item 7 of nPELICAN-fpga/docs/RESOURCE_REDUCTION_LEVERS.md ("fewer
# particles"): sweep the constituent truncation N on top of the block-FP momentum
# grid, and spend Lever 7's accuracy surplus on cutting the dot front-end.
#
# Why this is the lever that actually moves resources
# ---------------------------------------------------
# The dot DSP count is 4 x the number of masked pairs, and pairs go QUADRATICALLY
# in N because the two beam spurions ride along: pairs = (N+2)(N+3)/2.
#
#   N=20 -> 253 pairs -> 1012 dot DSP   (current)
#   N=16 -> 171 pairs ->  684 dot DSP   (-32%)
#   N=12 -> 105 pairs ->  420 dot DSP   (-58%)
#   N=10 ->  78 pairs ->  312 dot DSP   (-69%)
#
# That dominates every bit-width lever tried so far: below the DSP48 inference
# threshold a 12x12 and an 8x8 multiply BOTH cost exactly one DSP, so narrowing
# operands stopped buying DSP at width 18 and went the wrong way below it (pmu10
# vsynth: -233 DSP but +25.9k LUT). Cutting N removes multipliers outright.
#
# What to read
# ------------
# Lever 7 bought +17.9% bgRej@0.5 at zero width cost (blockfp-12 vs uniform-12,
# 0.9603/46.7 vs 0.9544/39.6). This sweep asks how much N that surplus can pay for:
# the smallest N still at or above the uniform-20 baseline is the operating point.
#
# Defaults are the winning configuration from Lever 7g/7h: block-FP mantissa 12,
# 6/6/6 widths, and CLIP_MIN=512. That floor is NOT optional -- 2 of the 12 runs in
# 7h fell into a clip-128 basin and collapsed (AUC ~0.93, bgRej ~13) purely from the
# learned dot scale drifting down. Leaving it off makes these results unreadable.
#
# NOBJ_AVG stays 49 by default and does NOT track NOBJ: it is the dataset's true
# average multiplicity (the 1/Nbar normalization, invnave in the firmware), not the
# truncation. Truncating to 16 does not change how many constituents the jets have.
# If you want to test retuning it, set NOBJ_AVG explicitly -- but change one thing
# at a time, and remember the firmware hardcodes invnave=1/49.
#
# Smoke test (sample_data, CPU, ~minutes):
#   bash scripts/sweep_nobj.sh
# Full run:
#   DATADIR=<full-dataset-dir> EPOCHS=16 DEVICE="--cuda --no-reproducible" \
#       bash scripts/sweep_nobj.sh
#
# Overridable: NOBJ_LIST, MANTISSA_W, BLOCKFP, NOBJ_AVG, WBITS/ABITS/IBITS,
#              CLIP_MIN, EXP_MIN, EXP_MAX, DATADIR, EPOCHS, DEVICE, SEED, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

NOBJ_LIST=${NOBJ_LIST:-"20 16 12 10"}
MANTISSA_W=${MANTISSA_W:-12}   # block-FP mantissa width (Lever 7 operating point)
BLOCKFP=${BLOCKFP:-1}          # 0 = uniform pmu grid instead (control arm)
NOBJ_AVG=${NOBJ_AVG:-49}       # 1/Nbar normalization; NOT the truncation
EXP_MIN=${EXP_MIN:-0}
EXP_MAX=${EXP_MAX:-10}
WBITS=${WBITS:-6}
ABITS=${ABITS:-6}
IBITS=${IBITS:-6}
CLIP_MIN=${CLIP_MIN:-512}      # d_ij clip floor; "" disables (don't -- see 7h)
DATADIR=${DATADIR:-./data/sample_data}
EPOCHS=${EPOCHS:-16}           # >=8 required: the cos scheduler needs warmup+cooldown
DEVICE=${DEVICE:---cpu}        # GPU: DEVICE="--cuda --no-reproducible"
SEED=${SEED:-42}               # fixed across N for comparability
PY=${PY:-python3}

if [[ "$BLOCKFP" == "1" ]]; then
    GRID_FLAGS=(--pmu-bit-width "$MANTISSA_W" --pmu-block-fp
                --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX")
    GTAG="bfp${MANTISSA_W}"; GRID="blockfp"
else
    GRID_FLAGS=(--pmu-bit-width "$MANTISSA_W" --no-pmu-block-fp)
    GTAG="p${MANTISSA_W}"; GRID="uniform"
fi

COMMON=(--datadir "$DATADIR" --target is_signal
        --nobj-avg "$NOBJ_AVG" --n-hidden 2
        --num-epoch "$EPOCHS" --batch-size 256
        --quant --po2-scales
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS"
        --input-bit-width "$IBITS"
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005
        --seed "$SEED")
[[ -n "$CLIP_MIN" ]] && COMMON+=(--input-clip-min "$CLIP_MIN")

PREFIXES=()
for N in $NOBJ_LIST; do
    PREFIX="fpga_model_qat_w${WBITS}a${ABITS}i${IBITS}${GTAG}n${N}"
    echo "=== training ${GRID} N=${N} -> model/${PREFIX}_best.pt ==="
    # shellcheck disable=SC2086
    $PY train_pelican_nano.py "${COMMON[@]}" "${GRID_FLAGS[@]}" \
        --nobj "$N" --prefix "$PREFIX" $DEVICE
    PREFIXES+=("$PREFIX:$N")
done

echo
echo "=== nobj sweep summary (grid=${GRID}, W=${MANTISSA_W}, clip_min=${CLIP_MIN:-none}) ==="
echo "Lever 7 reference at N=20: blockfp-12 AUC 0.9603 / bgRej 46.7;"
echo "uniform-12 baseline it has to stay above: AUC 0.9544 / bgRej 39.6."
echo
printf '%-40s %-5s %-7s %-9s %-8s %-8s %s\n' \
    "checkpoint" "N" "pairs" "dot DSP" "AUC" "acc" "bgRej@0.5"
for ENTRY in "${PREFIXES[@]}"; do
    PREFIX="${ENTRY%%:*}"; N="${ENTRY##*:}"
    PAIRS=$(( (N + 2) * (N + 3) / 2 ))
    DSP=$(( PAIRS * 4 ))
    BEST="model/${PREFIX}_best.pt"
    if [[ -f "$BEST" ]]; then
        # best_metrics on the checkpoint is authoritative; the log's last AUC line
        # is a noisy per-batch training value, NOT the best epoch.
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
    printf '%-40s %-5s %-7s %-9s %s\n' "$PREFIX" "$N" "$PAIRS" "$DSP" "$METRICS"
done

echo
echo "Per-checkpoint quantizer detail:"
for ENTRY in "${PREFIXES[@]}"; do
    PREFIX="${ENTRY%%:*}"
    BEST="model/${PREFIX}_best.pt"
    [[ -f "$BEST" ]] || continue
    echo; echo "--- ${PREFIX} ---"
    EXTRA=()
    [[ "$BLOCKFP" == "1" ]] && EXTRA=(--pmu-block-fp --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX")
    # Replay CLIP_MIN: Brevitas stores the raw stat and clamps every forward, so
    # omitting it here reports a scale the model never used (Lever 7h-ii).
    [[ -n "$CLIP_MIN" ]] && EXTRA+=(--input-clip-min "$CLIP_MIN")
    $PY scripts/check_scales.py --checkpoint "$BEST" \
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" \
        --input-bit-width "$IBITS" --pmu-bit-width "$MANTISSA_W" "${EXTRA[@]}" \
        | grep -E "pmu_quant|input_quant" || true
done

echo
echo "NEXT: pick the smallest N still at/above the uniform-20 baseline, then"
echo "      csynth it -- the dot DSP column above is arithmetic, not synthesized,"
echo "      and NPARTICLES is a firmware constant (nPELICAN.h) that must be"
echo "      changed to match before any resource claim is real."
