#!/usr/bin/env bash
# SPS -- STATIC PER-SLOT momentum exponent (--pmu-static-exp).
#
# Lever 7 (--pmu-block-fp, sweep_pmu_blockfp.sh) gives every particle a RUNTIME
# exponent floor(log2 E_i); it wins +0.006 AUC / +18% bgRej at W=12 but the encode +
# realign shifters cost ~+54k LUT. SPS keeps the same block-FP representation but the
# exponent is a TRAINED integer per particle SLOT (slot = position in the pT-ordered
# input; slots 0,1 = beams, 2..21 = constituents). It is a compile-time constant, so
# the firmware realignment is wiring: zero hardware cost vs the uniform grid.
# This sweep asks: how much of block-FP's accuracy win survives a static exponent?
#
# Arms (run SEQUENTIALLY, one training process at a time):
#   control   uniform grid     --pmu-bit-width 12                     sps_ctl_p12_s<seed>
#   static<W> SPS, mantissa W  --pmu-bit-width W --pmu-block-fp --pmu-static-exp
#                                                                     sps_stat_p<W>_s<seed>
#   staticf<W> SPS + clip floor: static flags + --pmu-exp-floor-batches $FLOOR_BATCHES
#                                                                     sps_statf_p<W>_s<seed>
#   fixed<W>   SPS, exponent NOT learned: staticf flags + --pmu-exp-fixed (held at the
#              data-derived floor table)                             sps_fix_p<W>_s<seed>
#   staticfz<W> SPS, exponents learned only until epoch $EXP_FREEZE (default 3), then
#              frozen (--pmu-exp-freeze-epoch); other scales follow FREEZE
#                                                                     sps_statfz_p<W>_s<seed>
#   dynamic<W> Lever 7, W      --pmu-bit-width W --pmu-block-fp       sps_dyn_p<W>_s<seed>
#
# Why staticf: the smoke showed soft slots learning clips below their particles (slot
# 15 -> 8 GeV while that rank's p95 E is ~12.6 GeV) -- the same learned-clip collapse
# basin input_quant had (fixed by --input-clip-min). staticf floors each slot's
# exponent at the running max of ceil(log2 max|E|)-1 over the first FLOOR_BATCHES
# training batches, so the clip covers every particle seen; above that it still learns.
# fixed<W> (not in the default queue) separates the static REPRESENTATION from the
# exponent LEARNING dynamics: same floor table, but frozen there for the whole run.
#
# Recipe = the RECOMMENDED one from scripts/capacity_retrain.py ("qatf12", h=2):
#   nobj 20, nobj-avg 49, batch 256, drop 0.05/0.05, wd 0.005, relu, batchnorm b,
#   --quant --po2-scales w6 a6 i6, lr-init 0.0025, cos decay, 20 epochs,
#   --freeze-scales-epoch 12 (also freezes the SPS exponents), --input-clip-min 512.
#
# Usage
#   bash scripts/sweep_pmu_static.sh                     # default queue, full toptag20
#   DATADIR=data/sample_data EPOCHS=8 ARMS="control static12" SEEDS=1 THREADS=4 \
#       bash scripts/sweep_pmu_static.sh                 # smoke (CPU, minutes)
#
# Environment (all optional)
#   QUEUE     explicit "arm:seed ..." list. Default:
#             "control:1 static12:1 staticf12:1 dynamic12:1 static12:2 staticf12:2
#              static12:3 staticf12:3 static10:1"
#   ARMS      if set, overrides QUEUE with ARMS x SEEDS (arm-major), e.g. "control static12"
#   SEEDS     seeds for ARMS (default "1")
#   DATADIR   default data/toptag20 (first-20-constituent full dataset)
#   EPOCHS    default 20            FREEZE   --freeze-scales-epoch, default 12 (0 = off)
#   THREADS   OMP/MKL threads, default 4   NHIDDEN default 2
#   EXP_MIN / EXP_MAX   exponent clamp, default 0 / 10
#   EXP_FREEZE     epoch for the staticfz arms (--pmu-exp-freeze-epoch), default 3.
#                  Why: full-data static runs showed Adam flipping the exponents upward
#                  every epoch (~lr/step regardless of grad size) and val loss climbing.
#   FLOOR_BATCHES  K for the staticf / fixed arms (--pmu-exp-floor-batches), default 16
#   CLIP_MIN  --input-clip-min, default 512 ("" disables)
#   DEVICE    default --cpu        PY  default .venv/bin/python (fallback python3)
#   WBITS / ABITS / IBITS  weight / act / d_ij input widths, default 6 / 6 / 6
#   TAG       inserted after 'sps_' in every prefix (TAG=w24 -> sps_w24_ctl_p12_s1);
#             no '-'. Empty (default) = untagged names
#   EXTRA     extra args appended to every train command (e.g. "--num-train 20000")
#
# Checkpoints are OVERWRITTEN on re-run (no skip-if-done): a smoke on sample_data writes
# the same prefixes as the real queue, so run the real queue after any smoke.
# Prefixes contain no '-' (argparse / downstream tooling).
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_QUEUE="control:1 static12:1 staticf12:1 dynamic12:1 static12:2 staticf12:2 static12:3 staticf12:3 static10:1"
QUEUE=${QUEUE:-$DEFAULT_QUEUE}
SEEDS=${SEEDS:-1}
DATADIR=${DATADIR:-data/toptag20}
EPOCHS=${EPOCHS:-20}
FREEZE=${FREEZE:-12}
THREADS=${THREADS:-4}
NHIDDEN=${NHIDDEN:-2}
EXP_MIN=${EXP_MIN:-0}
EXP_MAX=${EXP_MAX:-10}
FLOOR_BATCHES=${FLOOR_BATCHES:-16}
EXP_FREEZE=${EXP_FREEZE:-3}
CLIP_MIN=${CLIP_MIN-512}
DEVICE=${DEVICE:---cpu}
EXTRA=${EXTRA:-}
if [[ -z "${PY:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi
WBITS=${WBITS:-6}
ABITS=${ABITS:-6}
IBITS=${IBITS:-6}
TAG=${TAG:-}
if [[ "$TAG" == *-* ]]; then echo "TAG must not contain '-' (got '$TAG')" >&2; exit 1; fi
TP="sps_${TAG:+${TAG}_}"   # tagged stem prefix: sps_ or sps_<TAG>_
echo "=== SPS sweep: weight/act/input bits = ${WBITS}/${ABITS}/${IBITS}  TAG='${TAG}'  (prefix stem ${TP}...) ==="

if [[ -n "${ARMS:-}" ]]; then
    QUEUE=""
    for A in $ARMS; do for S in $SEEDS; do QUEUE+="$A:$S "; done; done
fi

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

COMMON=(--datadir "$DATADIR" --target is_signal
        --nobj 20 --nobj-avg 49 --n-hidden "$NHIDDEN"
        --num-epoch "$EPOCHS" --batch-size 256
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005
        --activation relu --batchnorm b
        --lr-init 0.0025 --lr-decay-type cos
        --quant --po2-scales
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" --input-bit-width "$IBITS"
        --summarize-csv all --no-predict)
[[ "$FREEZE" != "0" ]] && COMMON+=(--freeze-scales-epoch "$FREEZE")
[[ -n "$CLIP_MIN" ]] && COMMON+=(--input-clip-min "$CLIP_MIN")

# arm -> "prefix-stem|kind|W|flags"
arm_spec() {
    local arm=$1
    case "$arm" in
        control)    echo "${TP}ctl_p12|uniform|12|--pmu-bit-width 12 --no-pmu-block-fp" ;;
        static[0-9]*)
            local w=${arm#static}
            echo "${TP}stat_p${w}|static|${w}|--pmu-bit-width ${w} --pmu-block-fp --pmu-static-exp --pmu-exp-min ${EXP_MIN} --pmu-exp-max ${EXP_MAX}" ;;
        staticf[0-9]*)
            local w=${arm#staticf}
            echo "${TP}statf_p${w}|staticf|${w}|--pmu-bit-width ${w} --pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches ${FLOOR_BATCHES} --pmu-exp-min ${EXP_MIN} --pmu-exp-max ${EXP_MAX}" ;;
        staticfz[0-9]*)
            local w=${arm#staticfz}
            echo "${TP}statfz_p${w}|staticfz|${w}|--pmu-bit-width ${w} --pmu-block-fp --pmu-static-exp --pmu-exp-freeze-epoch ${EXP_FREEZE} --pmu-exp-min ${EXP_MIN} --pmu-exp-max ${EXP_MAX}" ;;
        fixed[0-9]*)
            local w=${arm#fixed}
            echo "${TP}fix_p${w}|fixed|${w}|--pmu-bit-width ${w} --pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches ${FLOOR_BATCHES} --pmu-exp-fixed --pmu-exp-min ${EXP_MIN} --pmu-exp-max ${EXP_MAX}" ;;
        dynamic[0-9]*)
            local w=${arm#dynamic}
            echo "${TP}dyn_p${w}|dynamic|${w}|--pmu-bit-width ${w} --pmu-block-fp --pmu-exp-min ${EXP_MIN} --pmu-exp-max ${EXP_MAX}" ;;
        *) echo "unknown arm '$arm' (control | static<W> | staticf<W> | staticfz<W> | fixed<W> | dynamic<W>)" >&2; return 1 ;;
    esac
}

RUNS=()   # "prefix|kind|W"
for ITEM in $QUEUE; do
    ARM=${ITEM%%:*}; SEED=${ITEM#*:}
    SPEC=$(arm_spec "$ARM")
    IFS='|' read -r STEM KIND MW FLAGS <<<"$SPEC"
    PREFIX="${STEM}_s${SEED}"
    echo "=== [$KIND W=$MW seed $SEED] -> model/${PREFIX}_best.pt ==="
    # shellcheck disable=SC2086
    $PY train_pelican_nano.py "${COMMON[@]}" $FLAGS --seed "$SEED" \
        --prefix "$PREFIX" $DEVICE $EXTRA
    RUNS+=("$PREFIX|$KIND|$MW")
done

echo
echo "=== SPS sweep summary (best_metrics from model/<prefix>_best.pt) ==="
printf '%-28s %-8s %-4s %-8s %-8s %s\n' "checkpoint" "grid" "W" "AUC" "acc" "bgRej@0.5"
for R in "${RUNS[@]}"; do
    IFS='|' read -r PREFIX KIND MW <<<"$R"
    BEST="model/${PREFIX}_best.pt"
    if [[ -f "$BEST" ]]; then
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
    printf '%-28s %-8s %-4s %s\n' "$PREFIX" "$KIND" "$MW" "$METRICS"
done

echo
echo "Per-checkpoint quantizer detail (static runs: per-slot exponent table, slots 0,1 = beams):"
for R in "${RUNS[@]}"; do
    IFS='|' read -r PREFIX KIND MW <<<"$R"
    BEST="model/${PREFIX}_best.pt"
    [[ -f "$BEST" ]] || continue
    echo
    echo "--- ${PREFIX} ---"
    # Replay EVERY model-shaping training flag (the rebuild trap): clip floor,
    # block-FP, static exponent + slot count, exponent clamp.
    CS=(--checkpoint "$BEST" --n-hidden "$NHIDDEN"
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" --input-bit-width "$IBITS"
        --pmu-bit-width "$MW" --nobj 20)
    [[ -n "$CLIP_MIN" ]] && CS+=(--input-clip-min "$CLIP_MIN")
    case "$KIND" in
        static)  CS+=(--pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches 0 --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX") ;;
        fixed)   CS+=(--pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches "$FLOOR_BATCHES" --pmu-exp-fixed --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX") ;;
        staticfz) CS+=(--pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches 0 --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX") ;;
        staticf) CS+=(--pmu-block-fp --pmu-static-exp --pmu-exp-floor-batches "$FLOOR_BATCHES" --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX") ;;
        dynamic) CS+=(--pmu-block-fp --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX") ;;
    esac
    $PY scripts/check_scales.py "${CS[@]}" | grep -E "pmu_quant|input_quant" || true
done

echo
echo "Compare bgRej@0.5 (monotone) over AUC (16-20 epoch noise floor ~+-0.002);"
echo "only seed-matched rows are a controlled comparison."
