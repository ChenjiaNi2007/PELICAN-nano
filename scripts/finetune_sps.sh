#!/usr/bin/env bash
# SPS warm-start FINE-TUNE: start from a trained uniform-grid CONTROL checkpoint and
# fine-tune it with FIXED static per-slot exponents (data-derived floor table).
#
# Why: fixed-SPS trained from scratch lands in a bad basin (full data: train AUC 0.926
# vs control 0.940), yet the trained control evaluated with fixed-SPS momenta swapped in
# scores 0.9562 vs its native 0.9519. So keep the control's weights and grids and only
# let the weights adapt to the SPS momentum grid.
#
# --init-from is a WEIGHTS-ONLY warm start (src/trainer/utils.py warm_start_from): every
# key present in both models with a matching shape is copied; the source's uniform
# pmu_quant.act_quant.* keys and the target's SPS pmu_quant state are skipped (the SPS
# exponent is data-initialised on the first training batch). Optimizer, scheduler and
# epoch counter start fresh.
#
# Usage
#   INIT=model/sps_ctl_p12_s1_best.pt bash scripts/finetune_sps.sh
#
# Environment
#   INIT           REQUIRED: control checkpoint (uniform grid, same recipe/widths)
#   W              mantissa width, default 12        SEED    default 1
#   EPOCHS         default 5                          LR      --lr-init, default 0.00025
#   DECAY          --lr-decay-type, default flat
#   FREEZE         --freeze-scales-epoch, default 1 (keep the control's quantizer grids)
#   FLOOR_BATCHES  --pmu-exp-floor-batches K, default 16
#   THREADS        OMP/MKL threads, default 4         DATADIR default data/toptag20
#   TAG            inserted after 'sps_' in the prefix (no '-'); prefix
#                  sps_[TAG_]ft_p<W>_s<SEED>
#   WBITS/ABITS/IBITS  default 6/6/6 (must match the control)
#   NHIDDEN default 2   EXP_MIN/EXP_MAX default 0/10   CLIP_MIN default 512
#   DEVICE default --cpu   PY default .venv/bin/python   EXTRA extra train args
#
# LR schedule caveat (src/trainer/trainer.py): any run with num_epoch > 4 gets a 4-epoch
# linear warmup from 0; with DECAY=flat the cooldown start is not warmup-adjusted, so at
# EPOCHS=5 the run is 4 warmup epochs + 1 flat epoch at LR and the cooldown never fires.
#
# The recipe block below REPLICATES the COMMON block of scripts/sweep_pmu_static.sh
# (minus the per-run epochs/lr/decay/freeze, set explicitly here). KEEP THEM IN SYNC.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${INIT:?INIT=<control checkpoint> is required}"
W=${W:-12}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-5}
LR=${LR:-0.00025}
DECAY=${DECAY:-flat}
FREEZE=${FREEZE:-1}
FLOOR_BATCHES=${FLOOR_BATCHES:-16}
THREADS=${THREADS:-4}
DATADIR=${DATADIR:-data/toptag20}
TAG=${TAG:-}
WBITS=${WBITS:-6}; ABITS=${ABITS:-6}; IBITS=${IBITS:-6}
NHIDDEN=${NHIDDEN:-2}
EXP_MIN=${EXP_MIN:-0}; EXP_MAX=${EXP_MAX:-10}
CLIP_MIN=${CLIP_MIN-512}
DEVICE=${DEVICE:---cpu}
EXTRA=${EXTRA:-}
if [[ -z "${PY:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi
if [[ "$TAG" == *-* ]]; then echo "TAG must not contain '-' (got '$TAG')" >&2; exit 1; fi
[[ -f "$INIT" || "$PY" == "true" ]] || { echo "INIT checkpoint not found: $INIT" >&2; exit 1; }
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

# MODE=sps (default): fine-tune onto the static per-slot momentum grid.
# MODE=uniform: CONTROL fine-tune — identical warm start and recipe, but the momenta stay on
# the uniform --pmu-bit-width grid (all 21 source keys load). Isolates the grid from the
# extra low-LR epochs. Prefix sps_ftu_* so the two never overwrite each other.
MODE=${MODE:-sps}
case "$MODE" in
    sps)     FTTAG=ft ;;
    uniform) FTTAG=ftu ;;
    dynamic) FTTAG=ftd ;;   # per-jet block-FP (Lever 7) grid, same warm start — parity reference for SPS
    *) echo "MODE must be sps or uniform (got '$MODE')" >&2; exit 1 ;;
esac
PREFIX="sps_${TAG:+${TAG}_}${FTTAG}_p${W}_s${SEED}"

# --- recipe: keep in sync with COMMON in scripts/sweep_pmu_static.sh ---
RECIPE=(--datadir "$DATADIR" --target is_signal
        --nobj 20 --nobj-avg 49 --n-hidden "$NHIDDEN"
        --batch-size 256
        --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005
        --activation relu --batchnorm b
        --quant --po2-scales
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" --input-bit-width "$IBITS"
        --summarize-csv all --no-predict)
[[ -n "$CLIP_MIN" ]] && RECIPE+=(--input-clip-min "$CLIP_MIN")

if [[ "$MODE" == "sps" ]]; then
    GRID=(--pmu-bit-width "$W" --pmu-block-fp --pmu-static-exp
          --pmu-exp-floor-batches "$FLOOR_BATCHES" --pmu-exp-fixed
          --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX")
elif [[ "$MODE" == "dynamic" ]]; then
    GRID=(--pmu-bit-width "$W" --pmu-block-fp --pmu-exp-min "$EXP_MIN" --pmu-exp-max "$EXP_MAX")
else
    GRID=(--pmu-bit-width "$W")
fi
FT=("${GRID[@]}"
    --init-from "$INIT" --num-epoch "$EPOCHS" --lr-init "$LR" --lr-decay-type "$DECAY"
    --freeze-scales-epoch "$FREEZE" --seed "$SEED" --prefix "$PREFIX")

echo "=== SPS fine-tune: init=$INIT  W=$W  seed=$SEED  epochs=$EPOCHS  lr=$LR ($DECAY)  freeze-scales=$FREEZE  K=$FLOOR_BATCHES"
echo "    bits ${WBITS}/${ABITS}/${IBITS}  TAG='${TAG}' -> model/${PREFIX}_best.pt ==="
echo "+ $PY train_pelican_nano.py ${RECIPE[*]} ${FT[*]} $DEVICE $EXTRA"
# shellcheck disable=SC2086
$PY train_pelican_nano.py "${RECIPE[@]}" "${FT[@]}" $DEVICE $EXTRA

echo
echo "=== fine-tune summary (best_metrics from model/${PREFIX}_best.pt) ==="
BEST="model/${PREFIX}_best.pt"
printf '%-28s %-8s %-4s %-8s %-8s %s\n' "checkpoint" "grid" "W" "AUC" "acc" "bgRej@0.5"
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
printf '%-28s %-8s %-4s %s\n' "$PREFIX" "ft-fixed" "$W" "$METRICS"

if [[ -f "$BEST" ]]; then
    echo
    CS=(--checkpoint "$BEST" --n-hidden "$NHIDDEN"
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" --input-bit-width "$IBITS"
        --nobj 20 "${GRID[@]}")
    [[ -n "$CLIP_MIN" ]] && CS+=(--input-clip-min "$CLIP_MIN")
    $PY scripts/check_scales.py "${CS[@]}" | grep -E "pmu_quant|input_quant" || true
fi
