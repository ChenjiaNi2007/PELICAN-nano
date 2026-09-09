# h=2 vs h=4 retrain (capacity check) — method and log

Status: **complete** (37 runs, 2026-09-07 22:21 → 2026-09-09 03:31 EDT on the M1 laptop). Companion memory/context:
`nPELICAN-fpga/sweep/` (the original capacity sweep) and `results/roc_summary.csv` at the
workspace root (the rows that prompted this).

## Why

The professor questioned the h=4 point because the "float h=4" AUC (0.9267) sits below
"float h=2" (0.9639). Those two rows are not the same kind of object:

| roc_summary row | what it actually is | trained | best epoch |
|---|---|---|---|
| nanoPELICAN float h=2 (master weights) | **24-bit** QAT ckpt `fpga_model_qat_best.pt`, quantizers off | 8 ep, seed 178296088952660 | 5 |
| nanoPELICAN float h=4 (master weights) | **6/6/6-pmu12** QAT ckpt `nhid4_best.pt`, quantizers off | 8 ep, seed 42 | 3 |
| nanoPELICAN quant (w6a6i6 pmu12), h=2 | `fpga_model_qat_w6a6i6p12_best.pt` | 8 ep, seed 42 | 3 |
| nanoPELICAN quant h=4 | `nhid4_best.pt` | 8 ep, seed 42 | 3 |

A 24-bit quantizer is float for all practical purposes, so "float h=2" is a float model.
"Float h=4" is the master weights of a *6-bit* QAT run; the quantizers are part of that
model and removing them yields something that was never trained. Its quant forward gives
0.9573, above h=2's 0.9519. So the anomaly is an artefact of the row definitions, not
evidence against h=4. **No true float-trained h=4 model existed before this retrain.**

Two further reasons the old numbers are weak, found while setting this up:

1. **LR cooldown bug (fixed, `src/trainer/scheduler.py`).** `GradualCooldownScheduler`
   computed `lr = lr0 * 0.5 ** step` — halving every *minibatch*. With ~118+ minibatches
   per epoch the LR reached ~1e-40 within the first cooldown epoch, so the last 3 epochs of
   every `cos` run (last third of every `flat` run) trained nothing. Visible in every old
   log: epochs 6, 7, 8 print identical metrics. An "8-epoch" production run was really
   4 warmup epochs + 1 epoch at peak LR. That is why every production checkpoint selected
   its best epoch at 3–5 (inside warmup) and never saw an annealed LR. The fix uses the
   intended geometric decay to `--lr-final` over the cooldown window (the formula was
   sitting commented out under the bug); `tests/test_scheduler_cooldown.py` pins it.
2. **Single seed, 8 epochs.** The original authors trained nano for 140 epochs
   (`slurm/job_pelican.sbatch`). Seed-to-seed spread at 8 epochs was already noted as
   ~±0.01 AUC in the capacity sweep.

## Setup

- **Data** — `data/toptag20/{train,valid,test}.h5` = the full top-tagging set
  (`../train_c.h5` 1,211,000 jets / `valid_c.h5` 403,000 / `test_c.h5` 404,000) truncated
  to the first 20 constituents, float32 momenta, `Nobj`/`is_signal`/`truth_Pmu` verbatim.
  Built by `scripts/make_toptag20.py` (seconds). Training with `--nobj 20` keeps only
  `p[:20]` and casts to float32 before `dot4`, so inputs are **bit-identical** to loading
  the 200-constituent files (verified on random jets); RAM drops from ~10 GB to ~0.6 GB
  per process, which is what makes local training possible on the 8 GB laptop.
- **Recipe** — production, from `scripts/sweep_pmu_width.sh` / `nPELICAN-fpga/sweep/`:
  nobj 20, nobj-avg 49, batch 256, AdamW wd 0.005, dropout 0.05/0.05, ReLU, BN `b`,
  `cos` LR (4 warmup + cosine + 3 cooldown). `qat` = `--quant --po2-scales` w6 a6 i6
  pmu12 (the firmware operating point). `float` = no quantizers. `q24` = `--quant
  --po2-scales` w24 a24 i24, float momenta = the exact recipe of the old "float h=2"
  reference checkpoint. Its learned ranges (from `scripts/check_scales.py` on
  `fpga_model_qat_best.pt`) clip d_ij at 2048 GeV², the ReLU output at 2.0, the 2→0
  aggregate at ±1 and the logit at ±8 — saturations a pure float model does not have.
- **Driver** — `scripts/capacity_retrain.py` (grid → `train_pelican_nano.py` runs, N
  concurrent, per-epoch metrics via `--summarize-csv all`, LR captured from stdout,
  resumable). Queue: `scripts/capacity_retrain_queue.sh`. Curves/table:
  `scripts/training_curves.py [--watch]` → `results/capacity_retrain/`.
- **Compute** — CPU only. Brevitas fails on MPS (`NYI: named tensors only support CPU,
  CUDA`), and the float model is *slower* on MPS than CPU (kernel-launch bound). Measured
  on 140k jets, 4 threads: float h2 11.7 s/epoch, h4 14.1 s, qat h2 25.8 s, h4 33.7 s;
  3 concurrent 2-thread jobs give 1.4× the throughput of one 4-thread job. Full set
  (8.6×) with 3 concurrent jobs ≈ 3.5–4.5 min/epoch float, 8–10 min/epoch qat.

## Finding from Phase 1a: late power-of-two scale flips undo the anneal (QAT only)

With the cooldown working, the float runs improve monotonically through it. The QAT runs
do not: the 24-bit h=2 control peaked at epoch 5 (test AUC 0.9580) and ended at 0.9512,
and 6-bit QAT h=4 peaked at epoch 5 (valid AUC 0.9571) and ended near 0.9525. Comparing
the best and final checkpoints with `scripts/check_scales.py`:

| run | scale that moved between best (ep 5) and final (ep 8) | effect |
|---|---|---|
| q24 h=2 | `input_quant` 2⁻¹⁵ → 2⁻¹⁶ | d_ij clip 256 → **128 GeV²** — the documented collapse basin (`quant.py`) |
| qat h=4 | `pmu_quant` 0.25 → 0.125; `post_agg_quant` 0.125 → 0.0625 | momentum clip 512 → **256 GeV** (saturates leading constituents); T clip halved |

A po2 scale is a discrete parameter: it jumps when its continuous log-scale parameter
crosses a rounding boundary, and once the LR has annealed to ~1e-5 the weights can no
longer compensate. The trainer's best-by-valid-loss selection masks the loss but wastes
the anneal. Remedy under test: `--freeze-scales-epoch N` (new trainer flag; the `qatf`
driver mode sets N = E−2 so the scales are frozen for the cooldown while the weights keep
annealing). `tests/test_freeze_scales.py` pins the mechanics.

**Update (Phase 3, 2026-09-08 07:30):** the cooldown-only freeze (`qatf`, epochs 18–20)
is too late for h=4: seed 2 flipped `pmu_quant` 0.25 → 0.125 (512 → 256 GeV) and
`act_layer` 0.0625 → 0.03125 during epoch 17 (LR 6.8e-4), one epoch before the freeze,
and dropped from 0.9586 (best, epoch 16) to 0.9550 (final). Added `qatf12` (freeze from
epoch 12 = last 40 % of a 20-epoch run) as a second queue ahead of the LR variants
(`scripts/capacity_retrain_queue2.sh`).

**Update (qatf12 phase, 2026-09-08 19:00):** freezing from epoch 12 removes the scale
flips but 6-bit QAT validation still jumps late (h=2 seed 3: 0.9529 at epoch 19 → 0.9471
at epoch 20 with LR ≤ 1e-5). Diffing best (ep 16) vs final checkpoints: every quantizer
scale identical; `mixing.weight` master values moved by up to 0.21 against a 6-bit grid
step of 0.031 (several integer flips among only 12 weights), and `msg_2to0` BN running
variance drifted 23 %. Discrete rounding of a tiny weight set makes the quantized network
jump under weight motion that would be invisible in float. Consequence: for 6-bit QAT
report the best-by-validation-loss checkpoint (the trainer already keeps it); the
final-epoch number is the honest one only for float. h=4 (24 weights) jumps less.

## Matrix (queue order)

| phase | h | mode | lr | epochs | seeds | purpose |
|---|---|---|---|---|---|---|
| 1a | 2, 4 | float, qat | 0.0025 | 8 | 1 | old budget, fixed cooldown: is 8 epochs the problem? |
| 1b | 2, 4 | float, qat | 0.0025 | 20 | 1, 2, 3 | **headline**: proper budget, seed spread |
| 3 | 2, 4 | qatf (scales frozen in cooldown) | 0.0025 | 20 | 1, 2, 3 | **headline QAT** comparison |
| 1b′ | 2, 4 | qat as-is | 0.0025 | 20 | 1 | quantifies the freeze effect at 20 epochs |
| 3a | 2, 4 | qatf | 0.0025 | 8 | 1 | direct pair to the 1a qat runs |
| 2 | 2, 4 | float, then qatf | 0.001, 0.005 | 20 | 1 | LR sensitivity (last; stop the queue if not needed) |
| ctrl | 2, 4 | q24 | 0.0025 | 8 | 1 | old "float h=2" recipe (24-bit quantizers, float momenta): isolates the learned-clip effect from capacity |

Queue was reordered after Phase 1a (the original 1b included 3-seed `qat` without the
freeze); `scripts/capacity_retrain_queue.sh` is the current order.

Prefix: `cap_h{H}_{mode}_lr{lr}_e{E}_s{seed}` → `model/<prefix>_best.pt` (lowest
validation loss), `log/<prefix>.log`, `predict/<prefix>.metrics.{train,valid}.csv`.
Test metrics of the best checkpoint are the `Best  Testing` line of the log
(`log/cap.Best.metrics.csv` accumulates them).

## Results (20 epochs, peak LR 0.0025, 3 seeds; test AUC on 404k jets, 1/ε_B at ε_S = 0.30)

| config | best-by-loss AUC, mean ± sd | final-epoch AUC, mean ± sd | 1/ε_B (best) |
|---|---|---|---|
| float h=2 | 0.9561 ± 0.0052 (0.9532 / 0.9621 / 0.9530) | 0.9560 ± 0.0052 | 85 |
| float h=4 | 0.9657 ± 0.0003 (0.9656 / 0.9660 / 0.9655) | 0.9663 ± 0.0004 | 111 |
| QAT w6a6i6 pmu12, scales free (seed 1 only) | h=2 0.9525, h=4 0.9607 | h=2 0.9472, h=4 0.9583 | 71 / 92 |
| QAT, scales frozen ep 18–20 | h=2 0.9550 ± 0.0018, h=4 0.9595 ± 0.0011 | h=2 0.9549, h=4 0.9585 | 89 / 90 |
| **QAT, scales frozen from ep 12** | **h=2 0.9547 ± 0.0013, h=4 0.9618 ± 0.0011** | h=2 0.9528, h=4 0.9621 | 80 / 109 |

Peak-LR sensitivity (seed 1): float h=2 0.953 at all three LRs (basin set by the seed);
float h=4 0.962 (0.001) / 0.966 (0.0025) / 0.966 (0.005); QAT-frozen-12 h=2 0.958 / 0.955 /
0.958; h=4 0.957 / 0.962 / 0.961. Eight-epoch reruns with the fixed cooldown reproduce the
old QAT checkpoints (h=2 0.9521 vs 0.9519; h=4 0.9576 vs 0.9573).

**Conclusions.** (1) h=4 beats h=2 in float (+0.010 mean, +0.004 vs the best h=2 seed) and
at the firmware operating point (+0.007); h=4 is seed-stable, h=2 has two basins.
(2) The old "float h=4 < float h=2" was an artefact of comparing a 24-bit checkpoint with a
6-bit checkpoint's stripped master weights. (3) Recommended recipe: fixed cooldown, 20
epochs, peak LR 0.0025, `--freeze-scales-epoch 12` for QAT; quote QAT at its
best-by-validation-loss checkpoint (6-bit weight rounding keeps late epochs jumpy).
Full tables: `results/capacity_retrain/{summary,eval,best_by_config}.csv`; curves
`curves_{2,4}.png`; report `report.html`; per-jet logits `logits/*.npz`.
