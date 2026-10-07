# hls4ml 5-class jet tagging (g/q/W/Z/t) — comparison with the MLP-Mixer paper

Status: **data converter + 5-class head implemented 2026-10-06; phase-1 float runs COMPLETE 2026-10-07 02:59 (18 runs); phase 1b (jet spurion + head, 40 ep) COMPLETE 2026-10-07 09:47: 71.8% at N=32 AND N=16; firmware NOUT=5 generalized and gated bit-exact on a smoke QAT ckpt.**

## Why

The professor wants a like-for-like comparison with arXiv:2503.03103 ("Fast Jet Tagging
with MLP-Mixers on FPGAs", Sun, Ngadiuba, Pierini, Spiropulu). Our earlier MLP-Mixer port
(workspace `mlpmixer/`) moved the Mixer onto *our* binary toptag task; this goes the other
way: nanoPELICAN on *their* dataset, split, input truncation and metrics (per-class
one-vs-rest AUC, TPR at FPR = 10% and 1%, accuracy), so Table 2 numbers can be quoted
directly next to ours.

## Dataset

hls4ml LHC jet dataset (150-constituent version), at the workspace root:

- `train 2/` — 62 files, 620,000 jets; `val 2/` — 26 files, 260,000 jets.
- Per file: `jetConstituentList (10000, 150, 16)` with columns
  `[px, py, pz, E, Erel, pT, pTrel, eta, etarel, etarot, phi, phirel, phirot, dR, costheta, costhetarel]`;
  the label is the one-hot `jets[:, -6:-1]` in g/q/w/z/t order.
- Facts: constituents pT-sorted, zero-padded, ~massless. Multiplicity min 8 (test) / 9 (train) / median 46 /
  max 150. Jet pT ~1 TeV (≈2x the toptag set). d_ij at N=16: p99.9 = 732 GeV², max 7.7e3.

Converted files (`scripts/make_hls4ml5.py`): `data/hls4ml5_n{16,32}/{train,valid,test}.h5`
with `Pmu (n, nobj, 4) float32` as (E, px, py, pz), `Nobj int16` (untruncated
multiplicity), `label int8` (g=0, q=1, w=2, z=3, t=4), `is_signal int8 = (label == 4)`.

## Preprocessing convention

Paper Table 2 (the MLPM-fp headline numbers): **top-N constituents by pT, no pT cut.** That
is the default. Sec. 5.5 (and the DeepSet / l1-jet-id convention) applies a pT ≥ 2 GeV
constituent cut first; that variant is available via `--pt-min 2` but is not the
comparison target. We only use the 4-momentum columns: nanoPELICAN's sole input is the
Minkowski Gram matrix d_ij, whereas the Mixer uses 16 features per constituent.

## Split

- `test.h5` = all 260,000 jets of `val 2/`, in file order (the paper also tests on `val`).
- From the 620,000 `train 2/` jets (concatenated in sorted file order): global index
  `% 10 == 0` → `valid.h5` (62,000), the rest → `train.h5` (558,000).
- Same sizes as the paper's 558k / 62k / 260k (their exact split is not published).

## How to build

```bash
.venv/bin/python scripts/make_hls4ml5.py --src-train "../train 2" --src-val "../val 2" \
    --out data/hls4ml5_n16 --nobj 16
.venv/bin/python scripts/make_hls4ml5.py --src-train "../train 2" --src-val "../val 2" \
    --out data/hls4ml5_n32 --nobj 32
```

## Model change

`PELICANNano(..., n_out=K)` / CLI `--n-out K --target label`. With K > 1 the 2→0 layer
emits K raw logits (`{'predict': (B, K)}`) trained with cross-entropy; K = 1 keeps the
binary `cat([-w, w])` path bit-for-bit. Only the 2→0 output channel count changes; the
2→2 block, BN and aggregation are untouched. Params (no BN) = 8h + K(2h+1):
h=2 → 41, h=4 → 77, h=8 → 149 for K = 5 (BN `b` adds 2 + 2h). Compare MLPM-fp: 2,465
(N=16) / 3,265 (N=32). Metrics: `src/models/metrics_multiclass.py`
(`compute_multiclass_metrics`, softmax one-vs-rest). Firmware NOUT=5 is a follow-up.

## Training

Recipe = production toptag recipe (`scripts/capacity_retrain.py::RECIPE`) with the 5-logit
head: nobj-avg 49, batch 256, wd 0.005, dropout 0.05/0.05, ReLU, BN `b`, 20 epochs,
lr 0.0025 `cos`, CPU, `--no-predict --summarize-csv all`. QAT: `--quant --po2-scales`
w6 a6 i6 pmu12 `--freeze-scales-epoch 12`.

Matrix: `scripts/hls4ml5_queue.sh` (one run per line; `MAX_PARALLEL=3`, `THREADS=2`;
bash-3.2-safe). Phase 1 float N ∈ {16, 32} × h ∈ {2, 4, 8} × seeds {1, 2, 3}
(`h5n{N}_h{h}_float_s{seed}`); phase 2 QAT (`h5n{N}_h{h}_qatf12_s{seed}`) commented out.

Cost (8 GB M1): toptag N=20 float h=2 is ~5–6 min/epoch on 1.2M jets; 558k jets and
cost ∝ N² give ~1.7 min/epoch at N=16 (~35 min/run) and ~7 min/epoch at N=32 (~2.3 h/run).
Phase 1 ≈ 30 CPU-h ≈ 10–12 h wall at 3 parallel jobs. The paper trains 500 epochs
(Adam 5e-3, cosine restarts every 100, batch 512, best val accuracy); we do not.

## Evaluation

```bash
.venv/bin/python -u scripts/eval_hls4ml5.py --checkpoint model/h5n16_h2_float_s1_best.pt \
    --print-paper --tag h5n16_h2_float_s1
```

Test split in file order through the trainer's own `collate_fn`; QAT checkpoints get the
train-mode-forward reload dance. Writes `../results/hls4ml5/<stem>_test_logits.npz`
(`logits (n,5) float32`, `label int8`, `nobj int16` = constituents seen after truncation)
and appends a row to `../results/hls4ml5/summary.csv` (accuracy, macro AUC, per-class AUC,
TPR@FPR 1% / 10%). `--print-paper` prints the matching MLPM-fp row and JEDI-net.

## Paper reference (arXiv:2503.03103 Table 2, MLPM-fp, 16 features; percent, g/q/W/Z/t)

| model | N | AUC | TPR@FPR10% | TPR@FPR1% | params | acc (quant, T4) |
|---|---|---|---|---|---|---|
| MLPM-fp | 16 | 94.08 / 92.15 / 96.06 / 95.34 / 96.14 | 82.2 / 77.8 / 89.5 / 86.4 / 90.8 | 40.8 / 25.9 / 51.5 / 66.4 / 60.3 | 2,465 | 77.5 |
| MLPM-fp | 32 | 95.08 / 93.09 / 97.49 / 97.12 / 96.91 | 85.3 / 80.3 / 93.1 / 91.2 / 92.5 | 44.9 / 28.1 / 70.2 / 77.6 / 64.2 | 3,265 | 80.7 |
| MLPM-fp | 64 | 95.53 / 93.43 / 97.83 / 97.50 / 97.13 | 86.7 / 81.1 / 93.8 / 92.2 / 93.0 | 46.9 / 28.2 / 75.4 / 80.9 / 64.7 | 6,401 | 81.6 |
| MLPM-fp | 128 | 95.48 / 93.32 / 97.78 / 97.43 / 97.01 | 86.6 / 81.0 / 93.7 / 92.0 / 92.7 | 46.6 / 28.0 / 74.8 / 80.5 / 63.6 | 18,817 | 81.3 |
| JEDI-net | 100 | 95.29 / 93.01 / 97.39 / 96.79 / 96.83 | — | — | — | — |

## Results (phase 1 float, 20 epochs, 3 seeds each; 260k test jets) — 2026-10-06 (N=16 unless marked)

| h | params | accuracy | macro AUC | AUC g / q / W / Z / t |
|---|---|---|---|---|
| 2 | 47 | 53.57 ± 0.17% | 81.36 ± 0.21% | 77.8 / 84.1 / 78.6 / 78.5 / 87.8 |
| 4 | 87 | 54.64 ± 0.21% | 82.35 ± 0.08% | 79.1 / 85.2 / 79.8 / 79.5 / 88.2 |
| 8 | 167 | 55.71 ± 0.18% | 83.23 ± 0.10% | 80.3 / 86.3 / 81.1 / 80.1 / 88.3 |
| **N=32** h=2 | 47 | 57.79 ± 0.46% | 82.86 ± 0.23% | 78.3 / 82.4 / 81.0 / 83.7 / 89.0 |
| **N=32** h=4 | 87 | 60.51 ± 0.48% | 85.36 ± 0.35% | 81.7 / 85.4 / 84.5 / 85.4 / 89.8 |
| **N=32** h=8 | 167 | 61.86 ± 0.21% | 87.00 ± 0.25% | 83.4 / 87.2 / 87.2 / 86.7 / 90.4 |
| **N=32 h=4 + jet spurion + head(16), 40 ep** | 271 | **71.80 ± 0.49%** | **91.17 ± 0.09%** | 87.7 / 88.8 / **94.3** / **93.5** / 91.5 |
| **N=16 h=4 + jet spurion + head(16), 40 ep** | 271 | 71.75% (1 seed) | 91.17% | 87.8 / 88.9 / 94.2 / 93.7 / 91.3 |
| N=16 h=4 jet+head, 24-bit QAT | 271 | 71.2% | 91.1% | 87.7 / 89.0 / 94.4 / 93.5 / 91.1 |
| N=16 h=4 jet+head, 6-bit QAT, ONE input scale | 271 | 53.5% | 82.0% | 70.7 / 83.1 / 83.2 / 84.9 / 88.0 |
| N=16 h=4 jet+head, 6-bit QAT, split jet quantizers @6 bit | 271 | 56.1% | 82.9% | 74.0 / 83.1 / 83.7 / 85.9 / 87.9 |
| **N=16 h=4 jet+head, 6-bit QAT, split @ 10/16/20 bit** | 271 | **69.9%** (s2), 67.0% (s1) | 89.3% | 82.6 / 87.9 / 93.4 / 92.5 / 90.6 |
| N=16 h=4 jet only, 6-bit QAT, split @ 10/16/20 bit | 87 | 61.0% | 85.3% | 77.6 / 84.7 / 88.7 / 87.9 / 87.9 |
| MLPM-fp N=16 (paper) | 2,465 | (quant 77.5%) | — | 94.1 / 92.2 / 96.1 / 95.3 / 96.1 |

Rows in `../results/hls4ml5/summary.csv`; logits npz alongside. Plain runs: best epochs 16–20 of 20 (not
converged). Jet+head (phase 1b, 2026-10-07): +10 points over the best plain model; W/Z AUC 0.87→0.94; the
remaining gap to the Mixer is g/q (0.88/0.89 vs 0.95/0.93). TPR@FPR10% g/q/W/Z/t 60.7/70.6/87.8/86.1/80.0;
TPR@FPR1% 19.1/24.3/38.5/46.3/36.6 (paper N=32: 85.3/80.3/93.1/91.2/92.5 and 44.9/28.1/70.2/77.6/64.2).
Ablations at N=32 h=4, 40 epochs, 1 seed each (test accuracy / macro AUC / AUC W, Z):
plain@40ep 60.0% / 85.4 / 84.4, 85.8 (= plain@20ep: the longer schedule alone buys nothing);
head only 63.8% / 88.3 / 89.4, 88.8 (+3.8); jet only 68.8% / 89.2 / 91.5, 91.2 (+8.8);
jet + head 71.8% / 91.2 / 94.3, 93.5 (+11.8, super-additive: the head needs the full-jet features to
carve the W/Z windows). **N=16 jet+head (1 seed): 71.75% / 91.2 / 94.2, 93.7 — identical to N=32 jet+head.** Once the full-jet
4-momentum is supplied, the truncation level stops mattering: the firmware-sized N=16 model matches N=32.
Confusions (h=4): g↔q 16–24%, Z→W 36%, t→W 17%; pairwise AUC g/q 0.77, W/Z 0.67.

### Why the gap (measured with histogram-Bayes baselines on the same jets)
| features | accuracy |
|---|---|
| mass of the 16 kept constituents | 51.4% |
| + raw multiplicity (never visible at N=16: 99% of jets have > 16) | 69.1% |
| + FULL-jet mass (all constituents) | 73.6% |
| full-jet mass alone | 67.7% |

So the N=16 nanoPELICAN is a slightly-better-than-mass classifier (55% vs 51%), and the paper's
Mixer gets its edge from FULL-JET information smuggled in through `Erel`/`pTrel`/`ΔR` (relative
to the whole jet): mass(16)+full mass alone reaches 73.6%, within 4 points of the quantized
Mixer. Truncation to 16 keeps ~69% of the mass (W/Z peaks 66/74 GeV instead of 81/91) and
erases multiplicity (g 64 vs q 35 constituents). Width buys ~1 point per doubling; N=16→32 at h=2
buys +4.2 points (Z AUC +5, W +2.4, t +1.2; g/q flat — 78% of jets still exceed 32 constituents).

### Recommended next steps (in order)
1. **Full-jet 4-momentum as a 3rd spurion** (Lorentz-invariant; gives m_jet, p_i·p_jet, captured
   fraction). Converter stores `Pjet` from all 150 constituents; collate appends it; firmware
   NPARTICLES2 = N+3. Re-derive the input-quant range (p_jet² up to ~1e5 GeV²).
2. N=32 (queued) / 64. 3. Small nonlinear head after 2→0 (2h→16→5; negligible firmware cost).
4. 60–100 epochs. 5. Pseudo-log d_ij encoding (firmware `psloglut` exists). 6. Second 2→2 layer.

## Open questions

- **nobj-avg.** 49 = firmware `invnave`, kept for comparability with toptag. True mean
  multiplicity after truncation is ~16 (N=16) / ~29 (N=32); one seed with
  `--nobj-avg 16` / `29` would show whether the normalization matters (queue script note).
- **QAT clip ranges must be re-derived.** Jet energies are ~2x toptag, so the learned
  d_ij / pmu clips (e.g. the 512 → 1024 GeV pmu clip finding) will not transfer;
  run `scripts/check_scales.py` on the first QAT checkpoint before launching phase 2.
- **Capacity.** h=2 was Pareto-optimal for binary toptag; 5 classes may need h=4–8.
- **Firmware.** DONE 2026-10-06 (uncommitted in nPELICAN-fpga): `NOUT`/`NHIDDEN`/`NPARTICLES`
  come from the checkpoint via `types_generated.h`; 2→0 weights `w2_2to0[o*2H + h*2 + a]`;
  `model_out[NOUT]`; golden gate reads NOUT logits/line. Gated bit-exact 200/200 (golden +
  dots-level, monolith == split) on `model/fwsmoke_h5n16_h2_qat_best.pt` after three loader
  fixes (nobj clamp, `--bias-guard-bits`, `--agg-guard-bits`); see
  `nPELICAN-fpga/reports/hls4ml5_smoke_gate/GATE.md` for the exact export/gate commands.
  2026-10-07: firmware extended to the jet spurion (`NSPURIONS=3`, `jet_input[4]` port, slot 2) and the
  head stage (`NPELICAN_HEAD K`: `Rq=(relu0_t)ReLU(Rp)`, `Hp = b_head + w_head·Rq` in `mach_t`), all
  generated from the checkpoint; gated 200/200 bit-exact on a synthetic 6-bit jet+head checkpoint, legacy
  build byte-identical (`nPELICAN-fpga/reports/hls4ml5_smoke_gate/GATE_JET_HEAD.md`). Accumulator headroom
  now follows the checkpoint's N (was fixed at 20). Gate on the REAL 24-bit jet+head checkpoint (`q24j16_h4_jh_e40_s1`, test acc 71.2%):
  dots-level max 3.4e-5 (network verified; float32-noise class on a 2^-21 grid), momenta-level 1.02e-3 (dot4
  float32 caveat, 4x toptag because the jet carries up to 5.3 TeV) — `GATE_Q24_JET_HEAD.md`.
  **6-bit QAT of the jet model FAILS with the production recipe**: `qat6j16_h4_jh` 53.5% (best epoch 6),
  and `qat6j16_h4_jet` (jet only) 50.8%,
  i.e. the whole jet gain is lost, because ONE per-tensor input scale must span particle-particle dots
  (median 1 GeV²) and jet dots (p_i·p_jet median 222, m_jet² median 7300, max 1.6e5). Fix in progress:
  `--jet-quant-split` = separate learned input quantizers for the jet row/column and for m_jet² (and a
  separate momentum grid for the jet 4-vector); legal because the spurion slot is fixed. Result of the split at ALL-6-bit widths (`qat6sj16_h4_jh_e40_s1`): 56.1% / mAUC 0.829 — only +2.6 over the
  single-scale probe, as the resolution arithmetic predicts (6-bit m_jet² LSB ≈ 1000 GeV², 12-bit jet momentum).
  Phase 1c (`scripts/hls4ml5_queue3.sh`, prefixes `qat6wj16_*`): split with WIDE jet grids 10/16/20 bits —
  RESULT: jet+head 69.9% / mAUC 0.893 (seed 2, best ep 25) and 67.0% (seed 1), jet-only 61.0% — vs 71.8%
  float and 53.5% for the single-scale recipe. The quantization gap is now 2–5 points, from 18.
  Per-class AUC (seed 2) g/q/W/Z/t 82.6/87.9/93.4/92.5/90.6.
  **Firmware gate on this real checkpoint** (`nPELICAN-fpga/reports/hls4ml5_smoke_gate/qat6_wide_split_jet_head/GATE.md`):
  199/200 bit-exact with the DSP-friendly 13-bit BN1 literal (the one residual is a 1.7e-5-LSB jdotp tie moved by
  the literal's 6.6e-5 relative error — the documented `--bn-frac-bits` trade-off), 200/200 bit-exact with --bn-guard-bits 8 (wider BN1 literal, DSP/timing cost per the Lever-8 measurements).
  Two loader fixes came out of it: per-quantizer jet widths must be replayed into QuantConfig (else 6/6/12-bit jet
  types → 51/200), and `--norm-guard-bits 8` for the non-po2 1/N̄ literals (a 1.4e-5 relative error flipped a tie
  5.9e-5 LSB away). Still owed: remote csynth/vsynth
  for the NOUT=5, N=16 (and N=32: NPARTICLES2=34 now fits `nobj_t`) builds.
- **pT cut variant.** Whether to also report the `--pt-min 2` (Sec. 5.5 / DeepSet) numbers.

## Improvements under test (2026-10-07)

Both are opt-in; with the flags off the model, collate and state-dict keys are identical to
before (golden float test + param-count tests enforce it). Tests: `tests/test_jet_spurion_head.py`.

### `--add-jet`: full-jet 4-momentum as a third spurion
- **Why.** A histogram classifier on (mass of the 16 kept constituents) gets 51.4%; adding the
  FULL-jet mass lifts it to 73.6% — more than the trained N=16/N=32 models (55% / 60.5%). The
  leading-N truncation throws away ~31% (N=16) / ~7% (N=32) of the jet mass and all multiplicity
  information; the model's only input (the Gram matrix of its own particles) cannot recover it.
- **What.** The converter now always writes `Pjet (n, 4) float32` = Σ (E,px,py,pz) over ALL
  constituents of the jet (after `--pt-min`, before truncation; attr `pjet_rule`). With
  `--add-jet`, `collate_fn(add_jet=True)` inserts `Pjet*scale` as ONE extra particle at slot 2:
  slot layout `[beam+, beam−, jet, constituents…]`, `Nobj` += 3, pdg placeholder 2212 (like a
  beam), masks from `E != 0` as before (jet E > 0 always). Requires `--add-beams`; `Pjet` stays in
  the batch as (B, 4). The new Gram entries are m²_jet and p_i·p_jet (each constituent's share of
  the jet), all Lorentz-invariant; permutation invariance over constituents is unchanged.
- **Datasets must be rebuilt** with the updated `scripts/make_hls4ml5.py` (old files have no
  `Pjet`; collate raises `KeyError`). Use new directories so running jobs are not disturbed:
  `--out data/hls4ml5j_n16 --nobj 16` and `--out data/hls4ml5j_n32 --nobj 32`.
- **Firmware later.** NPARTICLES2 = N + 3 (one more input row/column); the jet row is an input
  like the constituents (not a constant beam). Re-derive the input-quant/pmu ranges: p_jet² and
  p_jet·p_i are much larger than constituent dots (m_jet² up to ~1e5 GeV²; E_jet ~ 1 TeV).

### `--head-hidden K`: nonlinear output head
- **Why.** Today the K logits are a single linear map of the 2h pooled invariants (sum, trace per
  channel), so the head cannot form windows (e.g. "mass near 80 vs near 91" for W vs Z).
- **What.** `--head-hidden K > 0`: the 2→0 layer mixes to K channels with `activate_lin=True`
  (ReLU; `QuantReLU` under QAT), dropout as before, then `head = Linear(K, n_out)`
  (`QuantLinear` with the weight quantizer and float bias under QAT), then `output_quant`.
  `--head-hidden 0` (default) = exactly today's structure.
- **Cost.** Params (no BN) = 8h + K(2h+1) + n_out(K+1); e.g. h=4, K=16, n_out=5: 32+144+85 = 261.
  Firmware: (2h+1)K + (K+1)·n_out MACs on per-event scalars after the 2→0 pooling — a few
  hundred multiplies, negligible next to the N² front end.

### `--jet-quant-split`: separate quantizers for the jet-spurion populations
- **Why.** With `--add-jet`, `d_ij` mixes pair dots (median 1 GeV², p99 270) with jet dots
  p_i·p_jet (median 222, p99 3200, max 2.2e4) and m²_jet = d[2,2] (median 7300, max 1.6e5); Pmu
  mixes constituents (≤ 2.5 TeV) with E_jet (≤ 5.3 TeV). One per-tensor 6-bit `input_quant` is set
  by the jet dots (`qat6j16_h4_jh`: scale 512 GeV²) and rounds the pair dots to 0: 72% float → 53.5% QAT.
- **What.** Requires `--quant --add-jet` (trainer raises otherwise; the model ASSUMES slot 2 is the
  jet). Adds `input_quant_jet` (d[2,j], d[i,2], i,j≠2 — particle-jet and beam-jet), `input_quant_mjet`
  (d[2,2]) and, with `--pmu-bit-width`, `pmu_quant_jet` (Pmu row 2). Each sees only its own slice
  (jet row/col zeroed before `input_quant`/`pmu_quant`), so each learned scale fits its population;
  the masked combine is a plain tensor. Fixed spurion slots → permutation invariance over
  constituents is exact. Off → state dict unchanged. Not with `--pmu-block-fp` (NotImplementedError).
- **Independent jet widths.** At a shared 6 bits `input_quant_mjet`'s LSB on m²_jet (median 7300,
  max 1.6e5 GeV²) is ~1000 GeV² vs a ~1800 GeV² W/Z mass² gap; a 12-bit jet momentum grid (LSB 2 GeV
  on a 5 TeV vector) puts ~2·E·δE ≈ 1e4 GeV² of error into m² = E² − p². These are ONE row and ONE
  scalar per jet, so wider grids are free in hardware: `--jet-input-bit-width` (input_quant_jet),
  `--mjet-input-bit-width` (input_quant_mjet), `--jet-pmu-bit-width` (pmu_quant_jet; needs
  `--pmu-bit-width`). Default None = inherit `--input-bit-width`/`--pmu-bit-width`; all need
  `--jet-quant-split`. Recommended **10 / 16 / 20**. Model-shaping: rebuild tools replay them.
- **Firmware (not yet implemented).** Three dot types `dot_t` / `dotj_t` / `dotm_t` (pair / jet
  row+col / m²_jet) and a `jet_t` momentum type for the jet row. Tests: `tests/test_jet_quant_split.py`.
