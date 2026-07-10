# QAT_REFACTOR_PLAN.md — Brevitas Quantization-Aware Training for nanoPELICAN

Companion to `CLAUDE.md` (read it first — it contains the exact architecture math, the code
map, and the invariants). This document is the product/engineering spec: goals, design
decisions, a phased implementation plan with acceptance criteria, and open decisions.

## 1. Goal

Refactor PELICAN-nano so the nanoPELICAN model can be trained **from scratch with
quantization in the loop** (QAT / quantization-aware pre-training) using
[Brevitas](https://github.com/Xilinx/brevitas) (target version 0.12.x), producing models
whose weights and activations live on integer grids suitable for fixed-point FPGA inference
(the nPELICAN-fpga HLS implementation is the downstream consumer). The float training path
must remain available and unchanged.

Non-goals: changing the architecture's math, the dataset pipeline, or the trainer's overall
structure; post-training quantization (PTQ) is a nice-to-have calibration baseline, not the
deliverable; quantizing the `folklore` op, `Eq1to2`/`Eq2to1`, or `factorize=True` paths.

## 2. Requirements

**R1 — Functional parity switch.** A `--quant` flag (default off). Off → existing float
model, numerically identical to today (existing checkpoints still load). On → quantized
model. With quantization configured at high precision (e.g. 32-bit or quantizers disabled),
quant-model outputs must match the float model to ~1e-4 relative on sample data.

**R2 — Quantize every learned op and every tensor crossing a layer boundary** in the nano
path: input `d_ij`, the post-aggregation basis tensors of LinEq2→2 and LinEq2→0, the two
linear mixing layers (weights + biases), the hidden activation, and the output logit.
The 6 aggregation ops themselves are parameter-free sums/broadcasts; in QAT they run in the
de-quantized (fake-quant float) domain, with a quantizer applied to their stacked output —
this mirrors the FPGA design where accumulators widen and are then rescaled by 1/N̄.

**R3 — Configurable bit widths** via CLI: `--weight-bit-width` (default 8),
`--act-bit-width` (default 8), `--input-bit-width` (default 8 or 16; expose it — inputs are
heavy-tailed), `--bias-bit-width` (default 32 or None = unquantized float bias). Per-tensor
scaling for activations; per-tensor or per-output-channel for weights
(`--weight-per-channel`, default per-tensor to match the HLS code). Optional
`--po2-scales` to restrict all scale factors to powers of two (FPGA-friendly: shifts instead
of multiplies).

**R4 — Works for all model sizes.** `C_hidden ∈ {1, 2, 3, 10}` from the paper, and arbitrary
values. Nothing may hard-code `C_hidden`; parameter-count regression test: float nano with
one output, no BN = `10·C_hidden + 1` parameters.

**R5 — Symmetry preservation.** Quantization must not break permutation invariance or
masking. Scales are per-tensor/per-channel only — never indexed by particle. Padded entries
remain exactly zero through every quant layer (note: use symmetric/zero-centered activation
quantizers wherever masked zeros flow, so 0.0 is exactly representable; ReLU output quant is
unsigned with zero-point 0, which is fine).

**R6 — Export.** A script exporting a trained quant model to QONNX
(`brevitas.export.export_qonnx`) for the hls4ml/FINN-style downstream flow, plus a
weight-dump utility analogous to the existing FPGA repo's `model_loader.py` that emits
integer weights + scale factors.

**R7 — Tests.** Pytest suite covering: einsum↔Linear refactor equivalence, float-path
non-regression, high-precision quant parity (R1), permutation invariance (random
permutations of constituents leave the score unchanged, quant and float), masking (appending
zero-padded particles leaves the score unchanged), param counts, checkpoint round-trip,
QONNX export smoke test. Use `data/sample_data/` for end-to-end smoke tests; no GPU
required (guard/remove the `nvidia-smi` hard-exit in `train_pelican_nano.py` for CPU runs).

## 3. Design decisions

**D1 — Refactor einsum mixing into `nn.Linear` first, quantize second.**
`Eq2to2.forward`'s `einsum('dsb,ndbij->nijs', coefs, ops)` is exactly a Linear over the last
axis after reshaping ops to `[B,N,N,C_in*6]` with `W = coefs.permute(1,0,2).reshape(C_out,
C_in*6)` (verified numerically). Same for `Eq2to0` with basis 2. Step 1 of the refactor
replaces `self.coefs` with `nn.Linear(C_in*basis, C_out, bias=False)` (bias and diag_bias
stay separate Parameters in 2→2 because the diagonal bias is not expressible inside Linear;
in 2→0 the single bias can move into the Linear). Provide a converter that maps old
checkpoints (`coefs` tensors) into the new layout.

**D2 — Quantized modules by composition, not forking.** Add a `QuantConfig` dataclass
(bit widths, scaling impl, po2 flag, enabled flag) threaded through `PELICANNano` →
`Net2to2`/`Eq2to2`/`Eq2to0`. When enabled:
- `nn.Linear` → `brevitas.nn.QuantLinear` (weight quant: `Int8WeightPerTensorFloat` or
  per-channel / fixed-point variants per config; bias quant per `--bias-bit-width`;
  `return_quant_tensor=False` initially for simplicity).
- activation → `brevitas.nn.QuantReLU` (or `QuantIdentity` + existing activation if a
  non-ReLU is requested; for QAT prefer plain ReLU and say so in docs).
- New `QuantIdentity` instances at: model input (after `d_ij` computation and `--scale`
  pre-scaling), after each `eops_*` stack (post-aggregation re-quantization), and on the
  final logit.
- `bias` / `diag_bias` Parameters in Eq2to2: either leave float (folded at export) or wrap
  with a shared bias quantizer — default float, flag to quantize.

**D3 — Input handling.** `d_ij` spans many orders of magnitude. Default recipe:
keep `--scale` (collate-time multiplicative scaling) as the coarse knob, then a learned-scale
`QuantIdentity` (e.g. `Int8ActPerTensorFloat` with learned scale init from a few calibration
batches) at higher bit width (`--input-bit-width 16` recommended default for first
experiments). Document that input bit width is the dominant accuracy knob.

**D4 — BatchNorm.** Keep `MaskedBatchNorm2d` in float during QAT (standard practice), and
implement BN absorption into the following QuantLinear at export time (the paper §3 describes
exactly this absorption; biases become low-order polynomials in N only if absorbed fully —
for the default fixed-N̄ 's' config the absorption is a plain affine fold). Alternatively
support `--batchnorm None` QAT runs; both must work.

**D5 — The `N^α/N̄^α` scaling ('S'/'M' configs, nPELICAN_N).** Data-dependent non-integer
powers don't quantize. Policy: (a) default QAT supports `config='s'` only and raises a clear
error otherwise; (b) optional follow-up mode keeps the α-multiplier as a float side-channel
applied between de-quant and re-quant (legitimate in fake-quant training, but flag in docs
that the FPGA must then implement an N-indexed lookup table of multipliers — feasible since
N ≤ Nmax). Implement (a) now, stub (b) behind `--allow-alpha-scaling`.

**D6 — Dropout** stays as-is (it interacts fine with fake quantization; it's identity at
eval).

**D7 — Trainer/loss untouched** except: register new CLI args, pass QuantConfig into model
construction, and make sure `init_weights` skips Brevitas internal parameters/quant proxies.
Recommended-but-optional: a `--qat-warmup-epochs` that trains in float then enables
quantizers (Brevitas supports toggling; if messy, document two-stage training via
`--load` instead).

## 4. Phased plan (each phase = reviewable commit(s) + green tests)

**Phase 0 — Hygiene + baseline.**
Remove debug `print()`s from `PELICANNano.forward`; make the `nvidia-smi` check non-fatal on
CPU; add `requirements.txt` (torch, brevitas==0.12.*, h5py, scikit-learn, pytest, ...);
scaffold `tests/` with a fixture that builds the model and runs a forward pass on
`data/sample_data/`; add float-model tests: param count (`10·C_h + 1` config), permutation
invariance, masking invariance. Capture a seeded float forward-pass golden output for
non-regression.

**Phase 1 — Linear-ization refactor (float only).**
Implement D1 in `Eq2to2` and `Eq2to0`. Add `scripts/convert_checkpoint.py` (old `coefs` →
new Linear weights). Tests: golden-output non-regression vs Phase 0 capture; old-checkpoint
conversion round-trip.

**Phase 2 — Quant modules.**
Add `src/layers/quant.py` (QuantConfig + quantizer factory honoring bit widths, per-channel,
po2). Wire QuantLinear/QuantReLU/QuantIdentity per D2/D3 into `Eq2to2`, `Eq2to0`,
`PELICANNano` behind `quant_config`. Tests: high-precision parity (R1), permutation/masking
invariance with 8-bit quant, all C_hidden sizes, `config='S'` raises per D5.

**Phase 3 — Training integration.**
New CLI args (R3 + `--quant`); thread through `train_pelican_nano.py`; checkpoint
save/load including quantizer state; short smoke training run (few batches on sample data,
CPU) asserted in tests to produce finite loss and decreasing loss trend.

**Phase 4 — Export + docs.**
`scripts/export_qonnx.py` (R6) + integer weight/scale dump; README section documenting the
QAT workflow, recommended starting hyperparameters (8-bit weights/acts, 16-bit input,
`--batchnorm b`, ReLU, `config s`, same schedule as float paper run), and known accuracy
knobs. Final full test pass.

## 5. Acceptance criteria

1. `python train_pelican_nano.py ... ` without `--quant` reproduces today's behavior
   (golden-output test passes; old checkpoints load).
2. `--quant --weight-bit-width 8 --act-bit-width 8 --input-bit-width 16 --n-hidden 2`
   trains end-to-end on sample data on CPU without NaNs.
3. All R7 tests pass; QONNX export produces a loadable model with quant annotations.
4. Documented expectation: on the full top-tagging dataset, the 8-bit C_h=2 model should
   land within a few ×0.001 AUC of the float 0.9718 baseline (to be verified by the user on
   GPU; not a CI gate).

## 6. Open decisions for the user (defaults chosen, flag if you disagree)

- Bias quantization default: float biases, fold at export (vs. 32-bit quantized biases).
- Per-tensor weight scales by default (matches single shared shift in HLS) vs per-channel.
- Whether to pursue D5(b) α-scaling support now or later (default: later).
- Whether the output should stay the `cat([-w, w])` two-logit form (kept for trainer
  compatibility) or be simplified to a single logit + BCEWithLogits (out of scope by default).
