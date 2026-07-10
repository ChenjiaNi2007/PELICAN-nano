# CLAUDE.md — PELICAN-nano (Brevitas QAT refactor)

This file gives you (Claude Code) the architecture math, code map, and invariants you must
preserve while refactoring this repository to support Quantization-Aware Training (QAT) with
Brevitas. The full task spec is in `docs/QAT_REFACTOR_PLAN.md`. Read both before editing code.

## What this repo is

PELICAN-nano (nPELICAN) is an ultra-small (11–108 parameter) Lorentz- and
permutation-invariant top-quark jet tagger (arXiv:2310.16121, built on PELICAN,
arXiv:2211.00454). The end target for the quantized model is FPGA inference (a sibling HLS
repo, nPELICAN-fpga, implements the same math in fixed point), so QAT must produce
integer-friendly arithmetic.

## The exact math (do not change it)

Input: per event, N 4-momenta p_i (zero-padded to Nmax; with `--add-beams`, two beam vectors
(1,0,0,±1) are appended, so the array is (2+Nobj)×(2+Nobj)). The only network input is the
Gram matrix of Minkowski dot products `d_ij = p_i · p_j`, shape `[B, N, N, 1]`. Masks:
`particle_mask [B,N]`, `edge_mask [B,N,N]`. `nobj = particle_mask.sum(-1)`.

Model = LinEq2→2(nano) → activation → LinEq2→0, plus optional masked BatchNorm
("messaging" layers that reduce to BatchNorm here) and Dropout before each equivariant block.

**LinEq2→2 (nano), 6 aggregators.** Because d is symmetric and constituents are massless
(d_ii = 0), the 15 general permutation-equivariant basis ops reduce to 6. With
`s = colsum_j = Σ_i d_ij / N̄` and `S = totalsum = Σ_ij d_ij / N̄²` (N̄ = `average_nobj` = 49,
a fixed hyperparameter — note: division by the *constant* N̄, not by the per-event N, when
`config='s'`), the ops produced by `eops_2_to_2` are, in code order:

1. `d_ij` (identity)
2. `diag_embed(colsum)`  → (J·p_i) δ_ij in physics terms
3. `colsum` broadcast across rows → output[i,j] = colsum[j] (J·p_j)
4. `colsum` broadcast across cols → output[i,j] = colsum[i] (J·p_i)
5. `totalsum` broadcast to all ij → m²_J / N̄²
6. `diag_embed(totalsum)` → m²_J δ_ij

These 6 maps per input channel are mixed by a learned tensor `coefs [C_in, C_out, 6]` via
`einsum('dsb,ndbij->nijs', coefs, ops)`, then `+ bias[c]` (everywhere) `+ diag_bias[c]` (on
the diagonal only). **Verified fact:** this einsum is *exactly* `nn.Linear` acting on the
last axis of `ops` reshaped to `[B, N, N, C_in*6]`, with weight
`W = coefs.permute(1,0,2).reshape(C_out, C_in*6)` (max abs diff ~1e-6 in float32). This is
the intended path to `brevitas.nn.QuantLinear`.

**Activation.** Paper uses ReLU; code default arg is `leakyrelu` (configurable via
`--activation`). Applied when `activate_lin=True` (after the linear mix).

**LinEq2→0, 2 aggregators.** `op1 = Σ_ij T_ij / N̄²` (total sum), `op2 = Σ_i T_ii / N̄`
(trace). Mixed by `coefs [C_hidden, C_out, 2]` (`einsum('dsb,ndb->ns')` — same Linear
equivalence with `[C_hidden*2]` input) plus one bias per output channel. `C_out = 1`; the
model emits `prediction = cat([-w, w])` and tags top if w > 0.

**Parameter count check** (use as a regression test): with one output channel and no
BatchNorm, params = 6·C_h (coefs) + C_h (bias) + C_h (diag_bias) + 2·C_h (2→0 coefs) + 1
(2→0 bias) = **10·C_hidden + 1**. Paper: C_h = 2 → 21 nominal, 19 effective.

**nPELICAN_N variant (`config='S'`/`'M'` etc.).** Aggregators become means scaled by a
*learnable, data-dependent* factor `N^α / N̄^α` (per channel, per aggregator α). This is the
highest-performing variant but the `N^α` power is hostile to integer arithmetic — see plan
for how to handle it. Default `config='s'` (plain N̄-normalized sums) is the primary QAT
target.

**BatchNorm (`--batchnorm b`).** `MaskedBatchNorm2d` inside `MessageNet` before each
equivariant block (masked so zero-padding doesn't pollute statistics). The paper notes BN
weights can be absorbed into the following LinEq for inference. For QAT: keep BN in float
during training, fold/absorb at export.

## Code map

- `train_pelican_nano.py` — entrypoint; builds args → datasets → `PELICANNano` → `Trainer`.
- `src/models/pelican_nano.py` — `PELICANNano`. Computes `d_ij` via `dot4`, calls
  `Net2to2([1, n_hidden], ...)` → `MessageNet([n_hidden])` (BatchNorm only) → `Eq2to0(n_hidden, 1)`.
  ⚠ Contains stray debug `print()` calls in `forward` (jet mass / is_signal) — remove them.
- `src/layers/perm_equiv_layers.py` — `eops_2_to_2` (the 6 nano ops; 15-op general version is
  commented out), `eops_2_to_0`, `masked_sum/mean/...`. Note `masked_sum` divides by N̄ (or
  N̄² for 2-axis aggregation).
- `src/layers/perm_equiv_models.py` — `Eq2to2`, `Eq2to0` (the einsum mixing layers, `coefs`,
  `bias`, `diag_bias`, optional `factorize` low-rank form, α-scaling for 'S' configs),
  `Net2to2` (msg → dropout → agg loop). Also unused-in-nano `Eq1to2`, `Eq2to1`.
- `src/layers/generic_layers.py` — `MessageNet`, `BasicMLP`, `get_activation_fn`,
  `InputEncoder` (unused in nano), masked-norm plumbing.
- `src/layers/masked_batchnorm.py`, `masked_instancenorm.py` — masked norm layers.
- `src/trainer/` — `Trainer`, args (`args.py`: `--n-hidden`, `--config`, `--config-out`,
  `--activation` and friends), optimizers, schedulers, `init_weights`.
- `src/dataloaders/` — HDF5 jet datasets, `collate_fn` (adds beams, scaling, masks).
- `data/sample_data/` — small HDF5 samples usable for smoke tests without the full dataset.

## Invariants and gotchas

- **Permutation invariance** of the final score and **masking correctness** (padded rows and
  columns contribute exactly zero, including through BN statistics and any new quant layers)
  are non-negotiable. Add tests for both.
- Quantization scale factors must be **per-tensor** (or per-output-channel for weights),
  never per-particle-index — anything indexed by i,j breaks permutation equivariance.
- Inputs `d_ij` are extremely heavy-tailed (≈ Pareto); nano has *no* log embedding. Input
  quantization needs either the existing `--scale` pre-scaling plus a calibrated/learned
  scale (Brevitas learned-scale activation quant works), and this is the most likely accuracy
  failure point. Treat input quantizer config as a first-class hyperparameter.
- Aggregation sums grow with N (up to N=80+2 beams). Quantize *after* aggregation (the 6 ops
  are parameter-free), mirroring the FPGA firmware which widens accumulators then rescales by
  `1/N̄` (`invnave`, `invnave2`).
- `factorize=True` builds `coefs` as a product of two parameter tensors — not representable
  as a single QuantLinear. nano defaults to `factorize=False`; QAT may require/assert it.
- `dtype` flows via explicit `device=`/`dtype=` kwargs everywhere, and `Trainer` likely
  assumes the model returns `{'predict': tensor}` — keep that contract.
- Checkpoint format: plain `state_dict` saved by `Trainer` (`--save/--load/--bestfile`).
  Quant model state dicts will contain extra quantizer buffers; keep float ↔ quant loading
  paths explicit (a converter, not silent `strict=False`).

## Conventions for your edits

- Python ≥3.9, PyTorch ≥1.10 (Brevitas 0.12.x requires torch ≥1.11 in practice — pin
  reasonably in a new `requirements.txt`).
- Float (non-quant) path must remain the default and bit-for-bit unaffected when `--quant`
  is off: same module names where possible so old checkpoints still load.
- Prefer small, testable commits per phase of `docs/QAT_REFACTOR_PLAN.md`; run the test suite
  (`pytest tests/`) after each phase.
- No silent renames of CLI args; new quant args are additive.
