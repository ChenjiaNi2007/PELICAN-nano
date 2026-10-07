"""
src/layers/blockfp.py

Per-particle block-floating-point fake-quantizer for the raw 4-momenta feeding
dot4. This is Lever 7 of nPELICAN-fpga/docs/RESOURCE_REDUCTION_LEVERS.md.

Motivation
----------
The uniform momentum grid (`QuantConfig.pmu_bit_width`, a Brevitas QuantIdentity)
gives every particle the same ABSOLUTE LSB. Because the Minkowski dot error goes
as `dd ~ |p| * dp` with |p| up to the clip, and because the trained `dot_t` grid is
very coarse (w6a6i6p12: LSB 16 GeV^2, 54.5% of dots quantize to 0), soft particles
end up with almost no usable relative precision and dots get thrown across dot_t
boundaries. Giving each particle its own power-of-two exponent fixes that: the
quantization becomes RELATIVE per particle.

Measured (nPELICAN-fpga/analysis/blockfp_dots.py, fraction of d_ij landing in a
different dot_t cell than float-exact):

    mantissa W    uniform     block-FP/particle
        12         17.61%          1.27%
        10         37.53%          4.39%
         8         63.78%         12.20%

i.e. 8-bit block-FP mantissas beat the 12-bit uniform production grid.

Representation
--------------
    e_i = clamp(floor(log2 E_i), exp_min, exp_max)      # per particle
    m_i = quantize(p_i / 2^e_i, W bits signed, I=2)     # per component
    p_i ~ m_i * 2^e_i

Two deliberate choices, both measured free (see analysis script section 4):

  * **Exponent from E alone**, not a 4-way max over |components|. In hardware this
    is a single LZC on the energy word instead of a 4-way max tree. It is provably
    safe at I=2: for a physical particle E >= |p_k| for every k, and 2^e <= E <
    2^(e+1), so |m_k| = |p_k| / 2^e < 2 always -- the mantissa cannot overflow the
    I=2 range. (If exp_max clamps a very hard particle, |m| can exceed 2 and
    saturates, which is the same behaviour the uniform grid already has at its clip.)

  * **Exponent clamped to [0, 10]** -> a 4-bit exponent field. Unclamped the span is
    53 (6 bits) only because of ~2^-43 float padding artifacts.

Invariants
----------
  * Zero in -> zero out exactly. An all-zero (padded) 4-vector gives e = exp_min and
    m = 0, so masked entries stay exactly 0 as the firmware requires.
  * Straight-through estimator: gradient flows to the momenta as identity. The
    exponent is computed under no_grad and treated as a constant (it is a step
    function, zero gradient a.e. anyway).
  * Stateless: no parameters and no buffers, so `state_dict()` is unchanged and
    checkpoints stay strict-loadable. The configuration lives in the run's `args`
    (`--pmu-block-fp`, `--pmu-bit-width`, `--pmu-exp-min/max`), which is how
    `model_loader.py` already auto-detects the momentum grid.

Static per-slot mode (SPS, ``static=True``)
------------------------------------------
Same representation, but the exponent is a TRAINED integer per particle SLOT
(slot = position on the particle axis; beams are slots 0,1 as prepended by
``collate_fn(add_beams=True)``, constituents 2..) instead of floor(log2 E):

    e_s  = clamp(round(log2_exp[s]), exp_min, exp_max)    # learned, per slot
    p_i ~ quantize(p_i / 2^e_s, W bits signed, I=2) * 2^e_s

The exponent is a compile-time constant per slot, so the firmware realignment
shift is wiring. Training: ``log2_exp`` is an ``nn.Parameter`` (float), rounded
with a straight-through estimator, and the quantizer uses the LSQ scale gradient
(dy/dlog2_exp = ln2 * 2^e * lsb * (round(q) - q) inside the range,
ln2 * 2^e * lsb * q_{min,max} when clipped) so the exponent actually learns. It
is data-initialised on the first training-mode forward to
``ceil(log2(max_batch |E_s|)) - 1`` (the clip 2^(e+1) covers the batch max).
Static mode adds two state_dict entries (``log2_exp``, ``exp_initialized``);
the dynamic mode stays stateless. ``exponent_table()`` is the single source of
truth for the per-slot integer exponents (check_scales / export / loader).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockFPQuant(nn.Module):
    """Per-particle block-floating-point fake-quantizer for 4-momenta.

    Expects a tensor whose LAST dimension is the 4-momentum (E, px, py, pz);
    all leading dimensions are treated as independent particles.

    Parameters
    ----------
    bit_width : int
        Signed mantissa width W. The mantissa grid is I=2, i.e. LSB 2^-(W-2)
        and range [-2, 2 - LSB].
    exp_min, exp_max : int
        Inclusive clamp on the per-particle exponent. The default [0, 10] is a
        4-bit field and is free on the top-tagging momenta (GeV units).
    from_energy : bool
        True (default): e = floor(log2 |E|), one LZC in hardware.
        False: e = floor(log2 max_k |p_k|), a 4-way max tree. Measured identical.
        (Dynamic mode only; static mode uses |E| for its data init.)
    static : bool
        False (default): the runtime per-particle exponent above.
        True: a learned static integer exponent per particle SLOT (SPS). Input
        must then be (..., n_slots, 4).
    n_slots : int
        Particle-axis length incl. beams (nobj + 2). Required when static.
    floor_batches : int
        Static only. K > 0: over the first K training-mode batches, raise a per-slot
        exponent FLOOR to the running max of ceil(log2 max|E|) - 1 (the data-init
        formula), so the learned exponent can never clip below what those batches
        needed -- the SPS analogue of --input-clip-min. 0 (default) = no floor.
    fixed : bool
        Static only. True: the exponent is NOT learned -- log2_exp has
        requires_grad False from construction and is only ever written by the data
        init (K = 0) or the floor phase (K > 0), after which it holds the
        data-derived table (== exp_floor for K > 0). Separates the static
        representation from the exponent's learning dynamics.
    """

    def __init__(self, bit_width: int, exp_min: int = 0, exp_max: int = 10,
                 from_energy: bool = True, static: bool = False,
                 n_slots: int = None, floor_batches: int = 0, fixed: bool = False):
        super().__init__()
        if bit_width < 3:
            raise ValueError(f"block-FP mantissa needs >=3 bits (I=2 + sign), got {bit_width}")
        if exp_min > exp_max:
            raise ValueError(f"exp_min ({exp_min}) > exp_max ({exp_max})")
        self.bit_width = int(bit_width)
        self.exp_min = int(exp_min)
        self.exp_max = int(exp_max)
        self.from_energy = bool(from_energy)
        # Mantissa grid: signed, I=2 integer bits (sign + 1), so F = W - 2.
        self.mantissa_lsb = 2.0 ** -(self.bit_width - 2)
        self.q_min = -(2 ** (self.bit_width - 1))
        self.q_max = 2 ** (self.bit_width - 1) - 1
        self.static = bool(static)
        self.n_slots = None
        if self.static:
            if n_slots is None or int(n_slots) < 1:
                raise ValueError(f"static block-FP needs n_slots >= 1, got {n_slots}")
            self.n_slots = int(n_slots)
            # Learned (float) exponent per slot; the effective exponent is the
            # rounded+clamped integer (STE). Start at exp_max (widest clip) --
            # overwritten by the data init on the first training-mode forward.
            self.log2_exp = nn.Parameter(torch.full((self.n_slots,), float(self.exp_max)))
            self.register_buffer('exp_initialized', torch.tensor(False))
            # Exponent floor (always registered in static mode so state_dicts load
            # across floor_batches values; == exp_min, i.e. inert, when K = 0).
            self.floor_batches = int(floor_batches)
            if self.floor_batches < 0:
                raise ValueError(f"floor_batches must be >= 0, got {floor_batches}")
            self.register_buffer('exp_floor', torch.full((self.n_slots,), float(self.exp_min)))
            self.register_buffer('floor_batches_seen', torch.tensor(0, dtype=torch.int64))
            self.fixed = bool(fixed)
            if self.fixed:
                self.log2_exp.requires_grad_(False)

    def extra_repr(self) -> str:
        if self.static:
            return (f"bit_width={self.bit_width} (I=2, LSB=2^{-(self.bit_width - 2)}), "
                    f"STATIC {'fixed data-derived' if self.fixed else 'learned'} exponent per slot (n_slots={self.n_slots}) "
                    f"clamped to [{self.exp_min},{self.exp_max}]")
        src = 'E' if self.from_energy else 'max|p_k|'
        return (f"bit_width={self.bit_width} (I=2, LSB=2^{-(self.bit_width - 2)}), "
                f"exp=floor(log2 {src}) clamped to [{self.exp_min},{self.exp_max}]")

    def exponent(self, x: torch.Tensor) -> torch.Tensor:
        """Per-particle exponent, shape (..., 1). Constant w.r.t. autograd."""
        with torch.no_grad():
            base = x[..., :1].abs() if self.from_energy else x.abs().amax(dim=-1, keepdim=True)
            # A zero (padded) particle has no exponent; exp_min makes m = 0 exactly.
            e = torch.where(
                base > 0,
                torch.floor(torch.log2(base.clamp_min(torch.finfo(x.dtype).tiny))),
                torch.full_like(base, float(self.exp_min)),
            )
            return e.clamp_(self.exp_min, self.exp_max)

    # ------------------------------------------------------------ static (SPS)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Static checkpoints written before the floor existed lack exp_floor /
        # floor_batches_seen. Their exact meaning is "no floor" (exp_floor = exp_min),
        # so fill those two keys only -- every other key stays strict.
        if self.static:
            if prefix + 'exp_floor' not in state_dict and prefix + 'log2_exp' in state_dict:
                state_dict[prefix + 'exp_floor'] = torch.full_like(self.exp_floor, float(self.exp_min))
            if prefix + 'floor_batches_seen' not in state_dict and prefix + 'log2_exp' in state_dict:
                state_dict[prefix + 'floor_batches_seen'] = torch.zeros_like(self.floor_batches_seen)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _effective_exponent(self, log2_exp: torch.Tensor) -> torch.Tensor:
        """e_int = clamp(round(log2_exp), max(exp_min, exp_floor), exp_max) (no grad)."""
        e = torch.clamp(torch.round(log2_exp), self.exp_min, self.exp_max)
        return torch.maximum(e, self.exp_floor.to(e.dtype))

    def exponent_table(self) -> torch.Tensor:
        """Effective integer exponent per slot, int64, shape (n_slots,).

        Single source of truth for check_scales.py / export_golden.py /
        nPELICAN-fpga model_loader.py (NPELICAN_BFP_EXP_TABLE)."""
        if not self.static:
            raise RuntimeError("exponent_table() is only defined for static (SPS) block-FP")
        with torch.no_grad():
            return self._effective_exponent(self.log2_exp.detach()).to(torch.int64).cpu()

    @torch.no_grad()
    def _batch_exponent(self, x: torch.Tensor) -> torch.Tensor:
        """Per slot clamp(ceil(log2(max_batch |E|)) - 1, exp_min, exp_max), so the clip
        2^(e+1) covers the batch max; empty slots (max |E| == 0) -> exp_min.
        Shared by the data init and the floor so the two always agree."""
        E = x[..., 0].abs().reshape(-1, self.n_slots)          # (batch..., n_slots)
        emax = E.amax(dim=0).to(self.log2_exp.dtype)
        e = torch.where(
            emax > 0,
            torch.ceil(torch.log2(emax.clamp_min(torch.finfo(emax.dtype).tiny))) - 1,
            torch.full_like(emax, float(self.exp_min)),
        ).clamp_(self.exp_min, self.exp_max)
        return e.to(self.log2_exp.device)

    @torch.no_grad()
    def _init_static_exponent(self, x: torch.Tensor) -> None:
        """Per-slot data init: log2_exp = _batch_exponent(first training batch)."""
        self.log2_exp.data.copy_(self._batch_exponent(x))
        self.exp_initialized.fill_(True)

    @torch.no_grad()
    def _update_floor(self, x: torch.Tensor) -> None:
        f = self._batch_exponent(x).to(self.exp_floor.dtype)
        self.exp_floor.copy_(torch.maximum(self.exp_floor, f))
        self.log2_exp.data.copy_(torch.maximum(self.log2_exp.data,
                                               self.exp_floor.to(self.log2_exp.dtype)))
        self.floor_batches_seen += 1
        if self.fixed:
            # fixed: hold exactly the data-derived floor table (no learned drift)
            self.log2_exp.data.copy_(self.exp_floor.to(self.log2_exp.dtype))

    def _forward_static(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2 or x.shape[-2] != self.n_slots:
            raise ValueError(f"static BlockFPQuant expects (..., n_slots={self.n_slots}, 4), "
                             f"got {tuple(x.shape)}")
        if not bool(self.exp_initialized):
            if self.training:
                self._init_static_exponent(x)
            else:
                raise RuntimeError("static exponent not initialized — run a training-mode "
                                   "forward or load a checkpoint")
        if self.training and int(self.floor_batches_seen) < self.floor_batches:
            self._update_floor(x)
        if self.fixed and self.log2_exp.requires_grad:
            self.log2_exp.requires_grad_(False)   # e.g. re-enabled externally
        with torch.no_grad():
            # keep the float exponent within half a step of the clamp so it cannot
            # run away where the rounded value no longer moves (per-slot lower
            # bound = the floor; == exp_min without one)
            lo = torch.clamp(self.exp_floor, min=float(self.exp_min)).to(self.log2_exp.dtype) - 0.5
            self.log2_exp.data.copy_(torch.maximum(self.log2_exp.data, lo)
                                     .clamp_(max=self.exp_max + 0.5))
        with torch.no_grad():
            e_int = self._effective_exponent(self.log2_exp)
        e = self.log2_exp + (e_int - self.log2_exp).detach()   # STE: value e_int, grad 1
        scale = torch.exp2(e)[..., None]                       # (n_slots, 1)
        lsb = self.mantissa_lsb
        m = x / scale
        q = m / lsb
        with torch.no_grad():
            qc = torch.clamp(torch.round(q), self.q_min, self.q_max)
        inr = (q > self.q_min) & (q < self.q_max)
        q_ste = torch.where(inr, q + (qc - q).detach(), qc)
        # dy/dx = 1 inside, 0 clipped; dy/de = ln2 * 2^e * lsb * (qc - q) inside,
        # ln2 * 2^e * lsb * qc clipped (LSQ).
        return q_ste * lsb * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 4:
            raise ValueError(f"BlockFPQuant expects 4-momenta in the last dim, got {tuple(x.shape)}")
        if self.static:
            return self._forward_static(x)
        scale = torch.exp2(self.exponent(x))          # (..., 1), detached
        m = x / scale
        mq = torch.clamp(torch.round(m / self.mantissa_lsb),
                         self.q_min, self.q_max) * self.mantissa_lsb
        m_ste = m + (mq - m).detach()                 # straight-through
        return m_ste * scale
