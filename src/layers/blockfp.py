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
    """

    def __init__(self, bit_width: int, exp_min: int = 0, exp_max: int = 10,
                 from_energy: bool = True):
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

    def extra_repr(self) -> str:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 4:
            raise ValueError(f"BlockFPQuant expects 4-momenta in the last dim, got {tuple(x.shape)}")
        scale = torch.exp2(self.exponent(x))          # (..., 1), detached
        m = x / scale
        mq = torch.clamp(torch.round(m / self.mantissa_lsb),
                         self.q_min, self.q_max) * self.mantissa_lsb
        m_ste = m + (mq - m).detach()                 # straight-through
        return m_ste * scale
