"""
src/layers/quant.py

QuantConfig dataclass and Brevitas quantizer factory helpers for nanoPELICAN QAT.

Design decisions (from docs/QAT_REFACTOR_PLAN.md):
  D2 – QuantLinear for mixing, QuantReLU/QuantIdentity for activations,
        QuantIdentity at input, post-aggregation, and output.
  D3 – Learned-scale QuantIdentity for input (heavy-tailed d_ij).
  D5 – config='S'/'M' (N^alpha scaling) raises NotImplementedError unless
        allow_alpha_scaling=True.
  D6 – Float biases by default (bias_bit_width=None); no bias quantizer needed.

Usage
-----
    from src.layers.quant import QuantConfig
    qcfg = QuantConfig(enabled=True, weight_bit_width=8, act_bit_width=8,
                       input_bit_width=16)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantConfig:
    enabled: bool = False
    weight_bit_width: int = 8
    act_bit_width: int = 8
    input_bit_width: int = 8       # higher default recommended for heavy-tailed inputs
    # The Minkowski dot of two physical momenta is NON-NEGATIVE: for massless
    # constituents d = E1*E2*(1-cos theta) >= 0, the diagonal is m^2 >= 0, and against
    # a beam (1,0,0,+-1) it is E -+ pz >= 0. So input_quant's sign bit encodes nothing.
    # Measured on tb_data/10k_pmu_test.dat (2.28M masked i<j dots): 0.11% come out
    # negative, ALL particle-particle, max |d| = 2^-6 -- float32 cancellation noise on
    # near-collinear pairs, 512x below one dot_t LSB, so every one of them rounds to 0
    # on any grid the firmware uses. Setting this True buys a free bit of dot
    # resolution at identical width (ap_ufixed<6,9>, LSB 8, vs ap_fixed<6,10>, LSB 16).
    # model_loader.py reads the signedness off the quantizer, so the emitted dot_t
    # follows automatically -- but train and export MUST agree or bit-exactness breaks.
    input_unsigned: bool = False
    # Floor on the d_ij clip point, in GeV^2 (None = unconstrained, the historical
    # behaviour). input_quant's scale is a LEARNED parameter over a Pareto-tailed
    # input, and it has a bad basin: measured across 12 full-dataset runs, 2 of them
    # converged to a clip of 128 and collapsed (AUC 0.929/0.933, bgRej 12.9/13.1)
    # while all ten runs at clip >= 256 landed in 0.943-0.960. The dots reach 1.1e4
    # with p99.9 = 1279, so clipping at 128 destroys the hard-pair tail that carries
    # jet mass. dot_t is RANGE-limited, not resolution-limited -- doubling resolution
    # at fixed range measured as a wash, so this floor costs nothing and removes the
    # failure mode. 256 is the measured cliff; the best runs sit at 512.
    # ⚠ NOT training-only. Brevitas stores the raw RUNTIME STAT in scaling_impl.value
    # and applies the clamp on every forward, so the floor is part of the quantizer's
    # definition, not a one-off training constraint. Any tool that REBUILDS the model
    # (model_loader.py, check_scales.py, export_golden.py) must replay it or it will
    # report a different scale than the model actually uses — the same silent failure
    # mode as [[input_unsigned]]. Verified by test_floor_must_be_replayed_on_reload.
    input_clip_min: Optional[float] = None
    pmu_bit_width: Optional[int] = None  # raw 4-momentum grid before dot4 (firmware input_t); None = float momenta
    # Lever 7: make the momentum grid PER-PARTICLE block floating point instead of
    # one uniform po2 grid. Needs pmu_bit_width set (it is the mantissa width).
    # See src/layers/blockfp.py and nPELICAN-fpga/docs/RESOURCE_REDUCTION_LEVERS.md.
    pmu_block_fp: bool = False
    pmu_exp_min: int = 0                 # per-particle exponent clamp (4-bit field)
    pmu_exp_max: int = 10
    bias_bit_width: Optional[int] = None  # None = float bias (fold at export)
    weight_per_channel: bool = False
    po2_scales: bool = False
    allow_alpha_scaling: bool = False


def make_weight_quant(config: QuantConfig) -> type:
    """Return a Brevitas weight quantizer class matching config."""
    from brevitas.quant.scaled_int import Int8WeightPerChannelFloat, Int8WeightPerTensorFloat
    from brevitas.inject.enum import RestrictValueType

    base: type = (
        Int8WeightPerChannelFloat if config.weight_per_channel
        else Int8WeightPerTensorFloat
    )
    attrs: dict = {'bit_width': config.weight_bit_width}
    if config.po2_scales:
        attrs['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
    return type('_WeightQuant', (base,), attrs)


def clip_min_to_scaling_min(clip_min: float, bit_width: int, unsigned: bool,
                            po2: bool) -> float:
    """Convert a minimum CLIP POINT to Brevitas' scaling_min_val (a minimum scale).

    The clip point is what actually matters physically (GeV^2, saturation of the
    d_ij tail) and is width-independent, so that is what the CLI exposes; Brevitas
    clamps the scale. The largest representable value is 2^(W-1)*scale signed and
    (2^W - 1)*scale unsigned.

    Under po2 the result is rounded UP to a power of two: the firmware contract
    (ap_fixed<B, B-k> for a learned scale 2^-k) needs the scale to stay on the po2
    grid, and rounding up can only widen the clip, never narrow it below the floor.
    """
    if clip_min <= 0:
        raise ValueError(f"input_clip_min must be positive, got {clip_min}")
    threshold = (2 ** bit_width - 1) if unsigned else 2 ** (bit_width - 1)
    smv = clip_min / threshold
    if po2:
        smv = 2.0 ** math.ceil(math.log2(smv))
    return smv


def make_act_quant(config: QuantConfig, bit_width: Optional[int] = None,
                   unsigned: bool = False, clip_min: Optional[float] = None) -> type:
    """Return a Brevitas activation quantizer class matching config.

    unsigned=True drops the sign bit (Uint8ActPerTensorFloat), spending the whole
    width on magnitude — a free doubling of resolution at fixed width, valid only
    where the quantized tensor is provably non-negative. Anything negative that
    does arrive clamps to 0. See QuantConfig.input_unsigned.

    clip_min floors the saturation point so the learned scale cannot drift into the
    measured bad basin. See QuantConfig.input_clip_min.
    """
    from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Uint8ActPerTensorFloat
    from brevitas.inject.enum import RestrictValueType

    bw = config.act_bit_width if bit_width is None else bit_width
    base: type = Uint8ActPerTensorFloat if unsigned else Int8ActPerTensorFloat
    attrs: dict = {'bit_width': bw}
    if config.po2_scales:
        attrs['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
    if clip_min is not None:
        attrs['scaling_min_val'] = clip_min_to_scaling_min(
            clip_min, bw, unsigned, config.po2_scales)
    return type('_ActQuant', (base,), attrs)
