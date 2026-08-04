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

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantConfig:
    enabled: bool = False
    weight_bit_width: int = 8
    act_bit_width: int = 8
    input_bit_width: int = 8       # higher default recommended for heavy-tailed inputs
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


def make_act_quant(config: QuantConfig, bit_width: Optional[int] = None) -> type:
    """Return a Brevitas activation quantizer class matching config."""
    from brevitas.quant.scaled_int import Int8ActPerTensorFloat
    from brevitas.inject.enum import RestrictValueType

    bw = config.act_bit_width if bit_width is None else bit_width
    attrs: dict = {'bit_width': bw}
    if config.po2_scales:
        attrs['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
    return type('_ActQuant', (Int8ActPerTensorFloat,), attrs)
