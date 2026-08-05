"""
Tests for the unsigned d_ij quantizer (QuantConfig.input_unsigned, --input-unsigned).

Why this exists: the Minkowski dot of two physical momenta is non-negative
(d = E1*E2*(1-cos theta) >= 0 for massless constituents; d_ii = m^2 >= 0; against a
beam (1,0,0,+-1) it is E -+ pz >= 0), so input_quant's sign bit encodes nothing.
Dropping it spends the whole width on magnitude -- a free doubling of dot
resolution at identical width, which the firmware sees as ap_ufixed instead of
ap_fixed (model_loader.py reads the signedness off the quantizer).

Invariants:
  - the flag actually reaches the quantizer (a construction-only change would
    silently leave the datapath signed);
  - default stays SIGNED, so every existing checkpoint is unaffected;
  - unsigned really is finer at the same bit width (that is the whole point);
  - anything negative that does arrive clamps to 0 rather than wrapping;
  - masked/padded entries still give exactly 0.
"""
import pytest
import torch

from tests.conftest import make_batch

try:
    import brevitas  # noqa: F401
    BREVITAS = True
except ImportError:
    BREVITAS = False

pytestmark = pytest.mark.skipif(not BREVITAS, reason="brevitas not installed")

from src.layers.quant import QuantConfig, make_act_quant
from src.models.pelican_nano import PELICANNano


def _model(input_unsigned, input_bit_width=6, po2=True, seed=0):
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                       input_bit_width=input_bit_width,
                       input_unsigned=input_unsigned, po2_scales=po2)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                       dropout=False)


# ------------------------------------------------------------------ factory level

def test_factory_selects_signed_by_default():
    cfg = QuantConfig(enabled=True, input_bit_width=6)
    assert cfg.input_unsigned is False, "default must stay signed (existing checkpoints)"
    assert make_act_quant(cfg, 6).signed is True
    assert make_act_quant(cfg, 6, unsigned=True).signed is False


def test_factory_keeps_bit_width_and_po2_on_the_unsigned_path():
    cfg = QuantConfig(enabled=True, input_bit_width=6, po2_scales=True)
    q = make_act_quant(cfg, 6, unsigned=True)
    from brevitas.inject.enum import RestrictValueType
    assert q.bit_width == 6
    assert q.restrict_scaling_type == RestrictValueType.POWER_OF_TWO


# ------------------------------------------------------------------- model level

def test_flag_reaches_the_input_quantizer():
    assert bool(_model(input_unsigned=True).input_quant.act_quant.is_signed) is False
    assert bool(_model(input_unsigned=False).input_quant.act_quant.is_signed) is True


def test_only_input_quant_becomes_unsigned():
    """The flag must not leak into the other activation quantizers."""
    model = _model(input_unsigned=True)
    assert bool(model.output_quant.act_quant.is_signed) is True


def test_unsigned_is_finer_at_the_same_width():
    """The whole point: same bits, smaller LSB on non-negative data.

    Calibrated on identical batches, the signed grid spends one bit on a sign it
    never uses, so its scale is ~2x coarser. po2 off so the comparison is not
    quantized to a power-of-two step.
    """
    batch = make_batch(B=8, N_particles=12)
    scales = {}
    for unsigned in (False, True):
        model = _model(unsigned, po2=False)
        model.train()
        model(batch)                      # calibrate
        model.eval()
        scales[unsigned] = float(model.input_quant.act_quant.scale())
    assert scales[True] < scales[False], (
        f"unsigned scale {scales[True]} should be finer than signed {scales[False]}")


def test_negative_inputs_clamp_to_zero_not_wrap():
    """Cancellation noise (measured max |d| = 2^-6) must clamp, never wrap."""
    model = _model(input_unsigned=True, po2=False)
    batch = make_batch(B=4, N_particles=8)
    model.train()
    model(batch)                          # calibrate
    model.eval()
    x = torch.tensor([[-5.0, -0.5, 0.0, 0.5, 5.0]]).reshape(1, -1, 1, 1)
    out = model.input_quant(x)
    out = out.value if hasattr(out, 'value') else out
    assert (out >= 0).all(), f"unsigned quantizer produced negatives: {out}"
    assert (out.reshape(-1)[:2] == 0).all(), "negatives must clamp to 0"


def test_zero_stays_zero():
    """Masking invariant: padded entries are 0 and must survive the grid."""
    model = _model(input_unsigned=True, po2=False)
    batch = make_batch(B=4, N_particles=8)
    model.train()
    model(batch)
    model.eval()
    out = model.input_quant(torch.zeros(1, 3, 3, 1))
    out = out.value if hasattr(out, 'value') else out
    assert (out == 0).all()


def test_model_runs_end_to_end_unsigned():
    model = _model(input_unsigned=True)
    batch = make_batch(B=4, N_particles=10)
    model.train()
    model(batch)
    model.eval()
    out = model(batch)['predict']
    assert torch.isfinite(out).all()


def test_state_dict_roundtrip():
    """Checkpoints must stay strict-loadable across the unsigned path."""
    model = _model(input_unsigned=True)
    batch = make_batch(B=2, N_particles=6)
    model.train()
    model(batch)
    model.eval()
    ref = model(batch)['predict']

    model2 = _model(input_unsigned=True, seed=1)
    model2.train()
    model2(batch)                         # populate scale buffers before strict load
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.eval()
    assert torch.allclose(ref, model2(batch)['predict'], atol=0, rtol=0)


def test_physical_dots_are_non_negative():
    """The premise the lever rests on, checked on the synthetic timelike batch."""
    from src.models.pelican_nano import dot4
    pmu = make_batch(B=8, N_particles=12)['Pmu']
    dots = dot4(pmu.unsqueeze(1), pmu.unsqueeze(2))
    assert dots.min() > -1e-4, f"unexpected negative dot: {dots.min()}"
