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


# ------------------------------------------------ input_clip_min (the scale floor)
# Why: input_quant's scale is a LEARNED parameter over a Pareto-tailed input and has a
# measured bad basin -- 2 of 12 full-dataset runs converged to a clip of 128 and
# collapsed (AUC 0.929/0.933 vs 0.943-0.960 for every run at clip >= 256). dot_t is
# range-limited, not resolution-limited, so flooring the clip costs nothing.

from src.layers.quant import clip_min_to_scaling_min


def _clip_model(clip_min, input_bit_width=6, unsigned=False, po2=True, seed=0):
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                       input_bit_width=input_bit_width, input_unsigned=unsigned,
                       input_clip_min=clip_min, po2_scales=po2)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                       dropout=False)


@pytest.mark.parametrize('clip_min,bits,unsigned,expect', [
    (256.0, 6, False, 8.0),     # signed:   256 / 2^5 = 8
    (512.0, 6, False, 16.0),    # signed:   512 / 2^5 = 16
    (256.0, 7, False, 4.0),     # signed:   256 / 2^6 = 4
    (512.0, 6, True, 16.0),     # unsigned: 512 / 63 = 8.13 -> po2 round UP = 16
])
def test_clip_min_to_scaling_min(clip_min, bits, unsigned, expect):
    assert clip_min_to_scaling_min(clip_min, bits, unsigned, po2=True) == expect


def test_clip_min_rounds_up_so_the_floor_is_never_undercut():
    """po2 rounding must widen the clip, never narrow it below the requested floor."""
    for bits in (6, 7, 8):
        for unsigned in (False, True):
            smv = clip_min_to_scaling_min(300.0, bits, unsigned, po2=True)
            threshold = (2 ** bits - 1) if unsigned else 2 ** (bits - 1)
            assert smv * threshold >= 300.0


def test_clip_min_rejects_nonpositive():
    with pytest.raises(ValueError):
        clip_min_to_scaling_min(0.0, 6, False, True)


def test_floor_actually_raises_a_scale_that_would_have_gone_low():
    """The point of the lever: calibrate on data that wants a tiny scale, and check
    the floor holds it up. Without it the same data lands far below."""
    small = {**make_batch(B=8, N_particles=10)}
    small['Pmu'] = small['Pmu'] * 0.05        # tiny dots -> calibration wants a tiny scale

    scales = {}
    for clip_min in (None, 512.0):
        model = _clip_model(clip_min)
        model.train()
        model(small)
        model.eval()
        scales[clip_min] = float(model.input_quant.act_quant.scale())

    assert scales[None] < scales[512.0], "floor did not raise the scale"
    assert 2 ** 5 * scales[512.0] >= 512.0, "clip point below the requested floor"


def test_floor_is_inert_when_the_learned_scale_is_already_above_it():
    """It must be a floor, not a pin — a healthy run should be untouched."""
    batch = make_batch(B=8, N_particles=12)
    ref = _clip_model(None)
    ref.train(); ref(batch); ref.eval()
    s_ref = float(ref.input_quant.act_quant.scale())

    floored = _clip_model(s_ref * 32 / 4.0)    # floor well BELOW where it lands
    floored.train(); floored(batch); floored.eval()
    assert float(floored.input_quant.act_quant.scale()) == s_ref


def test_default_is_off_and_adds_no_state():
    cfg = QuantConfig(enabled=True, input_bit_width=6)
    assert cfg.input_clip_min is None
    a = _clip_model(None).state_dict().keys()
    b = _clip_model(512.0).state_dict().keys()
    assert set(a) == set(b), "floor must not change state_dict keys"


def test_floor_must_be_replayed_on_reload():
    """The floor is NOT training-only, and this locks that in.

    Brevitas stores the raw runtime stat in scaling_impl.value and applies the clamp
    on every forward, so a rebuild WITHOUT input_clip_min loads the same state dict
    and then quantizes on a different grid — silently. Every tool that rebuilds the
    model has to replay the flag (model_loader.py, check_scales.py, export_golden.py).
    """
    batch = make_batch(B=4, N_particles=8)

    # Calibrate the floor off where this batch naturally lands: 8x above it, so the
    # grid is visibly coarser but values still resolve (a floor far above the data
    # zeroes every logit in BOTH models and the comparison says nothing).
    probe = _clip_model(None)
    probe.train(); probe(batch); probe.eval()
    floor = 2 ** 5 * float(probe.input_quant.act_quant.scale()) * 8

    model = _clip_model(floor)
    model.train(); model(batch); model.eval()

    plain = _clip_model(None, seed=1)
    plain.train(); plain(batch)
    plain.load_state_dict(model.state_dict(), strict=True)
    plain.eval()

    # Same weights, different effective grid -> the rebuild is NOT interchangeable.
    # Asserted on the SCALE, not the logits: this model is 33 random-init parameters,
    # so its output saturates to the same value on either grid and would hide the
    # difference. The scale is what model_loader.py derives every fixed-point type
    # from, so it is also the thing that actually has to match.
    assert float(plain.input_quant.act_quant.scale()) \
        < float(model.input_quant.act_quant.scale())

    # replaying it does reproduce the model exactly
    same = _clip_model(floor, seed=1)
    same.train(); same(batch)
    same.load_state_dict(model.state_dict(), strict=True)
    same.eval()
    assert torch.allclose(model(batch)['predict'], same(batch)['predict'],
                          atol=0, rtol=0)
