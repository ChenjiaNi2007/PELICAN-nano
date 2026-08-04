"""
Tests for the per-particle block-floating-point momentum quantizer
(QuantConfig.pmu_block_fp, Lever 7 of
nPELICAN-fpga/docs/RESOURCE_REDUCTION_LEVERS.md).

Invariants that the firmware depends on:
  - masked/padded (all-zero) 4-vectors stay EXACTLY zero;
  - mantissas land on the 2^-(W-2) grid and never overflow I=2 when the
    exponent comes from the energy (the provable-safety argument in blockfp.py);
  - the exponent is per particle, drawn from E, and clamps to [exp_min, exp_max];
  - stateless => state_dict is unchanged, so checkpoints stay strict-loadable;
  - it actually runs inside forward() (a construction-only module would silently
    leave the datapath float);
  - gradients reach the momenta (straight-through), so QAT can train.
"""
import pytest
import torch

from tests.conftest import make_batch

try:
    import brevitas  # noqa: F401
    BREVITAS = True
except ImportError:
    BREVITAS = False

from src.layers.blockfp import BlockFPQuant
from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano


# ---------------------------------------------------------------- module tests
# (these need no Brevitas — BlockFPQuant is plain torch)

def test_zero_stays_exactly_zero():
    q = BlockFPQuant(8)
    z = torch.zeros(5, 4)
    assert (q(z) == 0).all()
    # and a zero particle mixed in with real ones
    x = torch.tensor([[100.0, 10.0, -5.0, 90.0], [0.0, 0.0, 0.0, 0.0]])
    assert (q(x)[1] == 0).all()


def test_mantissas_on_grid_and_within_I2():
    q = BlockFPQuant(8, exp_min=0, exp_max=10)
    torch.manual_seed(0)
    # physical particles: E >= |p_k| (build from a spatial momentum)
    p3 = torch.randn(500, 3) * 50
    E = p3.norm(dim=-1, keepdim=True)
    x = torch.cat([E, p3], dim=-1)

    y = q(x)
    e = q.exponent(x)
    m = y / torch.exp2(e)
    # on the 2^-(W-2) mantissa grid
    steps = m / q.mantissa_lsb
    assert torch.allclose(steps, torch.round(steps), atol=1e-4)
    # I=2 never overflows when e comes from E (no saturation clamp was hit)
    assert m.abs().max() < 2.0


def test_exponent_is_per_particle_from_energy_and_clamped():
    q = BlockFPQuant(8, exp_min=0, exp_max=10)
    x = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],       # beam spurion -> e = 0
        [8.0, 3.0, 0.0, 7.0],       # E=8  -> e = 3
        [1000.0, 10.0, 0.0, 999.0],  # E=1000 -> floor(log2)=9
        [0.25, 0.1, 0.0, 0.2],      # E<1 -> clamped up to exp_min=0
        [1e6, 0.0, 0.0, 1e6],       # huge -> clamped down to exp_max=10
    ])
    e = q.exponent(x).reshape(-1)
    assert torch.equal(e, torch.tensor([0.0, 3.0, 9.0, 0.0, 10.0]))


def test_beam_spurions_are_exact():
    """Beams (1,0,0,+-1) must survive the grid exactly — they do at e=0, any W."""
    beams = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]])
    for W in (4, 6, 8, 12):
        assert torch.equal(BlockFPQuant(W)(beams), beams)


def test_beats_uniform_grid_on_soft_particles():
    """The point of the lever: soft particles keep RELATIVE precision.

    A uniform grid with the same total width resolves a 0.5 GeV particle far
    worse than block-FP, which rescales it into full mantissa range.
    """
    W = 8
    soft = torch.tensor([[0.5, 0.3, -0.2, 0.31]])
    bfp = BlockFPQuant(W, exp_min=-4, exp_max=10)
    # uniform ap_fixed<8,10> equivalent: LSB 2^(10-8) = 4 GeV
    lsb = 4.0
    uni = torch.round(soft / lsb) * lsb
    assert (bfp(soft) - soft).abs().max() < (uni - soft).abs().max()


def test_straight_through_gradient():
    q = BlockFPQuant(8)
    x = torch.tensor([[100.0, 10.0, -5.0, 90.0]], requires_grad=True)
    q(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # STE: d(quantized)/dx == 1
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


def test_rejects_non_4vector_and_bad_config():
    with pytest.raises(ValueError):
        BlockFPQuant(8)(torch.zeros(3, 5))
    with pytest.raises(ValueError):
        BlockFPQuant(2)
    with pytest.raises(ValueError):
        BlockFPQuant(8, exp_min=5, exp_max=0)


# ----------------------------------------------------------------- model tests
pytestmark_model = pytest.mark.skipif(not BREVITAS, reason="brevitas not installed")


def _model(block_fp, pmu_bit_width=8, seed=0):
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                       input_bit_width=6, pmu_bit_width=pmu_bit_width,
                       pmu_block_fp=block_fp, po2_scales=True)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                       dropout=False)


@pytestmark_model
def test_model_selects_blockfp_and_adds_no_state():
    model = _model(block_fp=True)
    assert isinstance(model.pmu_quant, BlockFPQuant)
    # stateless: no new state_dict keys, so checkpoints stay strict-loadable
    assert not any(k.startswith('pmu_quant.') for k in model.state_dict())


@pytestmark_model
def test_uniform_path_unchanged_when_flag_off():
    import brevitas.nn as qnn
    model = _model(block_fp=False)
    assert isinstance(model.pmu_quant, qnn.QuantIdentity)


@pytestmark_model
def test_blockfp_runs_in_forward_datapath():
    model = _model(block_fp=True, pmu_bit_width=4)  # coarse enough to be visible
    batch = make_batch(B=4, N_particles=8)
    model.eval()

    seen = {}

    def hook(_mod, inp, out):
        seen['in'], seen['out'] = inp[0].detach(), out.detach()

    h = model.pmu_quant.register_forward_hook(hook)
    try:
        out = model(batch)['predict']
    finally:
        h.remove()
    assert 'out' in seen, "pmu_quant never ran during forward()"
    assert seen['in'].shape == batch['Pmu'].shape
    assert not torch.allclose(seen['in'], seen['out'])
    assert torch.isfinite(out).all()


@pytestmark_model
def test_blockfp_model_state_dict_roundtrip():
    model = _model(block_fp=True)
    batch = make_batch(B=2, N_particles=6)
    model.train()
    model(batch)
    model.eval()
    ref = model(batch)['predict']

    model2 = _model(block_fp=True, seed=1)
    model2.train()
    model2(batch)  # populate Brevitas scale buffers before strict load
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.eval()
    assert torch.allclose(ref, model2(batch)['predict'], atol=0, rtol=0)


@pytestmark_model
def test_padded_particles_stay_zero_through_model_quantizer():
    """Masking invariant end-to-end: padded rows of Pmu must come out exactly 0."""
    model = _model(block_fp=True)
    batch = make_batch(B=3, N_particles=10)
    pmu = batch['Pmu'].clone()
    pmu[:, 5:, :] = 0.0
    out = model.pmu_quant(pmu)
    assert (out[:, 5:, :] == 0).all()
