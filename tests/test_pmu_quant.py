"""
Tests for the optional raw-momentum quantizer (QuantConfig.pmu_bit_width,
Phase A* of nPELICAN-fpga/docs/INPUT_WIDTH_RETRAIN_PLAN.md).

Invariants:
  - pmu_bit_width=None (default) leaves the quant model state-dict-identical
    to before the field existed, so old QAT checkpoints keep loading strict.
  - The float path (quant_config=None) has pmu_quant=None and is unaffected.
  - With pmu_bit_width set: quantized momenta land on the learned po2 grid,
    exact zeros stay exact zeros (masking invariant), and save/load round-trips
    strict=True.
"""
import pytest
import torch

from tests.conftest import make_batch

try:
    import brevitas  # noqa: F401
    BREVITAS = True
except ImportError:
    BREVITAS = False

from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano

pytestmark = pytest.mark.skipif(not BREVITAS, reason="brevitas not installed")


def _quant_model(pmu_bit_width=None, seed=0):
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                       input_bit_width=6, pmu_bit_width=pmu_bit_width,
                       po2_scales=True)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                       dropout=False)


def test_default_off_no_new_state():
    model = _quant_model(pmu_bit_width=None)
    assert model.pmu_quant is None
    assert not any(k.startswith('pmu_quant.') for k in model.state_dict())


def test_float_path_has_no_pmu_quant():
    torch.manual_seed(0)
    model = PELICANNano(2, quant_config=None, dropout=False)
    assert model.pmu_quant is None
    batch = make_batch(B=2, N_particles=6)
    out = model(batch)['predict']
    assert torch.isfinite(out).all()


def test_pmu_quant_on_grid_and_zero_preserved():
    model = _quant_model(pmu_bit_width=14)
    assert model.pmu_quant is not None
    batch = make_batch(B=4, N_particles=8)
    model.train()
    model(batch)  # initialize the runtime-stats scale
    model.eval()

    x = batch['Pmu']
    y = model.pmu_quant(x)
    scale = model.pmu_quant.act_quant.scale().reshape(-1)[0]
    # every quantized momentum component is an integer multiple of the scale
    q = y / scale
    assert torch.allclose(q, torch.round(q), atol=1e-4)
    # po2 scale
    k = torch.log2(scale)
    assert torch.allclose(k, torch.round(k), atol=1e-5)
    # exact zeros (padded entries) stay exact zeros
    zeros = torch.zeros(3, 4)
    assert (model.pmu_quant(zeros) == 0).all()


def test_pmu_quant_state_dict_roundtrip():
    model = _quant_model(pmu_bit_width=14)
    batch = make_batch(B=2, N_particles=6)
    model.train()
    model(batch)
    model.eval()
    out_ref = model(batch)['predict']
    sd = model.state_dict()
    assert any(k.startswith('pmu_quant.') for k in sd)

    model2 = _quant_model(pmu_bit_width=14, seed=1)
    model2.train()
    model2(batch)  # populate scale buffers before strict load (CLAUDE.md gotcha)
    model2.load_state_dict(sd, strict=True)
    model2.eval()
    out2 = model2(batch)['predict']
    assert torch.allclose(out_ref, out2, atol=0, rtol=0)


def test_pmu_quant_in_forward_datapath():
    """The quantizer must run inside forward() and actually re-grid the momenta
    (a construction-only module would silently leave the datapath float)."""
    model = _quant_model(pmu_bit_width=4)  # aggressively coarse grid
    batch = make_batch(B=4, N_particles=8)
    model.train()
    model(batch)
    model.eval()

    seen = {}

    def hook(_mod, inp, out):
        seen['in'], seen['out'] = inp[0].detach(), out.detach()

    h = model.pmu_quant.register_forward_hook(hook)
    try:
        model(batch)
    finally:
        h.remove()
    assert 'out' in seen, "pmu_quant never ran during forward()"
    assert seen['in'].shape == batch['Pmu'].shape
    # a 4-bit grid cannot represent these random momenta exactly
    assert not torch.allclose(seen['in'], seen['out'])
