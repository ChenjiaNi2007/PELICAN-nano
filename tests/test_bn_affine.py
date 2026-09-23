"""
Tests for the offset-only BatchNorm replacement (MaskedOffset2d, ``--batchnorm a``).

At inference a trained BatchNorm is a fixed affine s*x + beta'; the multiplicative s is
absorbed by the adjacent quantizers' learned po2 scales, so only the offset carries
information. The 'a' variant keeps just a learnable per-channel offset (state-dict key
``normlayer.bias`` only), which the firmware loader reads as BN with mean=0, scale=1.

Invariants:
  - 'a' builds and runs (float and quant), and its state dict holds ONLY normlayer.bias;
  - padded entries stay exactly 0 after the offset (masking);
  - None/'None' means no norm layer at all;
  - the offset actually trains (receives gradient);
  - the default 'b' state-dict key set is unchanged.
"""
import pytest
import torch

from tests.conftest import make_batch, make_model
from src.layers.masked_batchnorm import MaskedOffset2d

try:
    import brevitas  # noqa: F401
    BREVITAS = True
except ImportError:
    BREVITAS = False

N_HIDDEN = 2


def _quant_model(batchnorm, seed=0):
    from src.layers.quant import QuantConfig
    from src.models.pelican_nano import PELICANNano
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=8, act_bit_width=8,
                       input_bit_width=8, po2_scales=True)
    return PELICANNano(N_HIDDEN, quant_config=qcfg, batchnorm=batchnorm,
                       activation='relu', dropout=False)


def _build(kind, batchnorm):
    if kind == 'float':
        return make_model(n_hidden=N_HIDDEN, batchnorm=batchnorm)
    if not BREVITAS:
        pytest.skip("brevitas not installed")
    return _quant_model(batchnorm)


def _ragged_batch(B=3, N_particles=8, n_active=(8, 5, 2)):
    """Batch whose events have different multiplicities (trailing particles zeroed)."""
    batch = make_batch(B=B, N_particles=N_particles, add_beams=True)
    Pmu = batch['Pmu'].clone()
    for b, n in enumerate(n_active):
        Pmu[b, 2 + n:] = 0.
    particle_mask = Pmu[..., 0] != 0.
    batch.update(
        Pmu=Pmu,
        particle_mask=particle_mask,
        edge_mask=particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2),
        Nobj=particle_mask.sum(-1),
    )
    return batch


KINDS = ['float', 'quant']


# ------------------------------------------------------------- 1. build / keys

@pytest.mark.parametrize('kind', KINDS)
def test_offset_builds_runs_and_has_only_bias_keys(kind):
    model = _build(kind, 'a')
    assert isinstance(model.net2to2.message_layers[0].normlayer, MaskedOffset2d)
    assert isinstance(model.msg_2to0.normlayer, MaskedOffset2d)

    model.train()
    out = model(make_batch(B=2, N_particles=6))['predict']
    assert torch.isfinite(out).all()
    model.eval()
    out = model(make_batch(B=2, N_particles=6))['predict']
    assert torch.isfinite(out).all()

    sd = model.state_dict()
    assert sd['net2to2.message_layers.0.normlayer.bias'].shape == (1,)
    assert sd['msg_2to0.normlayer.bias'].shape == (N_HIDDEN,)
    norm_keys = [k for k in sd if '.normlayer.' in k]
    assert sorted(norm_keys) == ['msg_2to0.normlayer.bias',
                                 'net2to2.message_layers.0.normlayer.bias']
    for k in sd:
        assert 'running_mean' not in k and 'running_var' not in k
        assert 'num_batches_tracked' not in k
        assert not k.endswith('normlayer.weight')


# ------------------------------------------------------------------ 2. masking

@pytest.mark.parametrize('kind', KINDS)
def test_offset_keeps_padded_entries_exactly_zero(kind):
    model = _build(kind, 'a')
    with torch.no_grad():  # non-zero offset so masking is actually exercised
        model.net2to2.message_layers[0].normlayer.bias.fill_(0.75)
        model.msg_2to0.normlayer.bias.copy_(torch.linspace(0.3, -0.4, N_HIDDEN))
    batch = _ragged_batch()

    captured = {}
    h = model.net2to2.message_layers[0].register_forward_hook(
        lambda m, i, o: captured.__setitem__('bn1', o.detach().clone()))
    model.eval()
    model(batch)
    h.remove()

    out = captured['bn1']                         # [B, N, N, C]
    edge = batch['edge_mask']
    assert out.shape[:3] == edge.shape
    assert (out[~edge] == 0).all(), "padded entries must be exactly 0"
    assert (~edge).any() and edge.any()
    # active entries do see the offset (not everything got zeroed)
    assert (out[edge] != 0).any()


def test_offset_module_math():
    layer = MaskedOffset2d(3)
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([1., -2., 0.5]))
    x = torch.randn(2, 4, 4, 3)
    mask = torch.rand(2, 4, 4, 1) > 0.4
    y = layer(x, mask)
    ref = torch.where(mask, x + layer.bias, torch.zeros(()))
    assert torch.equal(y, ref)
    assert torch.equal(layer(x), x + layer.bias)          # unmasked path
    assert list(layer.state_dict().keys()) == ['bias']


# ---------------------------------------------------------------- 3. no norm

@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('bn', [None, 'None'])
def test_none_has_no_normlayer(kind, bn):
    model = _build(kind, bn)
    assert not hasattr(model.net2to2.message_layers[0], 'normlayer')
    assert not hasattr(model.msg_2to0, 'normlayer')
    assert not any('normlayer' in k for k in model.state_dict())
    model.eval()
    out = model(make_batch(B=2, N_particles=6))['predict']
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------- 4. gradient

@pytest.mark.parametrize('kind', KINDS)
def test_offset_receives_gradient(kind):
    model = _build(kind, 'a')
    model.train()
    out = model(_ragged_batch())['predict']
    out[:, 1].sum().backward()
    for layer in (model.net2to2.message_layers[0].normlayer, model.msg_2to0.normlayer):
        assert layer.bias.grad is not None
        assert layer.bias.grad.abs().sum() > 0


# -------------------------------------------------------------- 5. regression

EXPECTED_B_SUFFIXES = {'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'}


@pytest.mark.parametrize('kind', KINDS)
def test_batchnorm_b_keys_unchanged(kind):
    sd = _build(kind, 'b').state_dict()
    for prefix in ('net2to2.message_layers.0.normlayer.', 'msg_2to0.normlayer.'):
        got = {k[len(prefix):] for k in sd if k.startswith(prefix)}
        assert got == EXPECTED_B_SUFFIXES, (prefix, got)
