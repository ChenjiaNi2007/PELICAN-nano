"""
Tests for --jet-quant-split: separate learned quantizers for the three d_ij populations
(particle-particle, particle/beam-jet, m_jet^2) and for the jet momentum row, with the
full-jet spurion at the fixed slot 2 ([beam+, beam-, jet, constituents...]).
"""
import argparse

import pytest
import torch

pytest.importorskip('brevitas')

from tests.conftest import make_batch
from src.layers.quant import QuantConfig
from src.trainer.args import validate_jet_quant_split

PROD = dict(enabled=True, po2_scales=True, weight_bit_width=6, act_bit_width=6,
            input_bit_width=6)


def _model(seed=0, batchnorm='b', **qkw):
    from src.models.pelican_nano import PELICANNano
    torch.manual_seed(seed)
    return PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True, activation='relu',
        add_beams=True, config='s', config_out='s', average_nobj=49, factorize=False,
        masked=True, batchnorm=batchnorm, dropout=False,
        quant_config=QuantConfig(**{**PROD, **qkw}), n_out=5, head_hidden=4,
        device=torch.device('cpu'), dtype=torch.float)


def _wrap(Pmu, is_signal):
    pm = Pmu[..., 0] != 0.
    return {'Pmu': Pmu, 'is_signal': is_signal, 'particle_mask': pm,
            'edge_mask': pm.unsqueeze(1) & pm.unsqueeze(2), 'Nobj': pm.sum(-1)}


def _jet_batch(B=4, N=10, seed=42, jet_boost=1.3):
    """[beams, jet, constituents]; the jet row = jet_boost * sum of the constituents."""
    b = make_batch(B=B, N_particles=N, add_beams=True, seed=seed)
    pjet = b['Pmu'][:, 2:].sum(1, keepdim=True) * jet_boost
    return _wrap(torch.cat([b['Pmu'][:, :2], pjet, b['Pmu'][:, 2:]], dim=1), b['is_signal'])


def _calibrated(batch, **qkw):
    m = _model(**qkw)
    m.train()
    with torch.no_grad():
        m(batch)
    return m.eval()


# (a) default path unchanged ---------------------------------------------------------

@pytest.mark.parametrize('pmu', [None, 12])
def test_default_state_dict_unchanged(pmu):
    batch = _jet_batch()
    a = _calibrated(batch, pmu_bit_width=pmu)
    b = _calibrated(batch, pmu_bit_width=pmu, jet_quant_split=False)
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    assert not any('_jet' in k or 'mjet' in k for k in a.state_dict())
    assert a.input_quant_jet is None and a.input_quant_mjet is None and a.pmu_quant_jet is None
    assert a.jet_quant_split is False


def test_split_adds_only_jet_keys():
    batch = _jet_batch()   # Brevitas scale params only exist after a training forward
    a = _calibrated(batch, pmu_bit_width=12).state_dict()
    b = _calibrated(batch, pmu_bit_width=12, jet_quant_split=True).state_dict()
    extra = set(b) - set(a)
    assert set(a) <= set(b)
    assert extra and all(k.split('.')[0] in ('input_quant_jet', 'input_quant_mjet',
                                             'pmu_quant_jet') for k in extra)


def test_plain_tensor_input_is_bit_identical_downstream():
    """The split path hands Net2to2 a plain tensor (no single scale exists); the default
    path still hands it input_quant's QuantTensor. Downstream must treat both alike."""
    batch = _jet_batch(seed=3)
    m = _calibrated(batch, pmu_bit_width=12)
    with torch.no_grad():
        a = m(batch, covariance_test=True)
        m.input_quant.return_quant_tensor = False
        b = m(batch, covariance_test=True)
    assert type(a['inputs']).__name__ != 'Tensor' and type(b['inputs']) is torch.Tensor
    assert torch.equal(a['predict'], b['predict'])


# (b) split path: shape, finite, permutation and masking invariance --------------------

@pytest.mark.parametrize('pmu', [None, 12])
def test_split_shape_finite(pmu):
    batch = _jet_batch()
    m = _calibrated(batch, pmu_bit_width=pmu, jet_quant_split=True)
    with torch.no_grad():
        out = m(batch)['predict']
    assert out.shape == (4, 5) and torch.isfinite(out).all()


@pytest.mark.parametrize('pmu', [None, 12])
def test_split_permutation_invariance_exact(pmu):
    batch = _jet_batch(B=4, N=12, seed=7)
    m = _calibrated(batch, pmu_bit_width=pmu, jet_quant_split=True)
    N = batch['Pmu'].shape[1]
    g = torch.Generator().manual_seed(99)
    perm = torch.cat([torch.arange(3), torch.randperm(N - 3, generator=g) + 3])
    bp = _wrap(batch['Pmu'][:, perm], batch['is_signal'])
    with torch.no_grad():
        a = m(batch, covariance_test=True)
        b = m(bp, covariance_test=True)
    # quantized d_ij permute exactly; the logits agree up to float summation order
    assert torch.equal(a['inputs'][:, perm][:, :, perm], b['inputs'])
    torch.testing.assert_close(a['predict'], b['predict'], rtol=1e-5, atol=1e-5)


def test_split_masking_invariance():
    batch = _jet_batch(B=3, N=8, seed=13)
    m = _calibrated(batch, pmu_bit_width=12, jet_quant_split=True)
    Pmu = batch['Pmu']
    padded = _wrap(torch.cat([Pmu, torch.zeros(Pmu.shape[0], 4, 4)], dim=1), batch['is_signal'])
    with torch.no_grad():
        a = m(batch)['predict']
        r = m(padded, covariance_test=True)
    torch.testing.assert_close(a, r['predict'], rtol=1e-5, atol=1e-5)
    N = Pmu.shape[1]
    assert (r['inputs'][:, N:] == 0).all() and (r['inputs'][:, :, N:] == 0).all()


def test_split_routes_populations():
    """Each region of the grid comes from its own quantizer."""
    batch = _jet_batch(B=3, N=8, seed=5, jet_boost=30.0)
    m = _calibrated(batch, jet_quant_split=True)
    from src.models.lorentz_metric import dot4
    P = batch['Pmu']
    d = dot4(P.unsqueeze(1), P.unsqueeze(2)).unsqueeze(-1)
    with torch.no_grad():
        q = m(batch, covariance_test=True)['inputs']
        pp = m.input_quant(d).value
        pj = m.input_quant_jet(d)
        jj = m.input_quant_mjet(d)
    keep = torch.ones(P.shape[1], dtype=torch.bool); keep[2] = False
    assert torch.equal(q[:, keep][:, :, keep], pp[:, keep][:, :, keep])
    assert torch.equal(q[:, 2, keep], pj[:, 2, keep])
    assert torch.equal(q[:, keep, 2], pj[:, keep, 2])
    assert torch.equal(q[:, 2, 2], jj[:, 2, 2])


def test_split_requires_jet_slot():
    m = _model(jet_quant_split=True).eval()
    P = torch.tensor([[[1., 0., 0., 1.], [1., 0., 0., -1.]]])
    with pytest.raises(ValueError):
        m(_wrap(P, torch.tensor([0])))


# (c) the three scales differ when jet dots are ~1000x the pair dots ------------------

def test_three_scales_differ():
    b = make_batch(B=8, N_particles=10, add_beams=True, seed=1)
    pjet = b['Pmu'][:, 2:].sum(1, keepdim=True) * 30.0      # jet dots ~1e3x pair dots
    batch = _wrap(torch.cat([b['Pmu'][:, :2], pjet, b['Pmu'][:, 2:]], dim=1), b['is_signal'])
    m = _calibrated(batch, pmu_bit_width=12, jet_quant_split=True)
    s_pp = float(m.input_quant.act_quant.scale())
    s_pj = float(m.input_quant_jet.act_quant.scale())
    s_jj = float(m.input_quant_mjet.act_quant.scale())
    assert s_pp < s_pj < s_jj, (s_pp, s_pj, s_jj)
    assert s_pj / s_pp >= 16 and s_jj / s_pj >= 4
    assert float(m.pmu_quant_jet.act_quant.scale()) > float(m.pmu_quant.act_quant.scale())


# (d) pmu_quant_jet iff pmu_bit_width ---------------------------------------------------

@pytest.mark.parametrize('pmu', [None, 12])
def test_pmu_quant_jet_iff_pmu_bits(pmu):
    m = _model(pmu_bit_width=pmu, jet_quant_split=True)
    assert (m.pmu_quant_jet is not None) == (pmu is not None)
    assert m.input_quant_jet is not None and m.input_quant_mjet is not None


# (e) block-FP + split ------------------------------------------------------------------

def test_blockfp_split_not_implemented():
    with pytest.raises(NotImplementedError, match='block'):
        _model(pmu_bit_width=12, pmu_block_fp=True, jet_quant_split=True)


# (f) trainer-level guards --------------------------------------------------------------

@pytest.mark.parametrize('quant,add_jet,ok', [(True, True, True), (False, True, False),
                                               (True, False, False), (False, False, False)])
def test_trainer_guards(quant, add_jet, ok):
    args = argparse.Namespace(jet_quant_split=True, quant=quant, add_jet=add_jet)
    if ok:
        validate_jet_quant_split(args)
    else:
        with pytest.raises(ValueError):
            validate_jet_quant_split(args)
    validate_jet_quant_split(argparse.Namespace(jet_quant_split=False, quant=False, add_jet=False))


def test_cli_flag_parses():
    from src.trainer.args import setup_argparse
    p = setup_argparse()
    assert p.parse_args([]).jet_quant_split is False
    assert p.parse_args(['--jet-quant-split']).jet_quant_split is True


def test_split_ignored_when_quant_disabled():
    from src.models.pelican_nano import PELICANNano
    m = PELICANNano(2, activation='relu', quant_config=QuantConfig(enabled=False,
                                                                     jet_quant_split=True))
    assert m.jet_quant_split is False and m.input_quant_jet is None


# (g) independent bit widths for the three jet quantizers ------------------------------

def _bw(mod):
    return int(float(mod.act_quant.bit_width().detach()))


def test_jet_bit_widths_override():
    m = _model(pmu_bit_width=12, jet_quant_split=True, jet_input_bit_width=10,
               mjet_input_bit_width=16, jet_pmu_bit_width=20)
    assert _bw(m.input_quant_jet) == 10
    assert _bw(m.input_quant_mjet) == 16
    assert _bw(m.pmu_quant_jet) == 20
    assert _bw(m.input_quant) == 6 and _bw(m.pmu_quant) == 12   # the shared grids untouched


def test_jet_bit_widths_default_inherit():
    m = _model(pmu_bit_width=12, jet_quant_split=True)
    assert _bw(m.input_quant_jet) == 6 and _bw(m.input_quant_mjet) == 6
    assert _bw(m.pmu_quant_jet) == 12


def test_jet_bit_widths_forward_finite():
    batch = _jet_batch()
    m = _calibrated(batch, pmu_bit_width=12, jet_quant_split=True, jet_input_bit_width=10,
                    mjet_input_bit_width=16, jet_pmu_bit_width=20)
    with torch.no_grad():
        assert torch.isfinite(m(batch)['predict']).all()


@pytest.mark.parametrize('field', ['jet_input_bit_width', 'mjet_input_bit_width',
                                   'jet_pmu_bit_width'])
def test_jet_bit_widths_require_split(field):
    with pytest.raises(ValueError, match='jet_quant_split'):
        QuantConfig(**{**PROD, 'pmu_bit_width': 12, field: 10})


def test_jet_pmu_bit_width_requires_pmu_bit_width():
    with pytest.raises(ValueError, match='pmu_bit_width'):
        QuantConfig(**{**PROD, 'jet_quant_split': True, 'jet_pmu_bit_width': 20})


@pytest.mark.parametrize('field', ['jet_input_bit_width', 'mjet_input_bit_width',
                                   'jet_pmu_bit_width'])
def test_jet_bit_width_cli(field):
    from src.trainer.args import setup_argparse
    flag = '--' + field.replace('_', '-')
    p = setup_argparse()
    assert getattr(p.parse_args([]), field) is None
    a = p.parse_args([flag, '16'])
    assert getattr(a, field) == 16
    a.quant, a.add_jet, a.pmu_bit_width = True, True, 12
    with pytest.raises(ValueError, match='jet-quant-split'):
        validate_jet_quant_split(a)          # without --jet-quant-split
    a.jet_quant_split = True
    validate_jet_quant_split(a)
    if field == 'jet_pmu_bit_width':
        a.pmu_bit_width = None
        with pytest.raises(ValueError, match='pmu-bit-width'):
            validate_jet_quant_split(a)
