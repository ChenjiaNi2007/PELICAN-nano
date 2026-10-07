"""
Tests for the two hls4ml-5-class improvements:
  * --add-jet      : full-jet 4-momentum (dataset key Pjet) as a third spurion at slot 2
  * --head-hidden K: hidden ReLU layer between the 2->0 aggregation and the logits
"""
import math

import pytest
import torch

from tests.conftest import make_batch, make_model
from src.dataloaders.collate import collate_fn
from src.layers.quant import QuantConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _events(n_list=(5, 3, 7), seed=0, with_pjet=True):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, n in enumerate(n_list):
        p3 = torch.randn(n, 3, generator=g)
        E = p3.norm(dim=-1, keepdim=True) + 0.5
        pmu = torch.cat([E, p3], dim=-1)
        d = {'Pmu': pmu, 'Nobj': torch.tensor(n), 'is_signal': torch.tensor(i % 2)}
        if with_pjet:
            # full jet = these constituents plus some extra (truncated-away) momentum
            d['Pjet'] = pmu.sum(0) + torch.tensor([3.0, 0.1, -0.2, 0.3])
        out.append(d)
    return out


def _jet_batch(B=4, N_particles=10, seed=42):
    """make_batch layout [beams, constituents] -> [beams, jet, constituents]."""
    b = make_batch(B=B, N_particles=N_particles, add_beams=True, seed=seed)
    pjet = b['Pmu'][:, 2:].sum(1, keepdim=True) * 1.3   # timelike, E > 0
    Pmu = torch.cat([b['Pmu'][:, :2], pjet, b['Pmu'][:, 2:]], dim=1)
    return _wrap(Pmu, b['is_signal'])


def _wrap(Pmu, is_signal):
    pm = Pmu[..., 0] != 0.
    return {'Pmu': Pmu, 'is_signal': is_signal, 'particle_mask': pm,
            'edge_mask': pm.unsqueeze(1) & pm.unsqueeze(2), 'Nobj': pm.sum(-1)}


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 1. collate with add_jet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scale', [1.0, 0.01])
def test_collate_add_jet_layout(scale):
    ev = _events()
    batch = collate_fn(ev, scale=scale, add_beams=True, beam_mass=0, add_jet=True)
    ref = collate_fn([{k: v for k, v in e.items() if k != 'Pjet'} for e in ev],
                     scale=scale, add_beams=True, beam_mass=0)
    Pmu = batch['Pmu']
    nmax = max(e['Pmu'].shape[0] for e in ev)
    assert Pmu.shape == (len(ev), 3 + nmax, 4)
    # slots 0-1: beams (identical to the no-jet collate)
    torch.testing.assert_close(Pmu[:, :2], ref['Pmu'][:, :2], rtol=0, atol=0)
    torch.testing.assert_close(Pmu[:, :2, 0], torch.ones(len(ev), 2))
    # slot 2: the jet, scaled like the constituents
    for i, e in enumerate(ev):
        torch.testing.assert_close(Pmu[i, 2], e['Pjet'] * scale, rtol=0, atol=0)
    # constituents follow, unchanged
    torch.testing.assert_close(Pmu[:, 3:], ref['Pmu'][:, 2:], rtol=0, atol=0)
    # Nobj = raw + 3
    torch.testing.assert_close(batch['Nobj'], torch.tensor([e['Nobj'].item() + 3 for e in ev]))
    # masks: jet slot always active; padding rows inactive
    assert batch['particle_mask'][:, :3].all()
    for i, e in enumerate(ev):
        n = e['Pmu'].shape[0]
        assert batch['particle_mask'][i, 3:3 + n].all()
        assert not batch['particle_mask'][i, 3 + n:].any()
    assert torch.equal(batch['edge_mask'],
                       batch['particle_mask'].unsqueeze(1) & batch['particle_mask'].unsqueeze(2))
    # Pjet survives in the batch as (B, 4); pdg one-hot marks the jet like a beam
    assert batch['Pjet'].shape == (len(ev), 4)
    assert batch['scalars'].shape == (len(ev), 3 + nmax, 2)
    assert (batch['scalars'][:, :3, 1] == 1).all()


def test_collate_add_jet_nobj_truncation_keeps_pjet():
    ev = _events(n_list=(6, 5))
    batch = collate_fn(ev, nobj=2, add_beams=True, beam_mass=0, add_jet=True)
    assert batch['Pmu'].shape == (2, 5, 4)
    assert batch['Pjet'].shape == (2, 4)
    for i, e in enumerate(ev):
        torch.testing.assert_close(batch['Pmu'][i, 2], e['Pjet'], rtol=0, atol=0)
        torch.testing.assert_close(batch['Pjet'][i], e['Pjet'], rtol=0, atol=0)


def test_collate_add_jet_requires_beams():
    with pytest.raises(ValueError):
        collate_fn(_events(), add_beams=False, add_jet=True)


def test_collate_add_jet_missing_pjet():
    with pytest.raises(KeyError):
        collate_fn(_events(with_pjet=False), add_beams=True, add_jet=True)


def test_collate_default_ignores_pjet():
    ev = _events()
    a = collate_fn(ev, add_beams=True, beam_mass=0)
    b = collate_fn([{k: v for k, v in e.items() if k != 'Pjet'} for e in ev],
                   add_beams=True, beam_mass=0)
    assert torch.equal(a['Pmu'], b['Pmu']) and torch.equal(a['Nobj'], b['Nobj'])


# ---------------------------------------------------------------------------
# 2. head_hidden model: shape, param count, invariances
# ---------------------------------------------------------------------------

def test_head_output_shape_finite():
    model = make_model(n_hidden=4, n_out=5, head_hidden=16).eval()
    out = model(make_batch(B=6, N_particles=10))['predict']
    assert out.shape == (6, 5) and torch.isfinite(out).all()


@pytest.mark.parametrize('h', [2, 4])
@pytest.mark.parametrize('n_out', [1, 5])
@pytest.mark.parametrize('K', [3, 16])
def test_head_param_count(h, n_out, K):
    model = make_model(n_hidden=h, n_out=n_out, head_hidden=K)
    assert _n_params(model) == 8 * h + K * (2 * h + 1) + n_out * (K + 1)


def test_head_binary_output():
    model = make_model(n_hidden=2, n_out=1, head_hidden=8).eval()
    out = model(make_batch(B=3))['predict']
    assert out.shape == (3, 2)
    torch.testing.assert_close(out[:, 0], -out[:, 1])


@pytest.mark.parametrize('with_jet', [False, True])
def test_head_permutation_invariance(with_jet):
    model = make_model(n_hidden=4, n_out=5, head_hidden=16).eval()
    batch = _jet_batch(B=4, N_particles=12, seed=7) if with_jet else \
        make_batch(B=4, N_particles=12, seed=7)
    n_spur = 3 if with_jet else 2
    N = batch['Pmu'].shape[1]
    torch.manual_seed(99)
    perm = torch.cat([torch.arange(n_spur), torch.randperm(N - n_spur) + n_spur])
    batch_p = _wrap(batch['Pmu'][:, perm], batch['is_signal'])
    with torch.no_grad():
        a = model(batch)['predict']
        b = model(batch_p)['predict']
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('with_jet', [False, True])
def test_head_masking_invariance(with_jet):
    model = make_model(n_hidden=4, n_out=5, head_hidden=16).eval()
    batch = _jet_batch(B=3, N_particles=8, seed=13) if with_jet else \
        make_batch(B=3, N_particles=8, seed=13)
    Pmu = batch['Pmu']
    padded = _wrap(torch.cat([Pmu, torch.zeros(Pmu.shape[0], 4, 4)], dim=1), batch['is_signal'])
    with torch.no_grad():
        a = model(batch)['predict']
        b = model(padded)['predict']
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_jet_spurion_changes_output():
    """Sanity: the jet row is actually seen by the model (not masked away)."""
    model = make_model(n_hidden=2, n_out=5, head_hidden=8).eval()
    jb = _jet_batch(B=3, seed=3)
    nb = _wrap(torch.cat([jb['Pmu'][:, :2], jb['Pmu'][:, 3:]], dim=1), jb['is_signal'])
    with torch.no_grad():
        assert not torch.allclose(model(jb)['predict'], model(nb)['predict'])


# ---------------------------------------------------------------------------
# 3. head_hidden=0 is today's model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_out', [1, 5])
def test_head_zero_identical(n_out):
    a = make_model(n_hidden=2, n_out=n_out, seed=5).eval()
    b = make_model(n_hidden=2, n_out=n_out, seed=5, head_hidden=0).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa.keys()) == list(sb.keys())
    assert not any(k.startswith('head.') for k in sa)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    batch = make_batch(B=4)
    with torch.no_grad():
        assert torch.equal(a(batch)['predict'], b(batch)['predict'])
    assert _n_params(a) == 8 * 2 + n_out * (2 * 2 + 1)   # no head: 8h + n_out(2h+1)


# ---------------------------------------------------------------------------
# 4. quant build with the head
# ---------------------------------------------------------------------------

def test_quant_head_runs():
    pytest.importorskip('brevitas')
    from src.models.pelican_nano import PELICANNano
    torch.manual_seed(0)
    model = PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True, activation='relu',
        add_beams=True, config='s', config_out='s', average_nobj=49, factorize=False,
        masked=False, batchnorm=None, dropout=False,
        quant_config=QuantConfig(enabled=True), n_out=5, head_hidden=16,
        device=torch.device('cpu'), dtype=torch.float)
    import brevitas.nn as bnn
    assert isinstance(model.head, bnn.QuantLinear)
    assert isinstance(model.agg_2to0.act_layer, bnn.QuantReLU)
    batch = make_batch(B=4, N_particles=10)
    model.train()
    with torch.no_grad():
        model(batch)
    model.eval()
    with torch.no_grad():
        out = model(batch)['predict']
    assert out.shape == (4, 5) and torch.isfinite(out).all()
