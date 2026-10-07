"""
Multi-class (n_out = K > 1) output head and metrics_multiclass tests.
"""
import os

import numpy as np
import pytest
import torch

from tests.conftest import make_batch, make_model
from src.models.pelican_nano import PELICANNano
from src.models import metrics_multiclass as mm


def _n_params(model):
    return sum(p.nelement() for p in model.parameters())


# 1. parameter count
@pytest.mark.parametrize('C_h', [1, 2, 4])
def test_param_count(C_h):
    assert _n_params(make_model(n_hidden=C_h, n_out=5)) == 8 * C_h + 5 * (2 * C_h + 1)
    assert _n_params(make_model(n_hidden=C_h, n_out=1)) == 10 * C_h + 1


def test_invalid_n_out():
    with pytest.raises(ValueError):
        make_model(n_out=0)


# 2. forward shape
def test_forward_shape_n_out5():
    model = make_model(n_hidden=2, n_out=5).eval()
    batch = make_batch(B=4, N_particles=10)
    with torch.no_grad():
        out = model(batch)['predict']
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


# 3. n_out=1 default stays binary
def test_n_out1_default():
    model = make_model(n_hidden=2).eval()
    assert model.n_out == 1
    with torch.no_grad():
        out = model(make_batch(B=4, N_particles=10))['predict']
    assert out.shape == (4, 2)
    assert torch.allclose(out[:, 0], -out[:, 1])


# 4. permutation and masking invariance
@pytest.mark.parametrize('n_hidden', [1, 2, 4])
def test_permutation_invariance_n_out5(n_hidden):
    model = make_model(n_hidden=n_hidden, n_out=5).eval()
    batch = make_batch(B=4, N_particles=12, seed=7)
    Pmu = batch['Pmu']
    N_total, n_beams = Pmu.shape[1], 2
    torch.manual_seed(99)
    perm_full = torch.cat([torch.arange(n_beams), torch.randperm(N_total - n_beams) + n_beams])
    Pmu_perm = Pmu[:, perm_full, :]
    mask = Pmu_perm[..., 0] != 0.
    batch_perm = {'Pmu': Pmu_perm, 'is_signal': batch['is_signal'],
                  'particle_mask': mask.bool(),
                  'edge_mask': (mask.unsqueeze(1) & mask.unsqueeze(2)).bool(),
                  'Nobj': mask.sum(-1)}
    with torch.no_grad():
        a = model(batch)['predict']
        b = model(batch_perm)['predict']
    assert a.shape == (4, 5)
    assert (a - b).abs().max().item() < 1e-5


@pytest.mark.parametrize('n_hidden', [1, 2, 4])
def test_masking_invariance_n_out5(n_hidden):
    model = make_model(n_hidden=n_hidden, n_out=5).eval()
    batch = make_batch(B=3, N_particles=8, seed=13)
    Pmu = batch['Pmu']
    B = Pmu.shape[0]
    Pmu_pad = torch.cat([Pmu, torch.zeros(B, 4, 4, dtype=Pmu.dtype)], dim=1)
    mask = Pmu_pad[..., 0] != 0.
    batch_pad = {'Pmu': Pmu_pad, 'is_signal': batch['is_signal'],
                 'particle_mask': mask.bool(),
                 'edge_mask': (mask.unsqueeze(1) & mask.unsqueeze(2)).bool(),
                 'Nobj': mask.sum(-1)}
    with torch.no_grad():
        a = model(batch)['predict']
        b = model(batch_pad)['predict']
    assert a.shape == (3, 5)
    assert (a - b).abs().max().item() < 1e-5


# 5. quant build
def test_quant_n_out5():
    pytest.importorskip('brevitas')
    from src.layers.quant import QuantConfig
    torch.manual_seed(0)
    model = PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True, activation='relu',
        add_beams=True, config='s', config_out='s', average_nobj=49,
        factorize=False, masked=False, batchnorm=None, dropout=False,
        quant_config=QuantConfig(enabled=True), n_out=5,
        device=torch.device('cpu'), dtype=torch.float,
    )
    batch = make_batch(B=4, N_particles=10)
    model.train()
    model(batch)
    model.eval()
    with torch.no_grad():
        out = model(batch)['predict']
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


# 6. metrics
def test_metrics_perfect():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 40)
    rng.shuffle(labels)
    logits = np.eye(5)[labels] * 10.
    m = mm.compute_multiclass_metrics(logits, labels)
    assert m['accuracy'] == 1.0
    for k, v in m.items():
        assert v == pytest.approx(1.0), k
    keys = list(m.keys())
    assert keys[:2] == ['accuracy', 'AUC']
    assert keys[2:7] == [f'AUC_{c}' for c in 'gqwzt']
    assert keys[7:12] == [f'TPR@FPR0.01_{c}' for c in 'gqwzt']
    assert keys[12:17] == [f'TPR@FPR0.1_{c}' for c in 'gqwzt']


def test_metrics_random():
    rng = np.random.default_rng(1234)
    labels = np.repeat(np.arange(5), 1000)
    rng.shuffle(labels)
    logits = rng.normal(size=(5000, 5))
    m = mm.compute_multiclass_metrics(logits, labels)
    assert abs(m['accuracy'] - 0.2) < 0.1
    for c in mm.class_names(5):
        assert 0.3 <= m[f'AUC_{c}'] <= 0.7
    assert 0.3 <= m['AUC'] <= 0.7


def test_metrics_missing_class():
    labels = np.array([0, 1, 1, 2, 0, 2])          # classes 3, 4 absent
    logits = np.random.default_rng(0).normal(size=(6, 5))
    m = mm.compute_multiclass_metrics(logits, labels)
    assert m['AUC_z'] == 0.0 and m['TPR@FPR0.1_t'] == 0.0
    out = mm.minibatch_metrics(torch.tensor(logits), torch.tensor(labels), 0.5)
    assert all(np.isfinite(v) for v in out)
    # single-class batch: no valid OvR class -> mAUC 0.0
    out = mm.minibatch_metrics(torch.randn(256, 5), torch.zeros(256, dtype=torch.long), 0.5)
    assert out[2] == 0.0
    assert isinstance(mm.minibatch_metrics_string(out), str)


def test_class_names():
    assert mm.class_names(5) == ['g', 'q', 'w', 'z', 't']
    assert mm.class_names(3) == ['c0', 'c1', 'c2']


def test_metrics_full(tmp_path):
    torch.manual_seed(0)
    targets = torch.arange(5).repeat(40)
    predict = torch.randn(200, 5, dtype=torch.double)
    prefix = str(tmp_path / 'valid')
    d, s = mm.metrics(predict, targets, torch.nn.CrossEntropyLoss(), prefix)
    assert list(d.keys())[0] == 'loss'
    assert np.isfinite(d['loss'])
    for c in 'gqwzt':
        f = prefix + f'_ROC_{c}.csv'
        assert os.path.exists(f)
        assert np.loadtxt(f, delimiter=',').shape[0] == 2
    assert 'mAUC' in s


# 7. state-dict key compatibility
def test_state_dict_keys():
    def build(n_out):
        torch.manual_seed(0)
        return PELICANNano(2, activation='relu', batchnorm='b', dropout=False,
                           n_out=n_out, device=torch.device('cpu'), dtype=torch.float)
    m1, m5 = build(1), build(5)
    sd1, sd5 = m1.state_dict(), m5.state_dict()
    assert sd1['agg_2to0.mixing.weight'].shape == (1, 4)
    assert sd5['agg_2to0.mixing.weight'].shape == (5, 4)
    assert set(sd1.keys()) == set(sd5.keys())
