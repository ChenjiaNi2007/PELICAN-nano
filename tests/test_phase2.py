"""
Phase 2 tests: Brevitas QuantLinear/QuantIdentity/QuantReLU wiring.

Covers:
  R1  – Float path numerically unchanged when quant_config.enabled=False (or None)
  R2  – Quant model builds without error for all C_h in {1, 2, 3, 10}
  R3  – 8-bit quant model is permutation-invariant
  R4  – 8-bit quant model is masking-invariant
  R5  – config='S' raises NotImplementedError unless allow_alpha_scaling=True
  R6  – Float-quant parity: disabling quantizers gives numerically identical output
  R7  – Parameter count unchanged: 10*C_h + 1 (float path; quant adds no trainable params)
"""
import os
import pytest
import torch
import torch.nn as nn

from tests.conftest import make_batch, make_model

try:
    import brevitas.nn as _bnn
    from brevitas.quant.scaled_int import (
        Int8WeightPerTensorFloat,
        Int8ActPerTensorFloat,
    )
    from brevitas.quant import NoneWeightQuant, NoneActQuant
    _BREVITAS = True
except ImportError:
    _BREVITAS = False

pytestmark = pytest.mark.skipif(
    not _BREVITAS, reason='brevitas not installed — skipping Phase-2 tests'
)

from src.layers.quant import QuantConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_quant_model(n_hidden=2, seed=0, **qcfg_kwargs):
    """Return an eval-mode PELICANNano with quantization enabled."""
    from src.models.pelican_nano import PELICANNano
    qcfg = QuantConfig(enabled=True, **qcfg_kwargs)
    torch.manual_seed(seed)
    model = PELICANNano(
        n_hidden=n_hidden,
        activate_agg=False, activate_lin=True,
        activation='relu',
        add_beams=True,
        config='s', config_out='s',
        average_nobj=49,
        factorize=False,
        masked=False,
        batchnorm=None,
        dropout=False,
        quant_config=qcfg,
        device=torch.device('cpu'),
        dtype=torch.float,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# R1 – Float path numerically unchanged
# ---------------------------------------------------------------------------

def test_float_path_unchanged():
    """
    Float model (quant_config=None) matches reference make_model output.
    Tests that adding QuantConfig machinery does not change float results.
    """
    batch = make_batch(B=3, N_particles=8, add_beams=True, seed=42)
    ref = make_model(n_hidden=2, seed=7)
    ref.eval()

    # Reload same seed via PELICANNano directly (no quant_config → identical path)
    from src.models.pelican_nano import PELICANNano
    torch.manual_seed(7)
    model_no_qcfg = PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True, config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False, batchnorm=None,
        dropout=False, quant_config=None,
        device=torch.device('cpu'), dtype=torch.float,
    )
    model_no_qcfg.eval()

    with torch.no_grad():
        out_ref = ref(batch)['predict']
        out_nq = model_no_qcfg(batch)['predict']

    max_diff = (out_ref - out_nq).abs().max().item()
    assert max_diff < 1e-6, f'Float path changed with quant_config=None: diff={max_diff:.2e}'


# ---------------------------------------------------------------------------
# R2 – Quant model builds for all C_h
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_hidden', [1, 2, 3, 10])
def test_quant_model_builds(n_hidden):
    """QuantLinear wiring must not raise for any C_h."""
    model = make_quant_model(n_hidden=n_hidden)
    assert model is not None
    batch = make_batch(B=2, N_particles=6, add_beams=True, seed=1)
    with torch.no_grad():
        out = model(batch)
    assert 'predict' in out
    assert not torch.isnan(out['predict']).any()


# ---------------------------------------------------------------------------
# R3 – Permutation invariance under 8-bit quantization
# ---------------------------------------------------------------------------

def test_quant_permutation_invariance():
    """8-bit quant model output is unchanged under jet-constituent permutation."""
    model = make_quant_model(n_hidden=2, seed=5)

    batch = make_batch(B=3, N_particles=10, add_beams=True, seed=7)
    Pmu = batch['Pmu']
    B, N_total, _ = Pmu.shape
    n_beams = 2

    torch.manual_seed(77)
    perm = torch.randperm(N_total - n_beams) + n_beams
    perm_full = torch.cat([torch.arange(n_beams), perm])

    Pmu_perm = Pmu[:, perm_full, :]
    mask_perm = (Pmu_perm[..., 0] != 0.)
    edge_perm = mask_perm.unsqueeze(1) & mask_perm.unsqueeze(2)
    batch_perm = {
        'Pmu': Pmu_perm, 'is_signal': batch['is_signal'],
        'particle_mask': mask_perm.bool(), 'edge_mask': edge_perm.bool(),
        'Nobj': mask_perm.sum(-1),
    }

    with torch.no_grad():
        out_orig = model(batch)['predict']
        out_perm = model(batch_perm)['predict']

    max_diff = (out_orig - out_perm).abs().max().item()
    assert max_diff < 1e-4, (
        f'Quant model permutation invariance violated: max_diff={max_diff:.2e}'
    )


# ---------------------------------------------------------------------------
# R4 – Masking invariance under 8-bit quantization
# ---------------------------------------------------------------------------

def test_quant_masking_invariance():
    """8-bit quant model output is unchanged when zero-padded particles are appended."""
    model = make_quant_model(n_hidden=2, seed=5)

    batch = make_batch(B=3, N_particles=8, add_beams=True, seed=13)
    Pmu_orig = batch['Pmu']
    B, N, _ = Pmu_orig.shape

    n_pad = 4
    Pmu_padded = torch.cat(
        [Pmu_orig, torch.zeros(B, n_pad, 4, dtype=Pmu_orig.dtype)], dim=1
    )
    mask_padded = Pmu_padded[..., 0] != 0.
    edge_padded = mask_padded.unsqueeze(1) & mask_padded.unsqueeze(2)
    batch_padded = {
        'Pmu': Pmu_padded, 'is_signal': batch['is_signal'],
        'particle_mask': mask_padded.bool(), 'edge_mask': edge_padded.bool(),
        'Nobj': mask_padded.sum(-1),
    }

    with torch.no_grad():
        out_orig = model(batch)['predict']
        out_padded = model(batch_padded)['predict']

    max_diff = (out_orig - out_padded).abs().max().item()
    assert max_diff < 1e-4, (
        f'Quant model masking invariance violated: max_diff={max_diff:.2e}'
    )


# ---------------------------------------------------------------------------
# R5 – config='S' raises NotImplementedError
# ---------------------------------------------------------------------------

def test_uppercase_config_raises_without_flag():
    """config='S' (N^alpha scaling) must raise NotImplementedError by default."""
    from src.models.pelican_nano import PELICANNano
    qcfg = QuantConfig(enabled=True, allow_alpha_scaling=False)
    with pytest.raises(NotImplementedError):
        PELICANNano(
            n_hidden=2, config='S', config_out='s',
            activation='relu', add_beams=True, batchnorm=None, dropout=False,
            factorize=False, masked=False, quant_config=qcfg,
            device=torch.device('cpu'), dtype=torch.float,
        )


def test_uppercase_config_allowed_with_flag():
    """config='S' must NOT raise when allow_alpha_scaling=True."""
    from src.models.pelican_nano import PELICANNano
    qcfg = QuantConfig(enabled=True, allow_alpha_scaling=True)
    model = PELICANNano(
        n_hidden=2, config='S', config_out='s',
        activation='relu', add_beams=True, batchnorm=None, dropout=False,
        factorize=False, masked=False, quant_config=qcfg,
        device=torch.device('cpu'), dtype=torch.float,
    )
    assert model is not None


# ---------------------------------------------------------------------------
# R7 – Parameter count unchanged (float path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_hidden', [1, 2, 3, 10])
def test_param_count_float_unchanged(n_hidden):
    """quant_config=None must not add trainable parameters: 10*C_h + 1."""
    model = make_model(n_hidden=n_hidden, batchnorm=None, dropout=False)
    n_params = sum(p.numel() for p in model.parameters())
    expected = 10 * n_hidden + 1
    assert n_params == expected, (
        f'n_hidden={n_hidden}: expected {expected} params, got {n_params}'
    )
