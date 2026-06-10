"""
Phase 1 tests: einsum → nn.Linear refactor (float path only).

Covers:
  - Einsum↔Linear mathematical equivalence for Eq2to2 and Eq2to0
  - Golden-output non-regression: Phase-1 model loaded from converted
    Phase-0 checkpoint must match the committed golden output exactly
  - Checkpoint converter round-trip (convert and load without error)
  - Phase-0 param count still holds after refactor
  - Permutation / masking invariance preserved after refactor
"""
import os
import sys
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.conftest import make_batch, make_model
from scripts.convert_checkpoint import convert_state_dict

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_float_output.pt')
PHASE0_SD_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'phase0_state_dict.pt')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_converted_model(n_hidden=2):
    """Load Phase-0 state dict, convert, return Phase-1 model + state dict."""
    old_sd = torch.load(PHASE0_SD_PATH, map_location='cpu', weights_only=True)
    new_sd = convert_state_dict(old_sd)
    model = make_model(n_hidden=n_hidden)  # fresh Phase-1 model (any init)
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Mathematical equivalence: einsum == Linear
# ---------------------------------------------------------------------------

def test_einsum_eq_linear_eq2to2():
    """
    ops reshaped to [B,N,N,C_in*6] @ W.T  ==  einsum('dsb,ndbij->nijs', coefs, ops)
    for random coefs/ops.
    """
    torch.manual_seed(7)
    B, N, C_in, C_out, basis = 3, 8, 1, 4, 6
    ops = torch.randn(B, C_in, basis, N, N)
    coefs = torch.randn(C_in, C_out, basis)

    # Reference einsum
    out_einsum = torch.einsum('dsb,ndbij->nijs', coefs, ops)

    # Linear equivalent
    W = coefs.permute(1, 0, 2).reshape(C_out, C_in * basis)
    ops_flat = ops.permute(0, 3, 4, 1, 2).reshape(B, N, N, C_in * basis)
    out_linear = ops_flat @ W.T

    assert (out_einsum - out_linear).abs().max().item() < 1e-5, \
        'Eq2to2 einsum != Linear'


def test_einsum_eq_linear_eq2to0():
    """
    ops reshaped to [B, C_h*2] @ W.T + bias  ==  einsum + bias  for Eq2to0.
    """
    torch.manual_seed(11)
    B, C_h, C_out, basis = 3, 4, 1, 2
    ops = torch.randn(B, C_h, basis)
    coefs = torch.randn(C_h, C_out, basis)
    bias = torch.randn(1, C_out)

    out_einsum = torch.einsum('dsb,ndb->ns', coefs, ops) + bias

    W = coefs.permute(1, 0, 2).reshape(C_out, C_h * basis)
    ops_flat = ops.reshape(B, C_h * basis)
    out_linear = ops_flat @ W.T + bias.squeeze(0)

    assert (out_einsum - out_linear).abs().max().item() < 1e-5, \
        'Eq2to0 einsum != Linear'


# ---------------------------------------------------------------------------
# Checkpoint conversion round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_conversion_loads():
    """Converted Phase-0 state dict loads into Phase-1 model with strict=True."""
    model = _load_converted_model(n_hidden=2)
    assert model is not None


def test_checkpoint_conversion_no_coefs_keys():
    """After conversion, no 'coefs' keys remain (they're replaced by mixing.weight)."""
    old_sd = torch.load(PHASE0_SD_PATH, map_location='cpu', weights_only=True)
    new_sd = convert_state_dict(old_sd)
    coefs_keys = [k for k in new_sd if k.endswith('.coefs')]
    assert len(coefs_keys) == 0, f'coefs keys still present: {coefs_keys}'


# ---------------------------------------------------------------------------
# Golden output: converted Phase-0 checkpoint → same output as Phase-0
# ---------------------------------------------------------------------------

def test_golden_output_after_conversion():
    """
    Phase-1 model loaded from converted Phase-0 checkpoint reproduces the
    committed golden output bit-for-bit (tolerance 1e-5, float32 rounding).
    """
    assert os.path.exists(GOLDEN_PATH), f'Golden file missing: {GOLDEN_PATH}'
    assert os.path.exists(PHASE0_SD_PATH), f'Phase-0 fixture missing: {PHASE0_SD_PATH}'

    golden = torch.load(GOLDEN_PATH, weights_only=True)

    model = _load_converted_model(n_hidden=2)
    batch = make_batch(B=2, N_particles=6, add_beams=True, dtype=torch.float, seed=42)

    with torch.no_grad():
        out = model(batch)['predict']

    max_diff = (out - golden).abs().max().item()
    assert max_diff < 1e-5, (
        f'Phase-1 output diverged from Phase-0 golden by {max_diff:.2e} (threshold 1e-5)'
    )


# ---------------------------------------------------------------------------
# Param count unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_hidden', [1, 2, 3, 10])
def test_param_count_unchanged(n_hidden):
    """Phase-1 refactor must not change the parameter count: 10*C_h + 1."""
    model = make_model(n_hidden=n_hidden, batchnorm=None, dropout=False)
    n_params = sum(p.numel() for p in model.parameters())
    expected = 10 * n_hidden + 1
    assert n_params == expected, (
        f'n_hidden={n_hidden}: expected {expected} params, got {n_params}'
    )


# ---------------------------------------------------------------------------
# Symmetries preserved
# ---------------------------------------------------------------------------

def test_permutation_invariance_phase1():
    """Permutation invariance holds for Phase-1 model (converted checkpoint)."""
    model = _load_converted_model(n_hidden=2)

    batch = make_batch(B=3, N_particles=10, add_beams=True, seed=55)
    Pmu = batch['Pmu']
    B, N_total, _ = Pmu.shape
    n_beams = 2

    torch.manual_seed(33)
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
    assert max_diff < 1e-5, f'Permutation broke: max_diff={max_diff:.2e}'


def test_masking_invariance_phase1():
    """Masking invariance holds for Phase-1 model (converted checkpoint)."""
    model = _load_converted_model(n_hidden=2)

    batch = make_batch(B=3, N_particles=8, add_beams=True, seed=77)
    Pmu_orig = batch['Pmu']
    B, N, _ = Pmu_orig.shape

    n_pad = 3
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
    assert max_diff < 1e-5, f'Masking broke: max_diff={max_diff:.2e}'
