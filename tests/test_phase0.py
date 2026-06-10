"""
Phase 0 tests: baseline float-model hygiene.

Covers:
  - Parameter count regression: 10*C_h + 1 (no BN, factorize=False)
  - Permutation invariance of the output score
  - Masking invariance (adding zero-padded particles leaves output unchanged)
  - Forward-pass golden-output non-regression (seeded, deterministic)
"""
import os
import math
import torch
import pytest

from tests.conftest import make_batch, make_model

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_float_output.pt')
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
PHASE0_STATE_DICT_PATH = os.path.join(FIXTURE_DIR, 'phase0_state_dict.pt')

# Import the checkpoint converter if available (Phase 1+)
try:
    from scripts.convert_checkpoint import convert_state_dict as _convert_sd
except ImportError:
    _convert_sd = None


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_hidden', [1, 2, 3, 10])
def test_param_count(n_hidden):
    """Float nano with one output channel and no BN: 10*C_h + 1 params."""
    model = make_model(n_hidden=n_hidden, batchnorm=None, dropout=False)
    n_params = sum(p.numel() for p in model.parameters())
    expected = 10 * n_hidden + 1
    assert n_params == expected, (
        f'n_hidden={n_hidden}: expected {expected} params, got {n_params}'
    )


# ---------------------------------------------------------------------------
# Forward pass sanity
# ---------------------------------------------------------------------------

def test_forward_no_nan(sample_batch, float_model):
    """Forward pass on a small batch produces finite logits."""
    with torch.no_grad():
        out = float_model(sample_batch)
    assert 'predict' in out
    assert not torch.isnan(out['predict']).any(), 'NaN in predictions'
    assert not torch.isinf(out['predict']).any(), 'Inf in predictions'


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------

def test_permutation_invariance():
    """
    Randomly permuting the jet constituents (non-beam particles) leaves the
    output score unchanged, for n_hidden in {1, 2, 3, 10}.
    """
    for n_hidden in [1, 2, 3, 10]:
        model = make_model(n_hidden=n_hidden)
        model.eval()

        batch = make_batch(B=4, N_particles=12, add_beams=True, seed=7)
        Pmu = batch['Pmu']  # [B, N_total, 4]
        B, N_total, _ = Pmu.shape
        n_beams = 2

        # Permute only the jet-constituent part (indices n_beams..N_total-1)
        torch.manual_seed(99)
        perm = torch.randperm(N_total - n_beams) + n_beams  # permutation of constituent indices
        perm_full = torch.cat([torch.arange(n_beams), perm])  # keep beams in place

        Pmu_perm = Pmu[:, perm_full, :]
        mask_perm = (Pmu_perm[..., 0] != 0.)
        edge_perm = mask_perm.unsqueeze(1) & mask_perm.unsqueeze(2)

        batch_perm = {
            'Pmu': Pmu_perm,
            'is_signal': batch['is_signal'],
            'particle_mask': mask_perm.bool(),
            'edge_mask': edge_perm.bool(),
            'Nobj': mask_perm.sum(-1),
        }

        with torch.no_grad():
            out_orig = model(batch)['predict']
            out_perm = model(batch_perm)['predict']

        max_diff = (out_orig - out_perm).abs().max().item()
        assert max_diff < 1e-5, (
            f'n_hidden={n_hidden}: permutation changed output by {max_diff:.2e}'
        )


# ---------------------------------------------------------------------------
# Masking invariance
# ---------------------------------------------------------------------------

def test_masking_invariance():
    """
    Appending zero-padded particles (particle_mask=False) leaves the output
    score unchanged, for n_hidden in {1, 2, 3, 10}.
    """
    for n_hidden in [1, 2, 3, 10]:
        model = make_model(n_hidden=n_hidden)
        model.eval()

        batch = make_batch(B=3, N_particles=8, add_beams=True, seed=13)
        Pmu_orig = batch['Pmu']  # [B, N, 4]
        B, N, _ = Pmu_orig.shape

        # Append 4 zero-padded particles
        n_pad = 4
        Pmu_padded = torch.cat(
            [Pmu_orig, torch.zeros(B, n_pad, 4, dtype=Pmu_orig.dtype)], dim=1
        )
        mask_padded = Pmu_padded[..., 0] != 0.
        edge_padded = mask_padded.unsqueeze(1) & mask_padded.unsqueeze(2)

        batch_padded = {
            'Pmu': Pmu_padded,
            'is_signal': batch['is_signal'],
            'particle_mask': mask_padded.bool(),
            'edge_mask': edge_padded.bool(),
            'Nobj': mask_padded.sum(-1),
        }

        with torch.no_grad():
            out_orig = model(batch)['predict']
            out_padded = model(batch_padded)['predict']

        max_diff = (out_orig - out_padded).abs().max().item()
        assert max_diff < 1e-5, (
            f'n_hidden={n_hidden}: padding changed output by {max_diff:.2e}'
        )


# ---------------------------------------------------------------------------
# Golden-output non-regression
# ---------------------------------------------------------------------------

def _get_golden_batch():
    """Fixed seed batch used for golden output."""
    return make_batch(B=2, N_particles=6, add_beams=True, dtype=torch.float, seed=42)


def _get_golden_model():
    """Fixed seed model used for golden output."""
    return make_model(n_hidden=2, batchnorm=None, dropout=False, seed=0)


def test_golden_output():
    """
    The committed Phase-0 weights must produce the committed golden output.

    Phase 0: loads the state dict and checks it directly.
    Phase 1+: if the state dict uses the old key names (coefs), converts via
    convert_state_dict before loading.  The golden file itself never changes.
    """
    assert os.path.exists(GOLDEN_PATH), f'Golden file missing: {GOLDEN_PATH}'
    assert os.path.exists(PHASE0_STATE_DICT_PATH), (
        f'Phase-0 fixture missing: {PHASE0_STATE_DICT_PATH}\n'
        'Re-run the Phase-0 tests once to regenerate it.'
    )

    old_sd = torch.load(PHASE0_STATE_DICT_PATH, map_location='cpu', weights_only=True)
    model = make_model(n_hidden=2)
    try:
        model.load_state_dict(old_sd, strict=True)
    except RuntimeError:
        # Phase 1+: key names changed; convert first.
        assert _convert_sd is not None, (
            'State dict has incompatible keys and convert_checkpoint is not available.'
        )
        new_sd = _convert_sd(old_sd)
        model.load_state_dict(new_sd, strict=True)
    model.eval()

    batch = _get_golden_batch()
    with torch.no_grad():
        out = model(batch)['predict']

    golden = torch.load(GOLDEN_PATH, weights_only=True)
    max_diff = (out - golden).abs().max().item()
    assert max_diff < 1e-5, (
        f'Float output drifted from golden by {max_diff:.2e} (expected < 1e-5)'
    )
