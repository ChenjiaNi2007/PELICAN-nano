"""
Shared pytest fixtures for PELICAN-nano QAT refactor test suite.
"""
import math
import torch
import pytest
import sys
import os

# Make the repo root importable from any test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.pelican_nano import PELICANNano


def make_batch(B=4, N_particles=10, add_beams=True, dtype=torch.float, seed=42):
    """
    Create a synthetic batch of B events, each with N_particles active particles
    plus (optionally) 2 beam particles prepended, all with dtype.

    The batch follows the exact format that PELICANNano.forward() expects:
        Pmu           [B, N_total, 4]
        particle_mask [B, N_total]   bool
        edge_mask     [B, N_total, N_total] bool
        is_signal     [B]
    """
    torch.manual_seed(seed)

    if add_beams:
        n_beams = 2
        beams = torch.tensor(
            [[1., 0., 0., 1.], [1., 0., 0., -1.]], dtype=dtype
        ).unsqueeze(0).expand(B, -1, -1)
    else:
        n_beams = 0

    # Random timelike particles: E = |p3| + epsilon so E^2 > |p|^2
    p3 = torch.randn(B, N_particles, 3, dtype=dtype)
    E = p3.norm(dim=-1, keepdim=True) + 0.1 + torch.rand(B, N_particles, 1, dtype=dtype)
    jets = torch.cat([E, p3], dim=-1)  # [B, N_particles, 4]

    if add_beams:
        Pmu = torch.cat([beams, jets], dim=1)  # [B, n_beams+N_particles, 4]
    else:
        Pmu = jets

    particle_mask = Pmu[..., 0] != 0.  # all active (no zeros were added)
    edge_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)

    return {
        'Pmu': Pmu,
        'is_signal': torch.zeros(B, dtype=torch.long),
        'particle_mask': particle_mask.bool(),
        'edge_mask': edge_mask.bool(),
        'Nobj': particle_mask.sum(-1),
    }


def make_model(n_hidden=2, batchnorm=None, dropout=False, config='s',
               config_out='s', factorize=False, average_nobj=49,
               device=torch.device('cpu'), dtype=torch.float, seed=0):
    """
    Build a PELICANNano model with fixed init seed.
    Default args match the minimal float-path configuration used for tests
    (no BN, no dropout, factorize=False).
    """
    torch.manual_seed(seed)
    model = PELICANNano(
        n_hidden,
        activate_agg=False,
        activate_lin=True,
        activation='relu',
        add_beams=True,
        config=config,
        config_out=config_out,
        average_nobj=average_nobj,
        factorize=factorize,
        masked=True,
        activate_agg_out=False,
        activate_lin_out=False,
        scale=1.0,
        dropout=dropout,
        drop_rate=0.0,
        drop_rate_out=0.0,
        batchnorm=batchnorm,
        device=device,
        dtype=dtype,
    )
    return model


@pytest.fixture
def sample_batch():
    return make_batch(B=4, N_particles=10, add_beams=True)


@pytest.fixture
def float_model():
    torch.manual_seed(0)
    model = make_model(n_hidden=2)
    model.eval()
    return model
