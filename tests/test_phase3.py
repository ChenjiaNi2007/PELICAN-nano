"""
Phase 3 tests: CLI arg wiring and checkpoint round-trip with quantizer state.

Covers:
  C1  – New QAT args parse without error; defaults are correct
  C2  – --quant flag constructs QuantConfig(enabled=True) with correct bit widths
  C3  – Float model (--no-quant) produces same output as quant_config=None baseline
  C4  – Checkpoint save → load preserves quantizer scale buffers exactly
  C5  – Cross-load (float ckpt into quant model) fails gracefully with RuntimeError
"""
import os
import sys
import copy
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.conftest import make_batch, make_model
from src.layers.quant import QuantConfig

try:
    import brevitas.nn as _bnn
    _BREVITAS = True
except ImportError:
    _BREVITAS = False

pytestmark = pytest.mark.skipif(
    not _BREVITAS, reason='brevitas not installed — skipping Phase-3 tests'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args(extra_args=None):
    """Parse args using setup_argparse directly (avoids sys.argv)."""
    from src.trainer.args import setup_argparse
    parser = setup_argparse()
    base = [
        '--datadir', 'data/sample_data',
        '--target', 'is_signal',
        '--prefix', 'pytest_phase3',
    ]
    argv = base + (extra_args or [])
    return parser.parse_args(argv)


def _build_quant_config_from_args(args):
    """Mirror the logic in train_pelican_nano.py."""
    return QuantConfig(
        enabled=args.quant,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        input_bit_width=args.input_bit_width,
        weight_per_channel=args.weight_per_channel,
        po2_scales=args.po2_scales,
        allow_alpha_scaling=args.allow_alpha_scaling,
    )


def _make_quant_model_from_args(args, n_hidden=2, seed=0):
    from src.models.pelican_nano import PELICANNano
    qcfg = _build_quant_config_from_args(args)
    torch.manual_seed(seed)
    model = PELICANNano(
        n_hidden=n_hidden,
        activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True,
        config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False,
        batchnorm=None, dropout=False,
        quant_config=qcfg,
        device=torch.device('cpu'), dtype=torch.float,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# C1 – Arg parsing defaults
# ---------------------------------------------------------------------------

def test_qat_arg_defaults():
    """New QAT args must be present with correct defaults."""
    args = _parse_args()
    assert hasattr(args, 'quant'), '--quant arg missing'
    assert args.quant is False
    assert args.weight_bit_width == 8
    assert args.act_bit_width == 8
    assert args.input_bit_width == 8
    assert args.weight_per_channel is False
    assert args.po2_scales is False
    assert args.allow_alpha_scaling is False


# ---------------------------------------------------------------------------
# C2 – --quant flag sets enabled=True with custom bit widths
# ---------------------------------------------------------------------------

def test_qat_arg_quant_flag():
    """--quant sets enabled=True; bit-width overrides propagate."""
    args = _parse_args(['--quant', '--weight-bit-width', '4', '--act-bit-width', '6'])
    qcfg = _build_quant_config_from_args(args)
    assert qcfg.enabled is True
    assert qcfg.weight_bit_width == 4
    assert qcfg.act_bit_width == 6
    assert qcfg.input_bit_width == 8  # unchanged default


def test_qat_arg_no_quant_flag():
    """--no-quant keeps enabled=False."""
    args = _parse_args(['--no-quant'])
    qcfg = _build_quant_config_from_args(args)
    assert qcfg.enabled is False


# ---------------------------------------------------------------------------
# C3 – Float model from --no-quant gives same output as quant_config=None
# ---------------------------------------------------------------------------

def test_float_model_from_args_matches_baseline():
    """Constructing model via args with --no-quant matches the direct float baseline."""
    args = _parse_args(['--no-quant'])
    from src.models.pelican_nano import PELICANNano

    # Both use same seed and same arch; quant_config.enabled=False vs None
    torch.manual_seed(42)
    model_via_args = PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True, config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False, batchnorm=None, dropout=False,
        quant_config=_build_quant_config_from_args(args),
        device=torch.device('cpu'), dtype=torch.float,
    )
    model_via_args.eval()

    torch.manual_seed(42)
    model_no_qcfg = PELICANNano(
        n_hidden=2, activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True, config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False, batchnorm=None, dropout=False,
        quant_config=None,
        device=torch.device('cpu'), dtype=torch.float,
    )
    model_no_qcfg.eval()

    batch = make_batch(B=2, N_particles=6, add_beams=True, seed=99)
    with torch.no_grad():
        out_a = model_via_args(batch)['predict']
        out_b = model_no_qcfg(batch)['predict']

    max_diff = (out_a - out_b).abs().max().item()
    assert max_diff < 1e-6, f'Arg-constructed float model differs from baseline: {max_diff:.2e}'


# ---------------------------------------------------------------------------
# C4 – Checkpoint round-trip preserves quantizer scale buffers
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_preserves_quant_scales():
    """
    Save quant model state_dict to disk, reload into fresh quant model.
    Brevitas ParameterFromRuntimeStats scales are only initialised during
    training-mode forward passes, so both models need one train-mode step.
    """
    args = _parse_args(['--quant'])
    batch = make_batch(B=4, N_particles=8, add_beams=True, seed=10)

    # Model 1: calibrate in train mode, then evaluate
    model = _make_quant_model_from_args(args, n_hidden=2, seed=3)
    model.train()
    with torch.no_grad():
        model(batch)  # initialises scaling_impl.value for all quantizers
    model.eval()
    with torch.no_grad():
        out_before = model(batch)['predict']

    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({'model_state': model.state_dict()}, ckpt_path)

        # Model 2: calibrate in train mode to initialise scale params,
        # then overwrite with the saved state dict
        model2 = _make_quant_model_from_args(args, n_hidden=2, seed=99)
        model2.train()
        with torch.no_grad():
            model2(batch)  # initialises scaling_impl.value
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model2.load_state_dict(ckpt['model_state'], strict=True)
        model2.eval()

        with torch.no_grad():
            out_after = model2(batch)['predict']

        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff < 1e-6, (
            f'Checkpoint round-trip changed output by {max_diff:.2e}'
        )
    finally:
        os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# C5 – Cross-load: float checkpoint into quant model fails with RuntimeError
# ---------------------------------------------------------------------------

def test_float_ckpt_into_quant_model_fails():
    """
    Loading a float model checkpoint into a quant model must raise RuntimeError
    because the quant model has extra quantizer parameter keys.
    The quant model needs a train-mode calibration pass first so its
    scaling_impl.value params exist (and are absent from the float state dict).
    """
    # Float model state dict
    float_model = make_model(n_hidden=2, seed=0)
    float_sd = float_model.state_dict()

    # Quant model: calibrate so quantizer params are present in model
    args = _parse_args(['--quant'])
    batch = make_batch(B=2, N_particles=6, add_beams=True, seed=1)
    quant_model = _make_quant_model_from_args(args, n_hidden=2, seed=0)
    quant_model.train()
    with torch.no_grad():
        quant_model(batch)

    with pytest.raises(RuntimeError):
        quant_model.load_state_dict(float_sd, strict=True)
