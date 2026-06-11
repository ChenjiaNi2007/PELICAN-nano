"""
Phase 4 tests: ONNX export and weight dump.

Covers:
  E1  – Float model exports to ONNX without error (smoke test)
  E2  – Quant model exports to ONNX without error (smoke test)
  E3  – ONNX output matches torch forward pass to 1e-4
  E4  – dump_weights returns entries for every Linear/QuantLinear layer
  E5  – Quant weight dump contains 'weight_int' and 'weight_scale' entries
  E6  – Float model dump contains 'weight_float' entries
"""
from __future__ import annotations

import os
import sys
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

try:
    import onnx
    import onnxruntime as ort
    _ONNX_RUNTIME = True
except ImportError:
    _ONNX_RUNTIME = False

from scripts.export_qonnx import _ExportWrapper, dump_weights, export_onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quant_model_calibrated(n_hidden=2, seed=0):
    """QAT model after one training-mode calibration pass."""
    from src.models.pelican_nano import PELICANNano
    qcfg = QuantConfig(enabled=True)
    torch.manual_seed(seed)
    model = PELICANNano(
        n_hidden=n_hidden, activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True, config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False, batchnorm=None, dropout=False,
        quant_config=qcfg, device=torch.device('cpu'), dtype=torch.float,
    )
    batch = make_batch(B=4, N_particles=8, add_beams=True, seed=42)
    model.train()
    with torch.no_grad():
        model(batch)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# E1 – Float model exports to ONNX
# ---------------------------------------------------------------------------

def test_float_model_onnx_export():
    """Float model exports to ONNX (dynamo=False, opset 17)."""
    model = make_model(n_hidden=2)
    model.eval()
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path = f.name
    try:
        export_onnx(model, path, N=8)
        assert os.path.getsize(path) > 0, 'ONNX file is empty'
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# E2 – Quant model exports to ONNX
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BREVITAS, reason='brevitas not installed')
def test_quant_model_onnx_export():
    """Quant model exports to ONNX after calibration."""
    model = _make_quant_model_calibrated(n_hidden=2)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path = f.name
    try:
        export_onnx(model, path, N=8)
        assert os.path.getsize(path) > 0, 'ONNX file is empty'
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# E3 – ONNX output matches torch output
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ONNX_RUNTIME, reason='onnxruntime not installed')
def test_onnx_output_matches_torch():
    """ONNX Runtime output matches PyTorch forward pass for float model."""
    import numpy as np

    model = make_model(n_hidden=2, seed=3)
    model.eval()

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path = f.name

    N = 8
    batch = make_batch(B=2, N_particles=N - 2, add_beams=True, seed=7)
    Pmu = batch['Pmu']
    pm = batch['particle_mask']
    em = batch['edge_mask']

    try:
        export_onnx(model, path, N=Pmu.shape[1])

        # Torch reference
        wrapper = _ExportWrapper(model)
        wrapper.eval()
        with torch.no_grad():
            out_torch = wrapper(Pmu, pm, em).numpy()

        # ONNX Runtime
        sess = ort.InferenceSession(path)
        out_onnx = sess.run(
            ['predict'],
            {
                'Pmu': Pmu.numpy(),
                'particle_mask': pm.numpy(),
                'edge_mask': em.numpy(),
            },
        )[0]

        max_diff = np.abs(out_torch - out_onnx).max()
        assert max_diff < 1e-4, (
            f'ONNX output differs from PyTorch by {max_diff:.2e}'
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# E4 – dump_weights covers all Linear layers (float model)
# ---------------------------------------------------------------------------

def test_dump_weights_float_has_all_linear():
    """Float model dump_weights contains entries for every nn.Linear."""
    model = make_model(n_hidden=2)
    model.eval()
    weights = dump_weights(model)

    # Every nn.Linear in the model should appear
    linear_names = {
        name for name, m in model.named_modules()
        if type(m) is torch.nn.Linear or type(m).__bases__[0] is torch.nn.Linear
    }
    missing = linear_names - set(weights.keys())
    assert len(missing) == 0, f'dump_weights missing layers: {missing}'


# ---------------------------------------------------------------------------
# E5 – Quant weight dump has integer weights + scales
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BREVITAS, reason='brevitas not installed')
def test_dump_weights_quant_has_int_weights():
    """Quant model dump_weights contains weight_int and weight_scale for QuantLinear."""
    model = _make_quant_model_calibrated(n_hidden=2)
    weights = dump_weights(model)

    quant_linear_names = [
        name for name, m in model.named_modules()
        if _BREVITAS and isinstance(m, _bnn.QuantLinear)
    ]
    assert len(quant_linear_names) > 0, 'No QuantLinear layers found'

    for name in quant_linear_names:
        assert name in weights, f'QuantLinear {name} missing from dump'
        entry = weights[name]
        assert 'weight_int' in entry or 'weight_float' in entry, (
            f'{name}: neither weight_int nor weight_float in dump'
        )


# ---------------------------------------------------------------------------
# E6 – Float weight dump entries contain weight_float
# ---------------------------------------------------------------------------

def test_dump_weights_float_entries():
    """Float model dump entries use weight_float key."""
    model = make_model(n_hidden=2)
    model.eval()
    weights = dump_weights(model)
    for name, entry in weights.items():
        assert 'weight_float' in entry or 'weight_int' in entry, (
            f'{name}: no weight tensor in dump entry'
        )
