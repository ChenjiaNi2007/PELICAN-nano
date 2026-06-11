"""
scripts/export_qonnx.py — Export a trained QAT nanoPELICAN model to ONNX/QONNX.

Usage
-----
    python scripts/export_qonnx.py \\
        --checkpoint model/best.pt \\
        --output model/pelican_nano.onnx \\
        [--n-hidden 2] \\
        [--N 10] \\
        [--quant-checkpoint]   # if the checkpoint was saved from a quant model

The script exports using the legacy TorchScript-based torch.onnx.export
(dynamo=False) so that the graph is traced once for a fixed input shape.

Input nodes:  Pmu [B, N, 4]  particle_mask [B, N]  edge_mask [B, N, N]
Output node:  predict [B, 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano


class _ExportWrapper(torch.nn.Module):
    """Thin wrapper that takes concrete tensors instead of a dict."""
    def __init__(self, pelican: PELICANNano):
        super().__init__()
        self.pelican = pelican

    def forward(
        self,
        Pmu: torch.Tensor,
        particle_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        data = {
            'Pmu': Pmu,
            'particle_mask': particle_mask.bool(),
            'edge_mask': edge_mask.bool(),
        }
        return self.pelican(data)['predict']


def load_model(
    checkpoint_path: str,
    n_hidden: int,
    quant: bool,
    device: torch.device,
) -> PELICANNano:
    qcfg = QuantConfig(enabled=quant)
    model = PELICANNano(
        n_hidden=n_hidden,
        activate_agg=False, activate_lin=True,
        activation='relu', add_beams=True,
        config='s', config_out='s',
        average_nobj=49, factorize=False, masked=False,
        batchnorm=None, dropout=False,
        quant_config=qcfg,
        device=device, dtype=torch.float,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get('model_state', ckpt)

    # Brevitas scale params require a calibration pass before load_state_dict
    # if the model has quantizers.  We do a dummy train-mode pass first.
    if quant:
        B, N = 2, 10
        dummy_Pmu = torch.zeros(B, N, 4)
        dummy_Pmu[:, :2, 0] = 1.0  # beam-like: energy = 1
        dummy_pm = (dummy_Pmu[..., 0] != 0.)
        dummy_em = dummy_pm.unsqueeze(1) & dummy_pm.unsqueeze(2)
        dummy_batch = {
            'Pmu': dummy_Pmu, 'is_signal': torch.zeros(B),
            'particle_mask': dummy_pm.bool(), 'edge_mask': dummy_em.bool(),
            'Nobj': dummy_pm.sum(-1),
        }
        model.train()
        with torch.no_grad():
            model(dummy_batch)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_onnx(
    model: PELICANNano,
    output_path: str,
    N: int = 10,
    batch_size: int = 1,
    opset: int = 17,
) -> None:
    """Export model to ONNX using TorchScript tracing."""
    wrapper = _ExportWrapper(model)
    wrapper.eval()

    Pmu = torch.zeros(batch_size, N, 4)
    Pmu[:, :2, 0] = 1.0  # give two beam-like particles so masks are non-trivial
    pm = (Pmu[..., 0] != 0.).bool()
    em = (pm.unsqueeze(1) & pm.unsqueeze(2)).bool()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (Pmu, pm, em),
            output_path,
            dynamo=False,
            opset_version=opset,
            input_names=['Pmu', 'particle_mask', 'edge_mask'],
            output_names=['predict'],
            dynamic_axes={
                'Pmu': {0: 'batch', 1: 'N'},
                'particle_mask': {0: 'batch', 1: 'N'},
                'edge_mask': {0: 'batch', 1: 'N', 2: 'N'},
                'predict': {0: 'batch'},
            },
        )
    print(f'Exported ONNX to {output_path} ({os.path.getsize(output_path)} bytes)')


def dump_weights(model: PELICANNano) -> dict:
    """
    Return a dict of integer weights and scale factors for every QuantLinear
    layer and QuantIdentity quantizer in the model.

    Each entry is keyed by module path and contains:
      - 'weight_int'  : integer weight tensor (if QuantLinear)
      - 'weight_scale': per-tensor or per-channel scale
      - 'act_scale'   : activation quantizer scale (where present)

    For float (non-quant) models returns raw float weights.
    """
    result = {}
    try:
        import brevitas.nn as bnn
    except ImportError:
        bnn = None

    for name, module in model.named_modules():
        if bnn is not None and isinstance(module, bnn.QuantLinear):
            entry: dict = {}
            try:
                qt = module.weight_quant(module.weight)
                entry['weight_int'] = qt.int(float_datatype=False)
                entry['weight_scale'] = qt.scale.detach().clone()
                entry['weight_zero_point'] = qt.zero_point.detach().clone()
            except Exception:
                entry['weight_float'] = module.weight.detach().clone()
            if module.bias is not None:
                entry['bias'] = module.bias.detach().clone()
            result[name] = entry

        elif bnn is not None and isinstance(module, bnn.QuantIdentity):
            entry = {}
            try:
                dummy = torch.zeros(1)
                qt = module.act_quant(dummy)
                entry['act_scale'] = qt.scale.detach().clone()
                entry['act_zero_point'] = qt.zero_point.detach().clone()
            except Exception:
                pass
            if entry:
                result[name] = entry

        elif isinstance(module, torch.nn.Linear):
            result[name] = {
                'weight_float': module.weight.detach().clone(),
                'bias': module.bias.detach().clone() if module.bias is not None else None,
            }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Export nanoPELICAN to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--output', required=True, help='Output .onnx file path')
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--N', type=int, default=10, help='Number of particles for tracing')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--quant-checkpoint', action='store_true',
                        help='Checkpoint was saved from a quantized model')
    parser.add_argument('--dump-weights', type=str, default='',
                        help='If set, also dump integer weights to this .pt file')
    args = parser.parse_args()

    device = torch.device('cpu')
    model = load_model(args.checkpoint, args.n_hidden, args.quant_checkpoint, device)
    print(f'Loaded {"quant" if args.quant_checkpoint else "float"} model '
          f'(n_hidden={args.n_hidden}) from {args.checkpoint}')

    export_onnx(model, args.output, N=args.N, opset=args.opset)

    if args.dump_weights:
        weights = dump_weights(model)
        torch.save(weights, args.dump_weights)
        print(f'Weight dump saved to {args.dump_weights} ({len(weights)} entries)')


if __name__ == '__main__':
    main()
