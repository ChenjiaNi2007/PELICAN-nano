"""
scripts/check_scales.py

Inspect every quantizer in a trained QAT nanoPELICAN model and report, for each:
  - the learned scale
  - the nearest power-of-two exponent k  (scale ~= 2^-k)
  - the implied number of fractional bits for an ap_fixed<W, W-k> typedef
  - for weight quantizers: the integer-weight range actually used

Run from the repo root AFTER training, e.g.:
    python3 scripts/check_scales.py \
        --checkpoint model/fpga_model_qat_best.pt \
        --n-hidden 2 --weight-bit-width 24 --act-bit-width 24 --input-bit-width 24
Make sure the args match what you trained with.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.disable(logging.CRITICAL)

from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano


def po2_exponent(scale: float) -> float:
    """Return k such that scale ~= 2^-k (k = -log2(scale))."""
    return -math.log2(scale)


def report_scale(name: str, scale_tensor: torch.Tensor) -> None:
    scale = float(scale_tensor.reshape(-1)[0])  # per-tensor: one value
    k = po2_exponent(scale)
    print(f"  {name:<28} scale = {scale:.6e}   ~ 2^-{k:0.2f}   "
          f"=> {k:0.0f} fractional bits")


def report_weight_layer(name: str, layer) -> None:
    qw = layer.quant_weight()
    report_scale(name + ".weight", qw.scale)
    ints = qw.int()
    print(f"  {'':<28} int range = [{int(ints.min())}, {int(ints.max())}]   "
          f"shape = {tuple(ints.shape)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to *_best.pt")
    p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--weight-bit-width", type=int, default=24)
    p.add_argument("--act-bit-width", type=int, default=24)
    p.add_argument("--input-bit-width", type=int, default=24)
    p.add_argument("--no-po2", action="store_true",
                   help="set if you trained WITHOUT --po2-scales")
    p.add_argument("--batchnorm", type=str, default="b")
    p.add_argument("--activation", type=str, default="relu")
    args = p.parse_args()

    qcfg = QuantConfig(
        enabled=True,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        input_bit_width=args.input_bit_width,
        po2_scales=not args.no_po2,
    )
    model = PELICANNano(
        args.n_hidden,
        quant_config=qcfg,
        batchnorm=args.batchnorm,
        activation=args.activation,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state)
    model.eval()

    print(f"\nModel: n_hidden={args.n_hidden}  "
          f"weight/act/input bits = "
          f"{args.weight_bit_width}/{args.act_bit_width}/{args.input_bit_width}\n")

    # --- Weight quantizers (the two mixing layers) ---
    print("Weight quantizers (mixing layers):")
    for i, eq in enumerate(model.net2to2.eq_layers):
        report_weight_layer(f"net2to2.eq_layers.{i}", eq.mixing)
    report_weight_layer("agg_2to0", model.agg_2to0.mixing)

    # --- Activation quantizers (QuantIdentity / QuantReLU instances) ---
    # These live at: model input, after each aggregation-ops stack, the hidden
    # activation, and the output logit. We discover them by walking the modules
    # so nothing is missed regardless of n_hidden or config.
    print("\nActivation / identity quantizers:")
    import brevitas.nn as qnn
    act_types = (qnn.QuantIdentity, qnn.QuantReLU)
    found = False
    for name, module in model.named_modules():
        if isinstance(module, act_types):
            # act_quant.scale() is the canonical way to read an act scale
            try:
                scale = module.act_quant.scale()
            except Exception:
                # not yet initialized (no calibration / forward pass run)
                print(f"  {name:<28} (scale not initialized — run a forward "
                      f"pass first)")
                found = True
                continue
            if scale is None:
                print(f"  {name:<28} (no scale — quantizer may be disabled)")
            else:
                report_scale(name, scale)
            found = True
    if not found:
        print("  (none found)")
    print()


if __name__ == "__main__":
    main()