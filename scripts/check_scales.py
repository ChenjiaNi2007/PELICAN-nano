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


def report_scale(name: str, scale_tensor: torch.Tensor, signed: bool = None) -> None:
    scale = float(scale_tensor.detach().reshape(-1)[0])  # per-tensor: one value
    k = po2_exponent(scale)
    # Signedness decides ap_fixed vs ap_ufixed in types_generated.h and, for
    # input_quant, is now a training flag (--input-unsigned). Kept on the SAME line as
    # the scale so name-based greps (e.g. sweep_pmu_blockfp.sh) carry it along.
    sg = "" if signed is None else \
        f"   signed={signed} -> {'ap_fixed' if signed else 'ap_ufixed'}"
    print(f"  {name:<28} scale = {scale:.6e}   ~ 2^-{k:0.2f}   "
          f"=> {k:0.0f} fractional bits{sg}")


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
    p.add_argument("--input-clip-min", type=float, default=None,
                   help="set if trained with --input-clip-min (d_ij clip floor)")
    p.add_argument("--input-unsigned", action="store_true",
                   help="set if trained with --input-unsigned (unsigned d_ij grid)")
    p.add_argument("--pmu-bit-width", type=int, default=None,
                   help="set if trained with --pmu-bit-width (momentum quantizer)")
    p.add_argument("--pmu-block-fp", action="store_true",
                   help="set if trained with --pmu-block-fp (Lever 7 block floating point)")
    p.add_argument("--pmu-exp-min", type=int, default=0)
    p.add_argument("--pmu-exp-max", type=int, default=10)
    p.add_argument("--pmu-static-exp", action="store_true",
                   help="set if trained with --pmu-static-exp (SPS static per-slot exponent)")
    p.add_argument("--pmu-exp-floor-batches", type=int, default=None,
                   help="trained --pmu-exp-floor-batches K (SPS exponent floor); default: "
                        "replayed from the checkpoint's saved args (0 if absent)")
    p.add_argument("--pmu-exp-fixed", action=argparse.BooleanOptionalAction, default=None,
                   help="trained with --pmu-exp-fixed; default: replayed from checkpoint args")
    p.add_argument("--nobj", type=int, default=20,
                   help="trained --nobj (SPS slot count = nobj + 2 beams); default 20")
    p.add_argument("--add-beams", action=argparse.BooleanOptionalAction, default=True,
                   help="trained with beams (default True, as the trainer)")
    # Model-shaping flags of the hls4ml-5-class runs. Default None = replay from the
    # checkpoint's saved args (head checkpoints otherwise fail with "Unexpected key(s)
    # head.weight"; a --jet-quant-split checkpoint has extra input_quant_jet/mjet keys).
    p.add_argument("--n-out", type=int, default=None,
                   help="trained --n-out (default: replayed from checkpoint args, else 1)")
    p.add_argument("--head-hidden", type=int, default=None,
                   help="trained --head-hidden (default: replayed from checkpoint args, else 0)")
    p.add_argument("--add-jet", action=argparse.BooleanOptionalAction, default=None,
                   help="trained with --add-jet (default: replayed from checkpoint args)")
    p.add_argument("--jet-quant-split", action=argparse.BooleanOptionalAction, default=None,
                   help="trained with --jet-quant-split (default: replayed from checkpoint args)")
    for _flag in ("--jet-input-bit-width", "--mjet-input-bit-width", "--jet-pmu-bit-width"):
        p.add_argument(_flag, type=int, default=None,
                       help=f"trained {_flag} (default: replayed from checkpoint args, else None)")
    p.add_argument("--no-po2", action="store_true",
                   help="set if you trained WITHOUT --po2-scales")
    p.add_argument("--batchnorm", type=str, default="b")
    p.add_argument("--activation", type=str, default="relu")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.pmu_exp_floor_batches is None:
        # replay from the checkpoint (rebuild-trap rule); older ckpts lack it -> 0
        args.pmu_exp_floor_batches = int(getattr(ckpt.get("args"), "pmu_exp_floor_batches", 0) or 0)
    _ck = ckpt.get("args")
    for _name, _dflt in (("n_out", 1), ("head_hidden", 0), ("add_jet", False),
                         ("jet_quant_split", False), ("jet_input_bit_width", None),
                         ("mjet_input_bit_width", None), ("jet_pmu_bit_width", None)):
        if getattr(args, _name) is None:
            setattr(args, _name, getattr(_ck, _name, _dflt) if _ck is not None else _dflt)
    if args.pmu_exp_fixed is None:
        args.pmu_exp_fixed = bool(getattr(ckpt.get("args"), "pmu_exp_fixed", False)) \
            and args.pmu_static_exp

    qcfg = QuantConfig(
        enabled=True,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        input_bit_width=args.input_bit_width,
        input_unsigned=args.input_unsigned,
        input_clip_min=args.input_clip_min,
        pmu_bit_width=args.pmu_bit_width,
        pmu_block_fp=args.pmu_block_fp,
        pmu_exp_min=args.pmu_exp_min,
        pmu_exp_max=args.pmu_exp_max,
        pmu_static_exp=args.pmu_static_exp,
        pmu_exp_floor_batches=args.pmu_exp_floor_batches,
        pmu_exp_fixed=args.pmu_exp_fixed,
        pmu_n_slots=args.nobj + (2 if args.add_beams else 0) + (1 if args.add_jet else 0),
        jet_quant_split=bool(args.jet_quant_split),
        jet_input_bit_width=getattr(args, 'jet_input_bit_width', None),
        mjet_input_bit_width=getattr(args, 'mjet_input_bit_width', None),
        jet_pmu_bit_width=getattr(args, 'jet_pmu_bit_width', None),
        po2_scales=not args.no_po2,
    )
    model = PELICANNano(
        args.n_hidden,
        quant_config=qcfg,
        batchnorm=args.batchnorm,
        activation=args.activation,
        n_out=int(args.n_out),
        head_hidden=int(args.head_hidden),
    )
    state = ckpt["model_state"]
    if ("pmu_quant.log2_exp" in state) != bool(args.pmu_static_exp):
        sys.exit("check_scales: checkpoint "
                 + ("HAS" if "pmu_quant.log2_exp" in state else "has NO")
                 + " SPS static exponents (pmu_quant.log2_exp) but --pmu-static-exp was "
                 + ("not " if not args.pmu_static_exp else "") + "given -- replay the "
                 "training flags (--pmu-block-fp --pmu-static-exp --nobj ...)")
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
    from src.layers.blockfp import BlockFPQuant
    act_types = (qnn.QuantIdentity, qnn.QuantReLU)
    found = False
    # Lever 7 block-FP has no learned scale (it is stateless and derives a
    # per-particle exponent at runtime), so report its static config explicitly —
    # the module-walk below only knows about Brevitas quantizers.
    if isinstance(getattr(model, "pmu_quant", None), BlockFPQuant) and model.pmu_quant.static:
        # SPS: one learned integer exponent per particle slot (beams = slots 0,1).
        bfp = model.pmu_quant
        tbl = [int(v) for v in bfp.exponent_table()]
        fx = "FIXED (data-derived) " if bfp.fixed else ""
        print(f"  {'pmu_quant':<28} STATIC {fx}block-FP W={bfp.bit_width} (I=2, "
              f"LSB=2^{-(bfp.bit_width - 2)}) exp[i] = {tbl} "
              f"-> clip 2^(e+1) GeV per slot")
        print(f"  {'pmu_quant clip GeV':<28} {[2 ** (e + 1) for e in tbl]}")
        fz = int(getattr(ckpt.get("args"), "pmu_exp_freeze_epoch", 0) or 0)
        if fz > 0:
            print(f"  {'pmu_quant exp frozen':<28} exp frozen from epoch {fz}")
        if bfp.floor_batches > 0:
            flo = [int(v) for v in bfp.exp_floor.tolist()]
            print(f"  {'pmu_quant exp floor':<28} {flo}   (K={bfp.floor_batches}, "
                  f"batches seen={int(bfp.floor_batches_seen)})")
        found = True
    elif isinstance(getattr(model, "pmu_quant", None), BlockFPQuant):
        bfp = model.pmu_quant
        print(f"  {'pmu_quant':<28} BLOCK-FP  mantissa W={bfp.bit_width} (I=2, "
              f"LSB=2^{-(bfp.bit_width - 2)}), exp=floor(log2 E) in "
              f"[{bfp.exp_min},{bfp.exp_max}] -> no single global scale")
        found = True
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
                report_scale(name, scale, signed=bool(module.act_quant.is_signed))
            found = True
    if not found:
        print("  (none found)")
    print()


if __name__ == "__main__":
    main()