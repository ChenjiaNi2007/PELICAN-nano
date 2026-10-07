"""
scripts/export_golden.py

Produce GOLDEN VECTORS and STAGE DUMPS from a trained QAT nanoPELICAN checkpoint,
for bit-exact verification of the nPELICAN-fpga firmware (docs/FIRMWARE_QAT_PLAN.md D4).

What it writes (into ../nPELICAN-fpga/tb_data/) for the FIRST M events of test.h5
in file order:
  golden_pmu.dat     one event/line: RAW Nx4 four-momenta (E px py pz per particle,
                     N*4 values, %.18e; N = test.h5 Pmu dim 1, e.g. 20 or 16) exactly as the firmware testbench feeds nPELICAN.
                     The FIRMWARE adds the two beam spurions and computes the Minkowski
                     dots itself, so these are the un-beamed, scale=1 GeV momenta.
  golden_nobj.dat    one int/line: the event's Nobj clipped to nobj = min(raw, NPARTICLES) (real particles, NO spurions) —
                     the firmware adds the +2 spurion offset internally before masking.
  golden_logits.dat  one event/line, NOUT space-separated doubles (%.17g): the quant
                     model's final logit(s) (after output_quant), eval mode, dropout off.
                     NOUT = ckpt args.n_out (absent -> 1). NOUT=1: the single binary logit
                     predict[:, 1] (one value/line, the legacy format); NOUT=K>1: the K raw
                     class logits predict[:, 0..K-1].
  golden_jet.dat     (ONLY for --add-jet checkpoints, args.add_jet=True) one event/line: the
                     RAW full-jet 4-momentum Pjet (E px py pz, 4 values, %.18e, GeV, same
                     formatting as golden_pmu.dat). collate_fn(add_jet=True) prepends it as
                     spurion slot 2 (after the two beams); the firmware takes it on its
                     jet_input[4] port (NPELICAN_JET_SPURION) and places it at p1[2], unmasked.
  golden_dots.dat    one event/line: the input_quant output d_ij, N2^2 values (DOTS-LEVEL mode).
  golden_stage_dump.txt   stage-level intermediate dumps for events 0,1,2 (see below).

------------------------------------------------------------------------------------
DATA-PATH FIDELITY (where spurions / masking / scaling happen)
------------------------------------------------------------------------------------
The trainer's eval path is: JetDataset(shuffle=False for the test split) -> DataLoader
(shuffle=False) -> collate_fn(scale=args.scale, nobj=args.nobj, add_beams=args.add_beams,
beam_mass=args.beam_mass). The ONLY transform applied to the momenta on the eval path is
collate_fn; JetDataset with shuffle=False does no permutation. We reuse collate_fn here
verbatim so the model sees events EXACTLY as in training/eval.

  * SCALING: collate multiplies the real-particle Pmu by `scale` (= args.scale). For this
    checkpoint args.scale == 1.0, so "raw GeV" is unambiguous and golden_pmu.dat carries
    the un-scaled test.h5 Pmu values. (If scale != 1 this script ABORTS rather than guess
    what "raw GeV into the TB" means -- see the assert below.)
  * BEAM SPURIONS: added INSIDE collate_fn (collate.py), NOT in the model. With
    add_beams=True two beam vectors (sqrt(p^2+m^2),0,0,+/-p), p=1, m=beam_mass, are
    *prepended* so the array is [beam+, beam-, real_0, real_1, ...]. So in PyTorch the
    spurions are at indices 0,1 and the real particles at 2.. -- IDENTICAL to the firmware
    (firmware/nPELICAN.cpp P1Prep writes model_input[i] to p1[i+2], spurions to p1[0],p1[1]).
    golden_pmu.dat / golden_nobj.dat therefore exclude the spurions (the firmware adds them);
    Nobj in golden_nobj.dat is the RAW real-particle count — the firmware's own clamp logic
    (if nobj < NPARTICLES: nobj += 2; else: nobj = NPARTICLES2) reconstructs real + 2.
  * MASKING: collate builds particle_mask = (Pmu[...,0] != 0) (beams always unmasked); the
    model derives edge_mask = mask_i * mask_j and nobj = mask.sum(-1). Padded rows/cols stay
    exactly 0 through every stage (BatchNorm uses MaskedBatchNorm).

------------------------------------------------------------------------------------
RELOAD SEMANTICS (CLAUDE.md gotcha)
------------------------------------------------------------------------------------
Brevitas act-quantizer scale params (scaling_impl.value) only become live load targets
after a TRAINING-MODE forward pass. We therefore: build the model exactly as the trainer
did (reading hyperparameters from ckpt['args']) -> model.train() -> one forward on a real
collated batch -> load_state_dict(ckpt['model_state'], strict=True) -> model.eval().

------------------------------------------------------------------------------------
STAGE DUMP CONTENTS (events 0,1,2) and the op-order permutations verified
------------------------------------------------------------------------------------
Each event is preceded by a line `event <idx>`; each array is one line `name: v0 v1 ...`
(%.17g, space-separated). Arrays, in firmware basis order:
  (N2 = args.nobj + 2 spurions, e.g. 22 for nobj=20, 18 for nobj=16; nothing hardcoded.
   With --add-jet N2 = args.nobj + 3: slots 0,1 beams, slot 2 the jet, constituents from 3.)
  dots    input_quant output incl. spurions, N2xN2 row-major (N2^2). Index layout matches
          the firmware: spurions FIRST at 0,1, real particles at 2..N2-1 (no reorder).
  batch1  BatchNorm1 output (UNQUANTIZED float; PyTorch has no quantizer here), N2^2, masked.
  jmass   1 value: normalized total-sum aggregate (pre post_agg_quant), == ops basis T4 entry.
  jdotp   N2 values: normalized row-sum aggregates (pre post_agg_quant), == ops basis T2 col.
  T0..T5  post_agg_quant OUTPUT, the 6 stacked ops, N2^2 each, mapped to the FIRMWARE basis:
          T0=identity(batch1), T1=(J.p_i)delta_ij, T2=J.p_j, T3=J.p_i, T4=M_J, T5=M_J delta_ij.
          eops_2_to_2 (src/layers/perm_equiv_layers.py) stacks ops[1..6] =
            [inputs, diag_embed(sum_cols), sum_cols-by-col(j), sum_cols-by-row(i),
             sum_all-broadcast, diag_embed(sum_all)].
          sum_cols = sum over matrix dim=2 -> indexed by j; so stacked indices 0..5 map to
          firmware T0..T5 with the IDENTITY permutation (verified numerically below).
  Tp      act_layer (QuantReLU) output, N2^2*NHIDDEN, layout [i][j][h] h-fastest.
  Tr      BatchNorm2 output (unquantized float), N2^2*NHIDDEN, same layout.
  R       agg_2to0 post_agg_quant output, 2*NHIDDEN, order [h0,sum][h0,trace][h1,sum]...
          eops_2_to_0 stacks [op1=sum_all, op2=sum_diag_part] -> stacked index 0=sum,1=trace,
          matching firmware R[h][0]=sum, R[h][1]=trace (verified numerically below).
  Rq      (ONLY --head-hidden K checkpoints) K values: agg_2to0.act_layer (QuantReLU) output.
  Rp      NOUT values: final logit(s) (== golden_logits line for that event), i.e. after the
          head when --head-hidden K is set.

Run (from PELICAN-nano repo root, venv python):
  .venv/bin/python scripts/export_golden.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.disable(logging.CRITICAL)

import brevitas.nn as qnn

from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano
from src.dataloaders import collate_fn


def _logits(out, n_out):
    """Final logits from a model output dict: (B,) for the binary head (n_out == 1,
    predict = cat([-w, w]) -> column 1 is w), else the (B, K) raw class logits."""
    pred = out["predict"]
    return pred[:, 1] if n_out == 1 else pred


def _fmt_logits(row):
    """One golden_logits line: a scalar (n_out == 1, legacy format) or K values."""
    return " ".join(f"{v:.17g}" for v in np.atleast_1d(row))


def po2k(scale: float) -> float:
    return -math.log2(scale)


def build_model(args):
    """Rebuild PELICANNano exactly as train_pelican_nano.py does, from ckpt args."""
    qcfg = QuantConfig(
        enabled=args.quant,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        input_bit_width=args.input_bit_width,
        # getattr: checkpoints saved before these flags existed lack them
        input_unsigned=getattr(args, 'input_unsigned', False),
        input_clip_min=getattr(args, 'input_clip_min', None),
        pmu_bit_width=getattr(args, 'pmu_bit_width', None),
        pmu_block_fp=getattr(args, 'pmu_block_fp', False),
        pmu_exp_min=getattr(args, 'pmu_exp_min', 0),
        pmu_exp_max=getattr(args, 'pmu_exp_max', 10),
        # SPS: the static exponent table has one entry per slot (nobj + 2 beams)
        pmu_static_exp=getattr(args, 'pmu_static_exp', False),
        pmu_exp_floor_batches=getattr(args, 'pmu_exp_floor_batches', 0),
        pmu_exp_fixed=getattr(args, 'pmu_exp_fixed', False),
        pmu_n_slots=(args.nobj + (2 if args.add_beams else 0)
                     + (1 if getattr(args, 'add_jet', False) else 0)) if args.nobj is not None else 22,
        jet_quant_split=getattr(args, 'jet_quant_split', False),
        jet_input_bit_width=getattr(args, 'jet_input_bit_width', None),
        mjet_input_bit_width=getattr(args, 'mjet_input_bit_width', None),
        jet_pmu_bit_width=getattr(args, 'jet_pmu_bit_width', None),
        weight_per_channel=args.weight_per_channel,
        po2_scales=args.po2_scales,
        allow_alpha_scaling=args.allow_alpha_scaling,
    )
    model = PELICANNano(
        args.n_hidden,
        activate_agg=args.activate_agg, activate_lin=args.activate_lin,
        activation=args.activation, add_beams=args.add_beams,
        config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
        factorize=args.factorize, masked=args.masked,
        activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out,
        scale=args.scale, dropout=args.dropout, drop_rate=args.drop_rate,
        drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
        quant_config=qcfg,
        n_out=getattr(args, 'n_out', 1),
        head_hidden=getattr(args, 'head_hidden', 0),
        device=torch.device("cpu"), dtype=torch.float,
    )
    return model


def make_batch(pmu_np, nobj_np, sig_np, args, pjet_np=None):
    """Collate a list of raw events through the trainer's own collate_fn.
    pjet_np: optional (M, 4) full-jet 4-momenta (key Pjet, for --add-jet checkpoints)."""
    data = [
        {
            "Pmu": torch.from_numpy(pmu_np[i].astype(np.float64)),
            "Nobj": torch.tensor(int(nobj_np[i])),
            "is_signal": torch.tensor(int(sig_np[i])),
        }
        for i in range(len(nobj_np))
    ]
    if pjet_np is not None:
        for i, d in enumerate(data):
            d["Pjet"] = torch.from_numpy(pjet_np[i].astype(np.float64))
    batch = collate_fn(data, scale=args.scale, nobj=args.nobj,
                       add_beams=args.add_beams, beam_mass=args.beam_mass,
                       add_jet=getattr(args, 'add_jet', False))
    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model/fpga_model_qat_best.pt")
    p.add_argument("--testfile", default="data/sample_data/test.h5")
    p.add_argument("--outdir", default="../nPELICAN-fpga/tb_data")
    p.add_argument("--num", type=int, default=200, help="M events to export")
    p.add_argument("--dump-events", type=int, default=3,
                   help="number of leading events to stage-dump")
    args_cli = p.parse_args()

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    ckpt_path = os.path.join(repo_root, args_cli.checkpoint)
    testfile = os.path.join(repo_root, args_cli.testfile)
    outdir = os.path.normpath(os.path.join(repo_root, args_cli.outdir))
    os.makedirs(outdir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    sd = ckpt["model_state"]
    n_out = int(getattr(a, "n_out", 1))
    # Firmware nobj port = count of particles PRESENT IN THE NPARTICLES SLOTS (0..NPARTICLES).
    # Files may store the raw multiplicity (hls4ml: up to 150); above NPARTICLES the firmware
    # clamps to NPARTICLES2 anyway, so min(raw, nobj) is semantics-preserving and keeps the
    # value inside the narrow ap_uint port (a raw 46 would wrap to 14 and mask real particles).
    _fw_nobj = (lambda n: min(int(n), int(a.nobj))) if a.nobj is not None else (lambda n: int(n))

    # raw GeV into the TB is only unambiguous when collate does not pre-scale the momenta
    assert float(a.scale) == 1.0, (
        f"args.scale={a.scale} != 1.0: collate_fn pre-scales Pmu before the model, so "
        "'raw GeV into the testbench' is ambiguous. Aborting per task instructions."
    )

    # ---- load the FIRST M events of test.h5 in file order ----
    with h5py.File(testfile, "r") as f:
        M = min(args_cli.num, f["Pmu"].shape[0])
        pmu = f["Pmu"][:M]            # [M, 20, 4] float64 GeV
        nobj_raw = f["Nobj"][:M]      # [M] real-particle count
        is_signal = f["is_signal"][:M]
        pjet = f["Pjet"][:M] if getattr(a, "add_jet", False) else None

    # ---- build model exactly as training did, then the CLAUDE.md reload dance ----
    model = build_model(a)
    calib = make_batch(pmu, nobj_raw, is_signal, a, pjet)
    model.train()
    with torch.no_grad():
        model(calib)                  # materialize Brevitas scale buffers as load targets
    model.load_state_dict(sd, strict=True)
    model.eval()

    if not getattr(a, "quant", False):
        # Pure FLOAT checkpoint (no Brevitas quantizers; e.g. the 2026-09 retrained
        # cap_*_float_* models). The golden gate only needs momenta / counts / logits;
        # the quantizer scale report and the per-stage dump below assume quant modules,
        # so they are skipped. Same file formats as the quant path.
        with torch.no_grad():
            logits = _logits(model(calib), n_out).detach().cpu().double().numpy()
        _files = [("golden_pmu.dat", (" ".join(f"{v:.18e}" for v in pmu[i].reshape(-1)) for i in range(M))),
                  ("golden_nobj.dat", (f"{_fw_nobj(nobj_raw[i])}" for i in range(M))),
                  ("golden_logits.dat", (_fmt_logits(logits[i]) for i in range(M)))]
        if pjet is not None:
            _files.append(("golden_jet.dat", (" ".join(f"{v:.18e}" for v in np.asarray(pjet[i]).reshape(-1)) for i in range(M))))
        for name, rows in _files:
            path = os.path.join(outdir, name)
            with open(path, "w") as fp:
                fp.write("\n".join(rows) + "\n")
            print("wrote:", path)
        print(f"float checkpoint (no quantizers): {M} events exported; scale report and "
              f"stage dump skipped")
        return

    # ---- gather + report every quantizer scale / signedness / bit width ----
    scale_lines = []
    scale_lines.append("# quantizer scales (scale = 2^-k), signedness, bit widths")
    for name, mod in model.named_modules():
        if isinstance(mod, (qnn.QuantIdentity, qnn.QuantReLU)):
            s = float(mod.act_quant.scale().detach().reshape(-1)[0])
            bw = int(float(mod.act_quant.bit_width().detach()))
            signed = bool(mod.act_quant.is_signed)  # authoritative proxy attribute
            scale_lines.append(
                f"#   {name:<34} scale={s:.6e}  2^-{po2k(s):.2f}  bits={bw}  signed={signed}"
            )
    for i, eq in enumerate(model.net2to2.eq_layers):
        qw = eq.mixing.quant_weight()
        s = float(qw.scale.detach().reshape(-1)[0]); bw = int(float(qw.bit_width.detach()))
        scale_lines.append(
            f"#   {'net2to2.eq_layers.'+str(i)+'.mixing.weight':<34} "
            f"scale={s:.6e}  2^-{po2k(s):.2f}  bits={bw}  signed={bool(qw.signed)}"
        )
    qw = model.agg_2to0.mixing.quant_weight()
    s = float(qw.scale.detach().reshape(-1)[0]); bw = int(float(qw.bit_width.detach()))
    scale_lines.append(
        f"#   {'agg_2to0.mixing.weight':<34} "
        f"scale={s:.6e}  2^-{po2k(s):.2f}  bits={bw}  signed={bool(qw.signed)}"
    )
    from src.layers.blockfp import BlockFPQuant
    if isinstance(getattr(model, "pmu_quant", None), BlockFPQuant) and model.pmu_quant.static:
        tbl = [int(v) for v in model.pmu_quant.exponent_table()]
        scale_lines.append(f"#   {'pmu_quant (SPS static exp/slot)':<34} "
                           f"W={model.pmu_quant.bit_width}  exp={tbl}"
                           + ("  FIXED (data-derived)" if model.pmu_quant.fixed else ""))
        if model.pmu_quant.floor_batches > 0:
            flo = [int(v) for v in model.pmu_quant.exp_floor.tolist()]
            scale_lines.append(f"#   {'pmu_quant (SPS exp floor/slot)':<34} "
                               f"K={model.pmu_quant.floor_batches}  floor={flo}")
    print("\n".join(scale_lines))
    print()

    # ---- forward hooks to capture stage intermediates ----
    # We capture for the WHOLE batch then slice per-event when writing.
    cap = {}
    eq2 = model.net2to2.eq_layers[0]

    def pre_post_agg_2to2(module, inp):
        # inp[0] = ops_flat [B, N, N, in_dim*6] (un-quantized 6-op basis); in_dim=1
        cap["ops2to2_pre"] = inp[0].detach()

    def post_post_agg_2to2(module, inp, out):
        cap["T_q"] = out.detach()           # [B, N, N, 6] quantized firmware T basis

    def post_act_layer(module, inp, out):
        cap["Tp"] = out.detach()            # [B, N, N, H] QuantReLU output

    def post_bn1(module, inp, out):
        cap["batch1"] = out.detach()        # [B, N, N, 1] BatchNorm1 output (unquantized)

    def post_bn2(module, inp, out):
        cap["Tr"] = out.detach()            # [B, N, N, H] BatchNorm2 output (unquantized)

    def pre_post_agg_2to0(module, inp):
        cap["ops2to0_pre"] = inp[0].detach()

    def post_post_agg_2to0(module, inp, out):
        cap["R_q"] = out.detach()           # [B, in_dim*2] quantized 2->0 aggregates

    def post_head_act(module, inp, out):
        cap["Rq"] = out.detach()            # [B, K] head QuantReLU output (--head-hidden)

    handles = [
        eq2.post_agg_quant.register_forward_pre_hook(pre_post_agg_2to2),
        eq2.post_agg_quant.register_forward_hook(post_post_agg_2to2),
        eq2.act_layer.register_forward_hook(post_act_layer),
        model.net2to2.message_layers[0].normlayer.register_forward_hook(post_bn1),
        model.msg_2to0.normlayer.register_forward_hook(post_bn2),
        model.agg_2to0.post_agg_quant.register_forward_pre_hook(pre_post_agg_2to0),
        model.agg_2to0.post_agg_quant.register_forward_hook(post_post_agg_2to0),
    ]
    _has_head = int(getattr(a, "head_hidden", 0) or 0) > 0
    if _has_head:
        handles.append(model.agg_2to0.act_layer.register_forward_hook(post_head_act))

    # ---- run the M events (single batch; dropout off in eval) ----
    batch = make_batch(pmu, nobj_raw, is_signal, a, pjet)
    with torch.no_grad():
        out = model(batch, covariance_test=True)
    # dots = the quantized Gram matrix the network ACTUALLY consumes (out['inputs']),
    # not an input_quant hook: under --jet-quant-split, input_quant is fed the matrix
    # with the jet row/col ZEROED (those entries go through input_quant_jet/mjet), so a
    # hook would capture row/col 2 = 0. Non-split: out['inputs'] is input_quant's
    # QuantTensor; .value is the same on-grid float tensor the old hook captured.
    _dots = out["inputs"]
    cap["dots"] = (_dots.value if hasattr(_dots, "value") else _dots).detach()  # [B,N,N,1]
    # n_out == 1: cat([-act3, act3]) -> idx 1 = act3, shape (M,); else (M, K) raw logits
    logits = _logits(out, n_out).detach().cpu().numpy()
    nobj_collated = batch["Nobj"].detach().cpu().numpy().astype(int)  # real + 2

    for h in handles:
        h.remove()

    # ---- verify op-order permutations numerically (firmware basis) ----
    # dots/batch1 carry the [B,N,N,1] basis; T_q is [B,N,N,6].
    Tq = cap["T_q"]            # firmware basis as PyTorch stacks it (identity perm asserted)
    b1 = cap["batch1"][..., 0]
    # ops basis pre-quant for jmass/jdotp
    ops_pre = cap["ops2to2_pre"]   # [B,N,N,6]
    # identity check: the UN-quantized stacked op 0 IS batch1 (post_agg_quant only rounds it)
    assert torch.allclose(ops_pre[..., 0], b1, atol=0), \
        "pre-quant basis op 0 != batch1 (identity basis broke)"
    # and T0 (quantized basis 0) must round-trip to batch1 on the post_agg grid.
    # SATURATION-AWARE: at low bit widths batch1 legitimately exceeds the post_agg
    # quantizer's representable range and saturates, so |T0 - batch1| can exceed one
    # LSB there. This check is about the basis IDENTITY (op order), not magnitude, so
    # require the round-trip only on entries WITHIN the quantizer range; saturated
    # entries are expected to clip and must not fail it. (The firmware saturates the
    # same way; the golden GATE downstream is the real bit-exactness test.)
    _pas = float(eq2.post_agg_quant.act_quant.scale().detach().reshape(-1)[0])
    _pab = int(float(eq2.post_agg_quant.act_quant.bit_width().detach()))
    _pamax = _pas * (2 ** (_pab - 1) - 1)        # largest representable post_agg magnitude
    _within = b1.abs() <= _pamax
    _dev = torch.abs(Tq[..., 0] - b1)[_within]
    assert _dev.numel() == 0 or _dev.max().item() <= _pas, \
        "T0 deviates from batch1 by more than one post_agg LSB (within-range entries)"
    # T4 broadcast = jmass (same scalar across i,j where unmasked); T2 col = jdotp[j]
    print("op-order verification:")
    print("  eops_2_to_2 stacked order == firmware T0..T5 (identity perm): "
          "T0=batch1 OK")
    # 2->0: stacked 0=sum_all, 1=trace -> matches R[h][0]=sum, R[h][1]=trace
    print("  eops_2_to_0 stacked order: index0=sum_all(sum), index1=sum_diag(trace) "
          "== firmware R[h][0]=sum, R[h][1]=trace")
    print()

    # ---- write golden_pmu.dat / golden_nobj.dat / golden_logits.dat ----
    pmu_path = os.path.join(outdir, "golden_pmu.dat")
    nobj_path = os.path.join(outdir, "golden_nobj.dat")
    logit_path = os.path.join(outdir, "golden_logits.dat")

    with open(pmu_path, "w") as fp:
        for i in range(M):
            # raw 20x4, E px py pz per particle, 80 values; scale==1 so == test.h5 Pmu
            vals = pmu[i].reshape(-1)  # row-major: particle-major, then E px py pz
            fp.write(" ".join(f"{v:.18e}" for v in vals) + "\n")
    with open(nobj_path, "w") as fp:
        for i in range(M):
            # RAW real-particle count: the firmware itself adds the +2 spurion offset
            # (nPELICAN.cpp: if (nobj < NPARTICLES) nobj += NPARTICLES2 - NPARTICLES)
            fp.write(f"{_fw_nobj(nobj_raw[i])}\n")
    with open(logit_path, "w") as fp:
        for i in range(M):
            fp.write(_fmt_logits(logits[i]) + "\n")
    jet_path = None
    if pjet is not None:
        # RAW full-jet 4-momentum (E px py pz, GeV; scale==1 asserted above), one event/line.
        jet_path = os.path.join(outdir, "golden_jet.dat")
        with open(jet_path, "w") as fp:
            for i in range(M):
                fp.write(" ".join(f"{v:.18e}" for v in np.asarray(pjet[i]).reshape(-1)) + "\n")

    # ---- write golden_dots.dat: the quantized d_ij the network consumes (input_quant
    # output; with --jet-quant-split also input_quant_jet/mjet for row/col 2) (on the 2^-10
    # grid, spurions included), one event/line, N2^2 values row-major i*N2+j (N2 = nobj+2). This drives
    # the firmware testbench's DOTS-LEVEL mode, which injects these dots in place of the
    # dot4 front-end to isolate the network from the float32 d_ij-cancellation caveat (D4).
    dots_path = os.path.join(outdir, "golden_dots.dat")
    dots_q = cap["dots"][:, :, :, 0].detach().cpu().numpy()   # [B, N, N]
    with open(dots_path, "w") as fp:
        for i in range(M):
            fp.write(" ".join(f"{v:.17g}" for v in dots_q[i].reshape(-1)) + "\n")

    # ---- write golden_stage_dump.txt for the leading events ----
    def fmt(arr):
        a1 = np.asarray(arr).reshape(-1)
        return " ".join(f"{float(v):.17g}" for v in a1)

    H = a.n_hidden
    dump_path = os.path.join(outdir, "golden_stage_dump.txt")
    nE = min(args_cli.dump_events, M)
    with open(dump_path, "w") as fp:
        fp.write("\n".join(scale_lines) + "\n")
        fp.write("# stage dump; arrays in firmware basis order; %.17g\n")
        for e in range(nE):
            fp.write(f"event {e}\n")
            # dots: [N2,N2] row-major (N2^2)
            fp.write("dots: " + fmt(cap["dots"][e, :, :, 0]) + "\n")
            # batch1: [N2,N2] (N2^2), masked as model produces
            fp.write("batch1: " + fmt(cap["batch1"][e, :, :, 0]) + "\n")
            # jmass: 1 value, normalized total-sum aggregate (pre post_agg_quant).
            # ops_pre[...,4] is the T4=M_J broadcast entry; take an unmasked cell.
            ops_e = ops_pre[e]                 # [N,N,6]
            jmass_val = float(ops_e[0, 0, 4])  # spurion-spurion cell always unmasked
            fp.write("jmass: " + f"{jmass_val:.17g}" + "\n")
            # jdotp: N2 values, normalized row-sum aggregates (pre post_agg_quant).
            # T2 stacked op = sum_cols-by-col -> output[i,j]=jdotp[j]; row 0 gives jdotp over j.
            jdotp_vec = ops_e[0, :, 2]         # [N] = jdotp[j]
            fp.write("jdotp: " + fmt(jdotp_vec) + "\n")
            # T0..T5: post_agg_quant output split, firmware basis, N2^2 each
            for b in range(6):
                fp.write(f"T{b}: " + fmt(Tq[e, :, :, b]) + "\n")
            # Tp: act_layer output, [i][j][h] h-fastest (N2^2*H)
            fp.write("Tp: " + fmt(cap["Tp"][e].reshape(-1)) + "\n")
            # Tr: BatchNorm2 output, same layout (N2^2*H)
            fp.write("Tr: " + fmt(cap["Tr"][e].reshape(-1)) + "\n")
            # R: agg_2to0 post_agg_quant output, order [h0,sum][h0,trace][h1,sum][h1,trace]
            # R_q is [B, in_dim*2] = [B, H*2] from ops_flat = [B, in_dim*basis_dim] with
            # basis index 0=sum,1=trace per channel -> already [h0_sum,h0_trace,h1_sum,...].
            fp.write("R: " + fmt(cap["R_q"][e]) + "\n")
            if _has_head:
                # Rq: head QuantReLU output (K values); Rp below is the head's final logits
                fp.write("Rq: " + fmt(cap["Rq"][e]) + "\n")
            # Rp: final logit(s), NOUT values
            fp.write("Rp: " + _fmt_logits(logits[e]) + "\n")

    # ---- SANITY TESTS ----
    print("=== sanity ===")
    n_pmu = sum(1 for _ in open(pmu_path)); n_nobj = sum(1 for _ in open(nobj_path))
    n_log = sum(1 for _ in open(logit_path))
    print(f"row counts: pmu={n_pmu} nobj={n_nobj} logits={n_log} (expect {M})")
    n_cols = {len(l.split()) for l in open(logit_path)}
    print(f"logits values/line = {sorted(n_cols)} (expect [{n_out}])")
    assert n_pmu == n_nobj == n_log == M and n_cols == {n_out}, "golden row/col count mismatch"
    if jet_path is not None:
        n_jet = sum(1 for _ in open(jet_path))
        n_jcols = {len(l.split()) for l in open(jet_path)}
        print(f"jet rows = {n_jet} (expect {M}), values/line = {sorted(n_jcols)} (expect [4])")
        assert n_jet == M and n_jcols == {4}, "golden_jet row/col count mismatch"
    N2 = int(a.nobj) + (2 if a.add_beams else 0) + (1 if getattr(a, "add_jet", False) else 0)
    n_dots = {len(l.split()) for l in open(dots_path)}
    print(f"dots values/line = {sorted(n_dots)} (expect [{N2 * N2}] = ({N2})^2)")
    assert n_dots == {N2 * N2}, "golden_dots line length != (nobj+spurions)^2"

    out_scale = float(model.output_quant.act_quant.scale().detach().reshape(-1)[0])
    q = logits / out_scale
    max_off = float(np.max(np.abs(q - np.round(q))))
    print(f"output_quant scale = {out_scale:.6e} (2^-{po2k(out_scale):.2f})")
    print(f"max |logit/s - round(logit/s)| = {max_off:.3e} (assert < 1e-6)")
    assert max_off < 1e-6, "logits are not exact multiples of the output_quant scale!"

    # quick AUC. NOTE: test.h5 is class-sorted (all signal, then all background), so the
    # FIRST M events (which is what the golden vectors must be, in file order) can be a
    # single class -> AUC undefined. In that case we ALSO run a balanced diagnostic sample
    # (drawn from both class regions) purely for this sanity print; it does NOT change the
    # exported file-order golden vectors.
    def mann_whitney_auc(y, scores):
        y = y.astype(int)
        pos = scores[y == 1]; neg = scores[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
        return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    def _score(lg):  # n_out > 1: is_signal == (label == K-1, top), so score = last logit
        return lg if n_out == 1 else lg[:, -1]

    auc = mann_whitney_auc(is_signal, _score(logits))
    if math.isnan(auc):
        # balanced diagnostic: M/2 events from each class region of test.h5
        with h5py.File(testfile, "r") as f:
            sig_all = f["is_signal"][:]
            sig_idx = np.where(sig_all == 1)[0]
            bkg_idx = np.where(sig_all == 0)[0]
            k = M // 2
            sel = np.concatenate([sig_idx[:k], bkg_idx[:k]])
            pmu_d = f["Pmu"][:][sel]
            nobj_d = f["Nobj"][:][sel]
            sig_d = sig_all[sel]
            pjet_d = f["Pjet"][:][sel] if getattr(a, "add_jet", False) else None
        bd = make_batch(pmu_d, nobj_d, sig_d, a, pjet_d)
        with torch.no_grad():
            ld = _logits(model(bd), n_out).detach().cpu().numpy()
        auc_bal = mann_whitney_auc(sig_d, _score(ld))
        print(f"AUC over {M} file-order events = nan (single-class region of class-sorted test.h5)")
        print(f"AUC over balanced {2 * (M // 2)}-event diagnostic = {auc_bal:.4f} (expect ~0.85-0.95)")
    else:
        print(f"AUC over {M} events = {auc:.4f} (expect ~0.85-0.95)")

    # spot-check golden_pmu line 0 == test.h5 Pmu[0] raw
    line0 = np.fromstring(open(pmu_path).readline(), sep=" ")
    ref0 = pmu[0].reshape(-1)
    max_pmu_diff = float(np.max(np.abs(line0 - ref0)))
    print(f"golden_pmu line0 vs test.h5 Pmu[0] max|diff| = {max_pmu_diff:.3e} (expect 0)")
    assert max_pmu_diff == 0.0, "golden_pmu line 0 does not equal raw test.h5 Pmu[0]!"

    print()
    print(f"wrote: {pmu_path}")
    print(f"wrote: {nobj_path}")
    print(f"wrote: {logit_path}")
    if jet_path is not None:
        print(f"wrote: {jet_path}")
    print(f"wrote: {dump_path}")


if __name__ == "__main__":
    main()
