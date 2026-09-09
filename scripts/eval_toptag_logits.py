"""
scripts/eval_toptag_logits.py

Evaluate a trained QAT nanoPELICAN checkpoint over a full toptag test file and save
the per-jet logits, in FILE ORDER, as an .npz that the figure scripts consume:

    results/npelican_logits_toptag_test.npz   ->  logits (N,) float64, is_signal (N,) int8

Why this exists: results/figures/gen_roc_overlay.py and gen_figures.py both read that
npz, but the npz was originally produced ad hoc with no script in the repo. This
reproduces it deterministically so a second model (e.g. n_hidden=4) can be added to
the ROC overlay on exactly the same jets in exactly the same order.

DATA-PATH FIDELITY
------------------
Identical to scripts/export_golden.py (which is the validated golden path):
JetDataset(shuffle=False) preserves file order, and collate_fn is reused verbatim so
the model sees jets exactly as in training/eval. collate_fn truncates to --nobj
constituents (batch_stack: p[:nobj]) and prepends the two beam spurions, so a
(N, 200, 4) toptag file and a (N, 20, 4) sample_data file both reduce to the same
20-constituent + 2-spurion input the firmware sees.

RELOAD SEMANTICS (CLAUDE.md gotcha)
-----------------------------------
Brevitas act-quantizer scale params only become live load targets after a
TRAINING-MODE forward pass, so we do: build -> model.train() -> one forward on a real
collated batch -> load_state_dict(strict=True) -> model.eval().

Run (from PELICAN-nano repo root, venv python):
  .venv/bin/python scripts/eval_toptag_logits.py \
      --checkpoint model/fpga_model_qat_w6a6i6p12_best.pt \
      --testfile ../test.h5 --out ../results/npelican_logits_toptag_test.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.disable(logging.CRITICAL)

from scripts.export_golden import build_model, make_batch  # noqa: E402


def auc_and_bgrej(logits: np.ndarray, y: np.ndarray, eff: float = 0.3):
    """AUC (rank/Mann-Whitney) and background rejection 1/eps_B at signal eff `eff`.

    Same convention as results/figures/gen_roc_overlay.py: sort by descending score,
    walk the ROC, read 1/fpr at the first point with tpr >= eff.
    """
    n_sig = int(y.sum())
    n_bkg = int(len(y) - n_sig)
    order = np.argsort(-logits, kind="mergesort")
    tpr = np.cumsum(y[order]) / n_sig
    fpr = np.cumsum(1 - y[order]) / n_bkg

    # AUC via rank sum on the ascending order (ties averaged)
    ranks = np.empty(len(logits), dtype=np.float64)
    asc = np.argsort(logits, kind="mergesort")
    sorted_scores = logits[asc]
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        ranks[asc[i:j + 1]] = avg
        r += (j - i + 1)
        i = j + 1
    auc = (ranks[y == 1].sum() - n_sig * (n_sig + 1) / 2.0) / (n_sig * n_bkg)

    k = int(np.searchsorted(tpr, eff))
    k = min(k, len(fpr) - 1)
    bgrej = float("inf") if fpr[k] == 0 else 1.0 / fpr[k]
    return float(auc), float(bgrej), float(np.mean((logits > 0) == (y == 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--testfile", default="../test.h5",
                   help="toptag test .h5 (Pmu / Nobj / is_signal), relative to repo root")
    p.add_argument("--out", required=True, help=".npz path, relative to repo root")
    p.add_argument("--chunk", type=int, default=2048, help="jets per forward pass")
    p.add_argument("--num", type=int, default=-1, help="limit jets (default: all)")
    p.add_argument("--key", default="logits",
                   help="array name for the logits inside the npz")
    p.add_argument("--float", dest="float_mode", action="store_true",
                   help="evaluate the checkpoint's FULL-PRECISION master weights with "
                        "quantization disabled (the equivariance harness's 'float' "
                        "reference). Quantizer state-dict keys are dropped on load.")
    a = p.parse_args()

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = os.path.join(repo_root, a.checkpoint)
    testfile = os.path.join(repo_root, a.testfile)
    outpath = os.path.normpath(os.path.join(repo_root, a.out))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    assert float(args.scale) == 1.0, f"args.scale={args.scale} != 1.0; refusing to guess"
    print(f"checkpoint      : {a.checkpoint}")
    print(f"  n_hidden      : {args.n_hidden}")
    print(f"  w/a/i/pmu bits: {args.weight_bit_width}/{args.act_bit_width}/"
          f"{args.input_bit_width}/{getattr(args, 'pmu_bit_width', None)}")
    print(f"  trained on    : {args.datadir}  ({args.num_epoch} epochs, seed {args.seed})")

    with h5py.File(testfile, "r") as f:
        n_total = f["Pmu"].shape[0]
        N = n_total if a.num < 0 else min(a.num, n_total)
        print(f"test file       : {a.testfile}  Pmu{f['Pmu'].shape} -> using {N} jets")

        if a.float_mode:
            # Float path: rebuild WITHOUT quantizers (mirrors model_loader.py without
            # --quant), load master weights only. No train-mode calibration forward is
            # needed — there are no Brevitas scale params to populate. strict=False is
            # required (the QAT state dict carries quantizer keys the float model lacks),
            # so gate it: nothing the float model needs may be missing, and every dropped
            # key must be quantizer internals.
            args.quant = False
            model = build_model(args)
            res = model.load_state_dict(ckpt["model_state"], strict=False)
            assert not res.missing_keys, f"float build missing keys: {res.missing_keys}"
            leftover = [k for k in res.unexpected_keys
                        if not any(t in k for t in ("quant", "scaling", "_impl"))]
            assert not leftover, f"non-quantizer keys dropped: {leftover}"
            print(f"  float mode    : master weights; dropped {len(res.unexpected_keys)} "
                  f"quantizer keys")
            model.eval()
        else:
            # --- build model + the CLAUDE.md reload dance on a real collated batch ---
            cal_n = min(256, N)
            cal = make_batch(f["Pmu"][:cal_n], f["Nobj"][:cal_n], f["is_signal"][:cal_n], args)
            model = build_model(args)
            model.train()
            with torch.no_grad():
                model(cal)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()

        # --- stream the file in chunks, file order preserved ---
        logits = np.empty(N, dtype=np.float64)
        labels = np.empty(N, dtype=np.int8)
        with torch.no_grad():
            for s in range(0, N, a.chunk):
                e = min(s + a.chunk, N)
                batch = make_batch(f["Pmu"][s:e], f["Nobj"][s:e], f["is_signal"][s:e], args)
                out = model(batch)["predict"][:, 1]  # cat([-w, w]) -> idx 1 = signal logit
                logits[s:e] = out.detach().reshape(-1).to(torch.float64).numpy()
                labels[s:e] = f["is_signal"][s:e].astype(np.int8)
                if (s // a.chunk) % 20 == 0:
                    print(f"  {e}/{N}", flush=True)

    auc, bgrej, acc = auc_and_bgrej(logits, labels.astype(int))
    print(f"\nAUC={auc:.4f}  1/eps_B@0.3={bgrej:.1f}  accuracy={acc:.4f}  n={N}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    np.savez(outpath, **{a.key: logits, "is_signal": labels})
    print(f"wrote {outpath}  ({a.key}, is_signal)")


if __name__ == "__main__":
    main()
