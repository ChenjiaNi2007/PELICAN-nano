"""
scripts/eval_hls4ml5.py

Evaluate a trained 5-class nanoPELICAN checkpoint (--n-out 5 --target label) on the
hls4ml LHC jet test split and produce numbers comparable to arXiv:2503.03103
(MLP-Mixer) Table 2: per-class one-vs-rest AUC, TPR@FPR=10%, TPR@FPR=1%, plus accuracy.
See docs/HLS4ML_5CLASS.md.

Outputs (default --out-dir ../results/hls4ml5, relative to the repo root):
    <ckpt-stem>_test_logits.npz   logits (n,K) float32, label (n,) int8, nobj (n,) int16
                                  (nobj = constituents seen after truncation to args.nobj)
    summary.csv                   one appended row per evaluation

DATA-PATH FIDELITY
------------------
Same as train_pelican_nano.py's test loader: the h5 is read whole into torch tensors,
JetDataset(shuffle=False) keeps FILE ORDER, DataLoader(shuffle=False), and the trainer's
collate_fn (truncate to args.nobj, prepend beams) is used verbatim.

RELOAD SEMANTICS (CLAUDE.md gotcha)
-----------------------------------
QAT checkpoints: build -> model.train() -> one forward on a real collated batch under
no_grad -> load_state_dict(strict=True) -> mark_scales_initialized -> model.eval().
Float checkpoints (args.quant False): direct strict load.

Run (repo root):
  .venv/bin/python -u scripts/eval_hls4ml5.py --checkpoint model/h5n16_h2_float_s1_best.pt \
      --print-paper --tag "h2 float s1"
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.disable(logging.CRITICAL)

from scripts.export_golden import build_model  # noqa: E402
from src.dataloaders import JetDataset, collate_fn  # noqa: E402
from src.models.metrics_multiclass import (  # noqa: E402
    CLASSES, class_names, compute_multiclass_metrics)
from src.trainer.utils import mark_scales_initialized  # noqa: E402

# arXiv:2503.03103 Table 2, full-precision MLP-Mixer ("MLPM-fp"), 16 features; per class
# in g/q/w/z/t order, all in percent. JEDI-net (N=100) reports AUC only.
PAPER = {
    16:  {"AUC": (94.08, 92.15, 96.06, 95.34, 96.14),
          "TPR10": (82.2, 77.8, 89.5, 86.4, 90.8),
          "TPR1": (40.8, 25.9, 51.5, 66.4, 60.3), "acc_q": 77.5},
    32:  {"AUC": (95.08, 93.09, 97.49, 97.12, 96.91),
          "TPR10": (85.3, 80.3, 93.1, 91.2, 92.5),
          "TPR1": (44.9, 28.1, 70.2, 77.6, 64.2), "acc_q": 80.7},
    64:  {"AUC": (95.53, 93.43, 97.83, 97.50, 97.13),
          "TPR10": (86.7, 81.1, 93.8, 92.2, 93.0),
          "TPR1": (46.9, 28.2, 75.4, 80.9, 64.7), "acc_q": 81.6},
    128: {"AUC": (95.48, 93.32, 97.78, 97.43, 97.01),
          "TPR10": (86.6, 81.0, 93.7, 92.0, 92.7),
          "TPR1": (46.6, 28.0, 74.8, 80.5, 63.6), "acc_q": 81.3},
}
PAPER_PARAMS = {16: 2465, 32: 3265, 64: 6401, 128: 18817}
JEDI_AUC = (95.29, 93.01, 97.39, 96.79, 96.83)


def count_params(model) -> int:
    """Learnable model parameters, excluding Brevitas quantizer internals (scales)."""
    return int(sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad
                   and not any(t in n for t in ("quant", "scaling", "_impl"))))


def load_h5(path, max_jets):
    """Read the test file into torch tensors exactly like initialize_datasets."""
    with h5py.File(path, "r") as f:
        n = f["Nobj"].shape[0] if max_jets < 0 else min(max_jets, f["Nobj"].shape[0])
        return {k: torch.from_numpy(v[:n]) for k, v in f.items()}


def fmt_row(name, acc, auc, tpr10, tpr1, extra=""):
    def j(xs):
        return " / ".join("—" if x is None else f"{x:.2f}" for x in xs)
    a = "—" if acc is None else f"{acc:.2f}"
    return f"| {name} | {a} | {j(auc)} | {j(tpr10)} | {j(tpr1)} | {extra} |"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="model/<prefix>_best.pt (repo-relative)")
    p.add_argument("--testfile", default=None,
                   help="default: <ckpt args.datadir>/test.h5")
    p.add_argument("--out-dir", default="../results/hls4ml5",
                   help="relative to the repo root")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-jets", type=int, default=-1, help="debug: limit test jets")
    p.add_argument("--print-paper", action="store_true",
                   help="also print the paper's MLPM-fp row for this N and JEDI-net")
    p.add_argument("--tag", default="", help="free text for the summary row")
    a = p.parse_args()

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt_path = os.path.join(repo_root, a.checkpoint)
    out_dir = os.path.normpath(os.path.join(repo_root, a.out_dir))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    n_out = int(getattr(args, "n_out", 1))
    if n_out == 1:
        sys.exit(f"ERROR: {a.checkpoint} is a binary checkpoint (n_out=1). "
                 "Use scripts/eval_toptag_logits.py for top-tag models.")
    target = args.target
    testfile = a.testfile or os.path.join(args.datadir, "test.h5")
    testfile = os.path.join(repo_root, testfile)
    names = class_names(n_out)

    print(f"checkpoint : {a.checkpoint}  (epoch {ckpt.get('epoch')})")
    print(f"  n_hidden={args.n_hidden} n_out={n_out} nobj={args.nobj} "
          f"nobj_avg={args.nobj_avg} quant={bool(args.quant)} target={target}")
    print(f"  trained on {args.datadir}, seed {args.seed}")
    print(f"test file  : {os.path.relpath(testfile, repo_root)}", flush=True)

    data = load_h5(testfile, a.max_jets)
    n = int(data["Nobj"].shape[0])
    print(f"  {n} jets, keys {sorted(data)}", flush=True)

    dataset = JetDataset(data, shuffle=False)
    collate = lambda d: collate_fn(d, scale=args.scale, nobj=args.nobj,  # noqa: E731
                                   add_beams=args.add_beams, beam_mass=args.beam_mass,
                                   add_jet=getattr(args, "add_jet", False))
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    model = build_model(args)
    if args.quant:
        cal = collate([dataset[i] for i in range(min(256, n))])
        model.train()
        with torch.no_grad():
            model(cal)
        model.load_state_dict(ckpt["model_state"], strict=True)
        mark_scales_initialized(model, loaded_keys=ckpt["model_state"].keys())
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    n_params = count_params(model)
    print(f"  {n_params} learnable params (excl. quantizer scales)", flush=True)

    logits = np.empty((n, n_out), dtype=np.float32)
    labels = np.empty(n, dtype=np.int8)
    nobjs = np.empty(n, dtype=np.int16)
    s = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            out = model(batch)["predict"]
            e = s + out.shape[0]
            logits[s:e] = out.detach().to(torch.float32).numpy()
            labels[s:e] = batch[target].reshape(-1).numpy().astype(np.int8)
            # constituents the model actually saw (after truncation to args.nobj): the
            # file's Nobj is the UNtruncated multiplicity, so count non-zero-E rows instead
            n_lead = (2 if args.add_beams else 0) + (1 if getattr(args, "add_jet", False) else 0)
            pm = batch["Pmu"][:, n_lead:, 0]
            nobjs[s:e] = (pm != 0).sum(dim=1).numpy().astype(np.int16)
            s = e
            if b % 20 == 0:
                print(f"  {e}/{n}", flush=True)
    assert s == n, (s, n)

    m = compute_multiclass_metrics(logits, labels, names=names, fpr_points=(0.01, 0.1))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    npz = os.path.join(out_dir, f"{stem}_test_logits.npz")
    np.savez(npz, logits=logits, label=labels, nobj=nobjs)
    print(f"\nwrote {npz}")

    header = (["tag", "checkpoint", "epoch", "N", "n_hidden", "n_params", "quant", "n_jets",
               "accuracy", "AUC_macro"] + [f"AUC_{c}" for c in names]
              + [f"TPR@FPR0.01_{c}" for c in names] + [f"TPR@FPR0.1_{c}" for c in names]
              + ["date"])
    row = ([a.tag, a.checkpoint, ckpt.get("epoch"), args.nobj, args.n_hidden, n_params,
            bool(args.quant), n, f"{m['accuracy']:.6f}", f"{m['AUC']:.6f}"]
           + [f"{m[f'AUC_{c}']:.6f}" for c in names]
           + [f"{m[f'TPR@FPR0.01_{c}']:.6f}" for c in names]
           + [f"{m[f'TPR@FPR0.1_{c}']:.6f}" for c in names]
           + [datetime.datetime.now().isoformat(timespec="seconds")])
    summ = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summ)
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    print(f"appended to {summ}")

    pct = lambda key: [100 * m[f"{key}_{c}"] for c in names]  # noqa: E731
    cls = "/".join(names)
    print(f"\nN={args.nobj}, {n} test jets. Per-class values in {cls} order, percent.\n")
    print(f"| model | acc | AUC ({cls}) | TPR@FPR10% | TPR@FPR1% | params |")
    print("|---|---|---|---|---|---|")
    label = f"nanoPELICAN h={args.n_hidden} {'QAT' if args.quant else 'float'}"
    if a.tag:
        label += f" ({a.tag})"
    print(fmt_row(label, 100 * m["accuracy"], pct("AUC"), pct("TPR@FPR0.1"),
                  pct("TPR@FPR0.01"), extra=str(n_params)))
    if a.print_paper:
        if tuple(names) != CLASSES:
            print("(paper rows skipped: class names are not g/q/w/z/t)")
        else:
            pr = PAPER.get(int(args.nobj))
            if pr is None:
                print(f"(no MLPM-fp row for N={args.nobj}; paper has N in {sorted(PAPER)})")
            else:
                print(fmt_row(f"MLPM-fp N={args.nobj} [2503.03103 T2]", None, pr["AUC"],
                              pr["TPR10"], pr["TPR1"],
                              extra=f"{PAPER_PARAMS[int(args.nobj)]} (quant acc "
                                    f"{pr['acc_q']}%, T4)"))
            print(fmt_row("JEDI-net N=100 [T2]", None, JEDI_AUC, [None] * 5, [None] * 5))
    print(f"\nmacro AUC = {100 * m['AUC']:.2f}%   accuracy = {100 * m['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
