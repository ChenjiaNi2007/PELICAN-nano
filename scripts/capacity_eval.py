#!/usr/bin/env python
"""
scripts/capacity_eval.py -- score every finished capacity-retrain checkpoint on the full
404k-jet test set with the SAME metric definitions as results/roc_summary.csv.

Why: the trainer's "BR @ 0.3" takes the ROC point NEAREST to eps_S = 0.30, which on a
coarse (drop_intermediate) curve can sit at 0.26 or 0.32 and swing 1/eps_B by +/-30%.
scripts/eval_toptag_logits.py walks the full ROC and reads 1/fpr at the first point with
tpr >= 0.30 (what the DeepSet / MLP-Mixer rows use), so numbers here are comparable to the
existing summary rows. It also saves the per-jet logits (test-file order) for ROC overlays.

For each model/cap_*_best.pt (and the matching final-epoch model/cap_*.pt) whose run has
finished:  results/capacity_retrain/logits/<prefix>_<best|final>.npz  +  one row in
results/capacity_retrain/eval.csv (prefix, which, epoch, AUC, inv_eps_b_03, accuracy).

  .venv/bin/python scripts/capacity_eval.py [--glob 'cap_*'] [--testfile data/toptag20/test.h5]
Re-runnable: checkpoints already in eval.csv are skipped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)  # checkpoint pickles reference `src` (scheduler state)
OUT = os.path.join(REPO, "results", "capacity_retrain")
LOGITS = os.path.join(OUT, "logits")
CSV = os.path.join(OUT, "eval.csv")
FIELDS = ["prefix", "which", "epoch", "AUC", "inv_eps_b_03", "accuracy", "n"]


def finished(prefix):
    log = os.path.join(REPO, "log", f"{prefix}.log")
    return os.path.exists(log) and "Inference phase complete" in open(log, errors="replace").read()


def epoch_of(ckpt):
    import torch
    return int(torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="cap_*")
    p.add_argument("--testfile", default="data/toptag20/test.h5")
    p.add_argument("--which", nargs="+", default=["best", "final"])
    a = p.parse_args()
    os.makedirs(LOGITS, exist_ok=True)
    done = set()
    if os.path.exists(CSV):
        with open(CSV, newline="") as f:
            done = {(r["prefix"], r["which"]) for r in csv.DictReader(f)}
    else:
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    prefixes = sorted(os.path.basename(m)[:-len("_best.pt")] for m in glob.glob(os.path.join(REPO, "model", a.glob + "_best.pt")))
    for prefix in prefixes:
        if not finished(prefix):
            print(f"skip (not finished): {prefix}"); continue
        for which in a.which:
            if (prefix, which) in done:
                continue
            ckpt = f"model/{prefix}_best.pt" if which == "best" else f"model/{prefix}.pt"
            if not os.path.exists(os.path.join(REPO, ckpt)):
                continue
            out = os.path.join("results", "capacity_retrain", "logits", f"{prefix}_{which}.npz")
            cmd = [sys.executable, "scripts/eval_toptag_logits.py", "--checkpoint", ckpt, "--testfile", a.testfile, "--out", out]
            res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            m = re.search(r"AUC=([\d.]+)\s+1/eps_B@0.3=([\d.]+|inf)\s+accuracy=([\d.]+)\s+n=(\d+)", res.stdout)
            if res.returncode != 0 or not m:
                print(f"FAILED {prefix} {which}: {res.stderr[-400:]}"); continue
            row = [prefix, which, epoch_of(os.path.join(REPO, ckpt)), m.group(1), m.group(2), m.group(3), m.group(4)]
            with open(CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(" ".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
