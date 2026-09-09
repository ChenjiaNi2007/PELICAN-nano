#!/usr/bin/env python
"""
scripts/capacity_retrain.py -- h=2 vs h=4 retrain matrix with per-epoch logging.

Why this exists
---------------
The h=4 "float" point in results/roc_summary.csv (AUC 0.9267) is NOT a float-trained
model: it is the 6/6/6-pmu12 QAT checkpoint (nhid4_best.pt) evaluated with its
quantizers switched off, whereas the h=2 "float" row is a 24-bit checkpoint. The two
rows are different objects, so "float h=4 < float h=2" was never a like-for-like
comparison. Every existing production checkpoint (h=2 and h=4) was also trained for
only 8 epochs and selected its best epoch at 3-5, i.e. INSIDE the 4-epoch LR warmup.

This driver retrains h=2 and h=4, in a real float mode and in the production QAT
mode, over a small grid of seeds / peak LR / epoch budgets, with per-epoch train and
validation metrics written to predict/<prefix>.metrics.{train,valid}.csv
(--summarize-csv all) and the per-step LR captured from stdout, so
scripts/training_curves.py can draw the curves and tabulate the best checkpoints.

Recipe (everything except the swept knobs is pinned to the production recipe used
by scripts/sweep_pmu_width.sh and nPELICAN-fpga/sweep/sweep_nhidden.sh):
  nobj 20, nobj-avg 49, batch 256, drop 0.05/0.05, wd 0.005, relu, batchnorm b,
  cos LR decay (4-epoch warmup + cosine + 3-epoch cooldown, min 8 epochs).
  qat   = --quant --po2-scales w6 a6 i6 pmu12   (the firmware operating point)
  q24   = --quant --po2-scales w24 a24 i24, float momenta (the old "float h=2" row's
          recipe: ~float arithmetic but with a learned d_ij clip at the input)
  float = no quantizers at all (true float model; the default code path)
  qatf / q24f = same as qat / q24 plus --freeze-scales-epoch (E-2): quantizer scales are
          frozen for the cooldown. Motivation: measured po2 scale flips late in training
          (d_ij clip 256->128, momentum clip 512->256) that the annealed weights cannot
          absorb; those runs peaked at epoch 5 and lost ~0.005 AUC by the end.

Prefix: cap_h{H}_{mode}_lr{LR}_e{E}_s{SEED}   (LR with '.' -> 'p'; no '-' anywhere,
the trainer splits the prefix on '-' when naming <prefix>.Best.metrics.csv).

Usage (repo root, venv python; CPU, jobs run concurrently with THREADS torch threads
each):
  .venv/bin/python scripts/capacity_retrain.py --hs 2 4 --modes float qat \
      --seeds 1 2 3 --lrs 0.0025 --epochs 20 --jobs 3 --threads 2
  ... --device mps --no-reproducible   (M1 GPU: ~1.7x faster than CPU for QAT, slower than
                                       CPU for float; one job at a time)
  ... --dry-run           print the commands only
  ... --force             rerun runs whose log already says "Inference phase complete"
  Resume: an unfinished run whose model/<prefix>.pt loads is continued with --load from
  its last completed epoch (the trainer replays the LR schedule); just re-run the same
  command / queue script after a crash or power loss.
  ... --extra "--lr-decay-type flat"   pass-through flags appended to every run
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import itertools
import os
import re
import shlex
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

RECIPE = ["--target", "is_signal", "--nobj", "20", "--nobj-avg", "49",
          "--batch-size", "256", "--drop-rate", "0.05", "--drop-rate-out", "0.05",
          "--weight-decay", "0.005", "--activation", "relu", "--batchnorm", "b"]
QAT = ["--quant", "--po2-scales", "--weight-bit-width", "6", "--act-bit-width", "6",
       "--input-bit-width", "6", "--pmu-bit-width", "12"]
# q24: the recipe of the old "float h=2" reference (model/fpga_model_qat_best.pt):
# 24-bit quantizers, float momenta. Numerically ~float EXCEPT that the learned
# input_quant scale also CLIPS d_ij (saturation of the Pareto tail), which a pure
# float run lacks -- this mode isolates that effect from capacity.
Q24 = ["--quant", "--po2-scales", "--weight-bit-width", "24", "--act-bit-width", "24",
       "--input-bit-width", "24"]


def lr_tag(lr: float) -> str:
    s = f"{lr:g}"
    return s.replace(".", "p").replace("-", "m")


def prefix_for(h: int, mode: str, lr: float, epochs: int, seed: int, tag: str) -> str:
    return f"{tag}_h{h}_{mode}_lr{lr_tag(lr)}_e{epochs}_s{seed}"


def is_done(prefix: str) -> bool:
    log = os.path.join(REPO, "log", f"{prefix}.log")
    best = os.path.join(REPO, "model", f"{prefix}_best.pt")
    if not (os.path.exists(log) and os.path.exists(best)):
        return False
    with open(log, "r", errors="replace") as f:
        return "Inference phase complete" in f.read()


def resumable(prefix):
    """True if model/<prefix>.pt exists and loads (an unfinished run we can continue)."""
    ck = os.path.join(REPO, "model", f"{prefix}.pt")
    if not os.path.exists(ck):
        return False
    try:
        import torch
        sys.path.insert(0, REPO)
        ep = torch.load(ck, map_location="cpu", weights_only=False)["epoch"]
        return ep >= 1
    except Exception as e:  # truncated / corrupt file (machine died mid-write): start over
        bad = ck + ".corrupt"
        os.replace(ck, bad)
        print(f"checkpoint {ck} unreadable ({e}); moved to {bad}, restarting {prefix} from scratch")
        return False


def build_cmd(a, h, mode, lr, epochs, seed, prefix):
    cmd = [sys.executable, "-u", "train_pelican_nano.py", "--prefix", prefix,   # -u: LR trace survives a crash
           "--datadir", a.datadir, "--n-hidden", str(h), "--num-epoch", str(epochs),
           "--lr-init", f"{lr:g}", "--lr-decay-type", a.decay, "--seed", str(seed),
           "--summarize-csv", "all", "--no-predict", "--" + a.device] + RECIPE
    m = re.fullmatch(r"(float|qat|q24)(?:f(\d*))?", mode)
    if not m:
        raise ValueError(f"unknown mode {mode}")
    base, freeze = m.group(1), m.group(2)
    if base == "qat":
        cmd += QAT
    elif base == "q24":
        cmd += Q24
    if freeze is not None and base != "float":
        # 'qatf' / 'q24f': freeze the learned po2 scales for the 3-epoch cos cooldown
        # (epochs E-2..E). 'qatf12': freeze from the start of epoch 12 -- measured on
        # h=4: the momentum clip still flipped 512->256 GeV in epoch 17 (LR 7e-4),
        # one epoch before the E-2 freeze, so the cooldown-only freeze is too late.
        n = int(freeze) if freeze else max(1, epochs - 2)
        cmd += ["--freeze-scales-epoch", str(n)]
    if a.num_train > 0:
        cmd += ["--num-train", str(a.num_train)]
    if a.num_valid > 0:
        cmd += ["--num-valid", str(a.num_valid)]
    if a.no_reproducible:
        cmd += ["--no-reproducible"]
    if resumable(prefix):
        cmd += ["--load"]   # continue from model/<prefix>.pt (trainer replays the LR schedule)
    if a.extra:
        cmd += shlex.split(a.extra)
    return cmd


def run_one(cmd, prefix, threads, manifest):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    out = os.path.join(REPO, "log", f"{prefix}.stdout")
    t0 = time.time()
    started = dt.datetime.now().isoformat(timespec="seconds")
    with open(out, "a" if "--load" in cmd else "w") as fo:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fo, stderr=subprocess.STDOUT)
    row = [prefix, started, dt.datetime.now().isoformat(timespec="seconds"),
           f"{time.time()-t0:.0f}", str(rc), " ".join(shlex.quote(c) for c in cmd[1:])]
    with open(manifest, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"[{row[2]}] {prefix} rc={rc} wall={row[3]}s", flush=True)
    return prefix, rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hs", type=int, nargs="+", default=[2, 4])
    p.add_argument("--modes", nargs="+", default=["float", "qat"], help="float | qat | q24 | qatf | q24f | qatfN (freeze from epoch N, e.g. qatf12)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.0025])
    p.add_argument("--epochs", type=int, nargs="+", default=[20])
    p.add_argument("--decay", default="cos", help="--lr-decay-type (cos needs >= 8 epochs)")
    p.add_argument("--datadir", default="data/toptag20")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                   help="cpu | mps | cuda  (GPU QAT needs --extra --no-reproducible: Brevitas kthvalue calibration)")
    p.add_argument("--num-train", type=int, default=-1)
    p.add_argument("--num-valid", type=int, default=-1)
    p.add_argument("--jobs", type=int, default=3, help="concurrent training processes")
    p.add_argument("--threads", type=int, default=2, help="torch/OMP threads per process")
    p.add_argument("--tag", default="cap", help="prefix tag (results grouped by it)")
    p.add_argument("--no-reproducible", action="store_true",
                   help="pass --no-reproducible to the trainer (required for QAT on cuda/mps: "
                        "Brevitas scale calibration uses kthvalue, no deterministic kernel)")
    p.add_argument("--extra", default="", help="extra flags appended to every run; "
                   "values starting with '-' need the --extra=\"...\" form")
    p.add_argument("--order", default="seed", choices=["seed", "grid"],
                   help="seed: interleave so every (h,mode) gets seed 1 before anyone gets seed 2")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    grid = list(itertools.product(a.epochs, a.lrs, a.seeds, a.hs, a.modes))
    if a.order == "seed":
        grid.sort(key=lambda t: (t[0], t[1], t[2]))  # (epochs, lr, seed) major; h/mode interleaved
    for d in ("log", "model", "predict"):
        os.makedirs(os.path.join(REPO, d), exist_ok=True)
    manifest = os.path.join(REPO, "log", f"{a.tag}_retrain_manifest.csv")
    if not os.path.exists(manifest):
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerow(["prefix", "started", "finished", "wall_s", "rc", "args"])

    jobs = []
    for epochs, lr, seed, h, mode in grid:
        prefix = prefix_for(h, mode, lr, epochs, seed, a.tag)
        if not a.force and is_done(prefix):
            print(f"skip (done): {prefix}")
            continue
        jobs.append((build_cmd(a, h, mode, lr, epochs, seed, prefix), prefix))

    print(f"{len(jobs)} runs to launch, {a.jobs} at a time, {a.threads} threads each "
          f"(datadir={a.datadir}, decay={a.decay})", flush=True)
    for cmd, prefix in jobs:
        print("  " + " ".join(shlex.quote(c) for c in cmd[1:]))
    if a.dry_run or not jobs:
        return
    failures = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, cmd, prefix, a.threads, manifest) for cmd, prefix in jobs]
        for fut in cf.as_completed(futs):
            prefix, rc = fut.result()
            if rc != 0:
                failures.append(prefix)
    print(f"done: {len(jobs)-len(failures)} ok, {len(failures)} failed {failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
