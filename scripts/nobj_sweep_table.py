#!/usr/bin/env python
"""
scripts/nobj_sweep_table.py -- accuracy-vs-particle-count table from results/capacity_retrain/eval.csv.

Groups the cap{N}_h{H}_{mode}_lr0p0025_e20_s{S} rows written by scripts/capacity_eval.py
(tag `cap` = N=20, `cap16` = N=16, ..., `capc{N}` = the --input-clip-min 512 arm, shown as
mode "qat+clip512") and prints, per (h, mode) and per N, the test AUC mean +/- sd over
seeds with the per-seed values underneath, then 1/eps_B at eps_S = 0.30 (interpolated,
the roc_summary.csv definition -- NOT the trainer's nearest-point value).

  .venv/bin/python scripts/nobj_sweep_table.py [best|final]
"""
import csv, os, re, statistics as st, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
rows = list(csv.DictReader(open(os.path.join(REPO, "results", "capacity_retrain", "eval.csv"))))
pat = re.compile(r"^(capc?)(\d*)_h(\d)_(float|qatf12)_lr0p0025_e20_s(\d)$")
data = {}
for r in rows:
    m = pat.match(r["prefix"])
    if not m:
        continue
    tag, n, h, mode, seed = m.groups()
    N = int(n) if n else 20
    if tag == "capc":
        mode = "qat+clip512"
    data.setdefault((h, mode, N, r["which"]), {})[int(seed)] = (float(r["AUC"]), float(r["inv_eps_b_03"]), int(r["epoch"]))

which = sys.argv[1] if len(sys.argv) > 1 else "best"
Ns = sorted({k[2] for k in data})
MODES = ("float", "qatf12", "qat+clip512")
print(f"test AUC ({which} checkpoint), mean +/- sd over seeds [n]; per-seed below")
print(f"{'config':<18}" + "".join(f"{'N=' + str(N):>24}" for N in Ns))
for h in "24":
    for mode in MODES:
        line, seeds_line, any_ = f"h{h} {mode:<14}", f"{'':<18}", False
        for N in Ns:
            d = data.get((h, mode, N, which))
            if not d:
                line += f"{'':>24}"; seeds_line += f"{'':>24}"; continue
            any_ = True
            a = [v[0] for v in d.values()]
            sd = st.stdev(a) if len(a) > 1 else 0.0
            line += f"{st.mean(a):.4f} +/- {sd:.4f} [{len(a)}]".rjust(24)
            seeds_line += ("/".join(f"{v[0]:.4f}" for _, v in sorted(d.items()))).rjust(24)
        if any_:
            print(line); print(seeds_line)
print()
print(f"1/eps_B @ eps_S=0.3 ({which}), mean over seeds")
print(f"{'config':<18}" + "".join(f"{'N=' + str(N):>10}" for N in Ns))
for h in "24":
    for mode in MODES:
        line, any_ = f"h{h} {mode:<14}", False
        for N in Ns:
            d = data.get((h, mode, N, which))
            if not d:
                line += f"{'':>10}"; continue
            any_ = True
            line += f"{st.mean(v[1] for v in d.values()):10.1f}"
        if any_:
            print(line)
