#!/usr/bin/env python
"""
scripts/training_curves.py -- parse PELICAN-nano training logs into curves + a table.

Reads, for every run matching --glob (default 'cap_*', the prefixes written by
scripts/capacity_retrain.py):
  log/<prefix>.log      per-epoch "Epoch N  Training/Validation L: .. ACC: .. AUC: .. BR: ..",
                        "Total epoch time", "Lowest loss achieved" (= best epoch),
                        "Best/Final Testing" (test metrics of the best-valid-loss and
                        last checkpoints), and the Namespace line (run args).
  log/<prefix>.stdout   per-minibatch lines; the LR at the last minibatch of each epoch.

Writes to --outdir (default results/capacity_retrain/):
  summary.csv                 one row per run (status, best epoch, valid/test metrics)
  curves_<group>.png          valid loss / valid AUC / train loss / LR vs epoch, one line
                              per run, grouped by --group-by (default: mode, i.e. float
                              vs qat), coloured by n_hidden, seed as line style.
  best_by_config.csv          mean +/- std over seeds of the best-checkpoint test AUC and
                              1/eps_B@0.3 for each (h, mode, lr, epochs)

  --watch    print the table only (no files), for monitoring running jobs
  --no-plots skip PNGs

Usage (repo root, venv python):
  .venv/bin/python scripts/training_curves.py
  .venv/bin/python scripts/training_curves.py --watch
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

EPOCH_RE = re.compile(
    r"^Epoch (\d+)\s+(Training|Validation)\s+L:\s*([\d.]+), ACC:\s*([\d.]+), AUC:\s*([\d.]+),"
    r"\s*BR:\s*([\d.]+) @ ([\d.]+),\s*BR:\s*([\d.]+) @ ([\d.]+),\s*FP:\s*([\d.]+), FN:\s*([\d.]+)")
TEST_RE = re.compile(
    r"^(Best|Final)\s+Testing\s+L:\s*([\d.]+), ACC:\s*([\d.]+), AUC:\s*([\d.]+),"
    r"\s*BR:\s*([\d.]+) @ ([\d.]+),\s*BR:\s*([\d.]+) @ ([\d.]+),\s*FP:\s*([\d.]+), FN:\s*([\d.]+)")
TIME_RE = re.compile(r"^Total epoch time:\s*([\d.]+)s")
START_RE = re.compile(r"^STARTING Epoch (\d+)")
BEST_RE = re.compile(r"^Lowest loss achieved!")
NS_RE = re.compile(r"^Namespace\((.*)\)\s*$")
LR_RE = re.compile(r"E:\s*(\d+)/(\d+), B:\s*(\d+)/(\d+).*?\s([\d.]+E[+-]\d+)\s*$")
PREFIX_RE = re.compile(r"^(?P<tag>[A-Za-z0-9]+)_h(?P<h>\d+)_(?P<mode>[a-z0-9]+)_lr(?P<lr>[0-9pm]+)_e(?P<e>\d+)_s(?P<seed>\d+)$")


def parse_namespace(line: str) -> dict:
    """Pull the handful of args we care about out of the logged Namespace(...) line."""
    body = NS_RE.match(line).group(1)
    out = {}
    for key in ("n_hidden", "quant", "lr_init", "lr_final", "lr_decay_type", "num_epoch",
                "seed", "weight_bit_width", "act_bit_width", "input_bit_width",
                "pmu_bit_width", "datadir", "num_train", "num_valid", "batch_size"):
        m = re.search(rf"\b{key}=('[^']*'|[^,)]+)", body)
        if m:
            out[key] = m.group(1).strip("'")
    return out


def parse_run(prefix: str) -> dict:
    log = os.path.join(REPO, "log", f"{prefix}.log")
    r = {"prefix": prefix, "epochs": {}, "test": {}, "best_epoch": None, "args": {},
         "status": "missing", "epoch_time": [], "lr": {}}
    if not os.path.exists(log):
        return r
    cur = None
    with open(log, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if NS_RE.match(line):
                r["args"] = parse_namespace(line)
                continue
            m = START_RE.match(line)
            if m:
                cur = int(m.group(1)); continue
            m = EPOCH_RE.match(line)
            if m:
                ep = int(m.group(1)); split = "train" if m.group(2) == "Training" else "valid"
                d = r["epochs"].setdefault(ep, {})
                d[split] = {"loss": float(m.group(3)), "acc": float(m.group(4)), "auc": float(m.group(5)),
                            "br03": float(m.group(6)), "br05": float(m.group(8)),
                            "fp": float(m.group(10)), "fn": float(m.group(11))}
                continue
            m = TIME_RE.match(line)
            if m and cur is not None:
                r["epoch_time"].append(float(m.group(1))); continue
            if BEST_RE.match(line) and cur is not None:
                r["best_epoch"] = cur; continue
            m = TEST_RE.match(line)
            if m:
                r["test"][m.group(1).lower()] = {"loss": float(m.group(2)), "acc": float(m.group(3)),
                                                 "auc": float(m.group(4)), "br03": float(m.group(5)),
                                                 "br05": float(m.group(7))}
                continue
            if line.startswith("Inference phase complete"):
                r["status"] = "done"
    if r["status"] != "done":
        r["status"] = "running" if r["epochs"] else "started"
        # a crashed run looks like "running"; the manifest rc disambiguates
    # LR trace from stdout (last minibatch of each epoch)
    so = os.path.join(REPO, "log", f"{prefix}.stdout")
    if os.path.exists(so):
        with open(so, "r", errors="replace") as f:
            for line in f:
                m = LR_RE.search(line)
                if m and m.group(3) == m.group(4):
                    r["lr"][int(m.group(1))] = float(m.group(5))
    m = PREFIX_RE.match(prefix)
    r["cfg"] = m.groupdict() if m else {}
    return r


def row_of(r: dict) -> dict:
    a, cfg = r["args"], r.get("cfg", {})
    n_done = len([e for e, d in r["epochs"].items() if "valid" in d])
    be = r["best_epoch"]
    bv = r["epochs"].get(be, {}).get("valid", {}) if be else {}
    tb, tf = r["test"].get("best", {}), r["test"].get("final", {})
    last = r["epochs"].get(max(r["epochs"]) if r["epochs"] else 0, {})
    mode = cfg.get("mode") or ("qat" if a.get("quant") == "True" else "float")
    return {
        "prefix": r["prefix"], "status": r["status"],
        "h": a.get("n_hidden", cfg.get("h")), "mode": mode,
        "lr": a.get("lr_init", cfg.get("lr")), "decay": a.get("lr_decay_type", ""),
        "epochs": a.get("num_epoch", cfg.get("e")), "seed": a.get("seed", cfg.get("seed")),
        "epochs_done": n_done, "sec_per_epoch": round(st.mean(r["epoch_time"]), 1) if r["epoch_time"] else "",
        "best_epoch": be or "",
        "best_valid_loss": bv.get("loss", ""), "best_valid_auc": bv.get("auc", ""), "best_valid_br03": bv.get("br03", ""),
        "last_valid_loss": last.get("valid", {}).get("loss", ""), "last_valid_auc": last.get("valid", {}).get("auc", ""),
        "last_train_loss": last.get("train", {}).get("loss", ""), "last_train_auc": last.get("train", {}).get("auc", ""),
        "test_auc_best": tb.get("auc", ""), "test_br03_best": tb.get("br03", ""), "test_acc_best": tb.get("acc", ""),
        "test_auc_final": tf.get("auc", ""), "test_br03_final": tf.get("br03", ""),
        "datadir": a.get("datadir", ""),
    }


def fmt(v, nd=4):
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if nd else f"{v:.0f}"
    return str(v)


def print_table(rows):
    cols = [("prefix", 34), ("status", 8), ("done", 5), ("s/ep", 6), ("best", 4),
            ("v_loss", 7), ("v_auc", 7), ("v_br03", 7), ("last_v_auc", 10),
            ("t_auc", 7), ("t_br03", 7), ("t_auc_fin", 9)]
    print("  ".join(f"{c:<{w}}" for c, w in cols))
    for r in rows:
        vals = [r["prefix"], r["status"], f"{r['epochs_done']}/{r['epochs']}", fmt(r["sec_per_epoch"], 0),
                fmt(r["best_epoch"]), fmt(r["best_valid_loss"]), fmt(r["best_valid_auc"]),
                fmt(r["best_valid_br03"], 1), fmt(r["last_valid_auc"]), fmt(r["test_auc_best"]),
                fmt(r["test_br03_best"], 1), fmt(r["test_auc_final"])]
        print("  ".join(f"{v:<{w}}" for v, (c, w) in zip(vals, cols)))


def aggregate(rows):
    g = defaultdict(list)
    for r in rows:
        if r["test_auc_best"] == "":
            continue
        g[(r["h"], r["mode"], r["lr"], r["decay"], r["epochs"])].append(r)
    out = []
    for (h, mode, lr, decay, ep), rs in sorted(g.items(), key=lambda kv: (kv[0][1], int(kv[0][0]), float(kv[0][2]), int(kv[0][4]))):
        aucs = [r["test_auc_best"] for r in rs]; brs = [r["test_br03_best"] for r in rs]
        vl = [r["best_valid_loss"] for r in rs]; be = [r["best_epoch"] for r in rs]
        sd = (lambda x: st.stdev(x) if len(x) > 1 else 0.0)
        out.append({"h": h, "mode": mode, "lr": lr, "decay": decay, "epochs": ep, "n_seeds": len(rs),
                    "test_auc_mean": round(st.mean(aucs), 4), "test_auc_std": round(sd(aucs), 4),
                    "test_auc_max": round(max(aucs), 4),
                    "test_br03_mean": round(st.mean(brs), 1), "test_br03_std": round(sd(brs), 1),
                    "best_valid_loss_mean": round(st.mean(vl), 4), "best_epoch_mean": round(st.mean(be), 1),
                    "seeds": " ".join(str(r["seed"]) for r in rs)})
    return out


def plot(runs, rows, outdir, group_by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for r, row in zip(runs, rows):
        if not r["epochs"]:
            continue
        groups[row[group_by]].append((r, row))
    h_colors = {"2": "#1f77b4", "4": "#d62728", "1": "#7f7f7f", "3": "#2ca02c", "6": "#9467bd"}
    mode_colors = {"float": "#1f77b4", "qat": "#d62728", "q24": "#2ca02c", "qatf": "#ff7f0e", "q24f": "#17becf", "qatf12": "#9467bd"}
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    files = []
    for gname, items in sorted(groups.items()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        ax_vl, ax_va, ax_tl, ax_lr = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
        seeds = sorted({row["seed"] for _, row in items}, key=lambda s: int(s))
        for r, row in items:
            eps = sorted(e for e in r["epochs"] if "valid" in r["epochs"][e])
            if not eps:
                continue
            # colour by the dimension NOT used for grouping (h when grouping by mode, else mode)
            mode_key = row["mode"] if row["mode"] in mode_colors else ("qatf" if row["mode"].startswith("qatf") else row["mode"])
            c = mode_colors.get(mode_key) if group_by == "h" else h_colors.get(str(row["h"]))
            ls = styles[seeds.index(row["seed"]) % len(styles)]
            lab = f"h={row['h']} {row['mode']} lr={row['lr']} e={row['epochs']} s={row['seed']}"
            ax_vl.plot(eps, [r["epochs"][e]["valid"]["loss"] for e in eps], ls, color=c, label=lab, lw=1.4)
            ax_va.plot(eps, [r["epochs"][e]["valid"]["auc"] for e in eps], ls, color=c, lw=1.4)
            teps = [e for e in eps if "train" in r["epochs"][e]]
            ax_tl.plot(teps, [r["epochs"][e]["train"]["loss"] for e in teps], ls, color=c, lw=1.4)
            if r["lr"]:
                leps = sorted(r["lr"]); ax_lr.plot(leps, [r["lr"][e] for e in leps], ls, color=c, lw=1.4)
            if r["best_epoch"] and r["best_epoch"] in r["epochs"] and "valid" in r["epochs"][r["best_epoch"]]:
                ax_vl.plot([r["best_epoch"]], [r["epochs"][r["best_epoch"]]["valid"]["loss"]], "o", color=c, ms=5)
                ax_va.plot([r["best_epoch"]], [r["epochs"][r["best_epoch"]]["valid"]["auc"]], "o", color=c, ms=5)
        # focus the y-ranges on epoch >= 2 (epoch-1 values are off-scale and uninformative)
        vl2 = [r["epochs"][e]["valid"]["loss"] for r, _ in items for e in r["epochs"] if e >= 2 and "valid" in r["epochs"][e]]
        va2 = [r["epochs"][e]["valid"]["auc"] for r, _ in items for e in r["epochs"] if e >= 2 and "valid" in r["epochs"][e]]
        tl2 = [r["epochs"][e]["train"]["loss"] for r, _ in items for e in r["epochs"] if e >= 2 and "train" in r["epochs"][e]]
        if vl2: ax_vl.set_ylim(min(vl2) - 0.005, max(vl2) + 0.01)
        if va2: ax_va.set_ylim(min(va2) - 0.005, max(va2) + 0.003)
        if tl2: ax_tl.set_ylim(min(tl2) - 0.005, max(tl2) + 0.01)
        ax_vl.set_title("validation loss (dot = selected best epoch)"); ax_va.set_title("validation AUC")
        ax_tl.set_title("training loss"); ax_lr.set_title("learning rate (end of epoch)")
        ax_lr.set_yscale("log"); ax_tl.set_xlabel("epoch"); ax_lr.set_xlabel("epoch")
        for ax in (ax_vl, ax_va, ax_tl, ax_lr):
            ax.grid(alpha=0.3)
        ax_vl.legend(fontsize=7, ncol=2)
        fig.suptitle(f"{group_by} = {gname}: nanoPELICAN training curves (data: {items[0][1]['datadir']})")
        fig.tight_layout()
        fn = os.path.join(outdir, f"curves_{gname}.png")
        fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="cap_*", help="prefix glob under log/")
    p.add_argument("--outdir", default="results/capacity_retrain")
    p.add_argument("--group-by", default="mode", choices=["mode", "h", "lr", "epochs", "seed"])
    p.add_argument("--watch", action="store_true", help="print the table only")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    prefixes = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(REPO, "log", a.glob + ".log")))
    runs = [parse_run(pf) for pf in prefixes]
    rows = [row_of(r) for r in runs]
    rows_sorted = sorted(zip(runs, rows), key=lambda t: (t[1]["mode"], int(t[1]["h"] or 0), float(t[1]["lr"] or 0), int(t[1]["epochs"] or 0), int(t[1]["seed"] or 0)))
    runs, rows = [t[0] for t in rows_sorted], [t[1] for t in rows_sorted]
    print_table(rows)
    agg = aggregate(rows)
    if agg:
        print("\nbest-checkpoint test metrics, mean over seeds:")
        for g in agg:
            print(f"  h={g['h']} {g['mode']:5s} lr={g['lr']} {g['decay']} e={g['epochs']:>3} n={g['n_seeds']}  "
                  f"AUC {g['test_auc_mean']:.4f} +/- {g['test_auc_std']:.4f} (max {g['test_auc_max']:.4f})  "
                  f"1/eps_B@0.3 {g['test_br03_mean']:.1f} +/- {g['test_br03_std']:.1f}  best_epoch~{g['best_epoch_mean']}")
    if a.watch:
        return
    outdir = os.path.join(REPO, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    if rows:
        with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if agg:
        with open(os.path.join(outdir, "best_by_config.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    if not a.no_plots:
        for fn in plot(runs, rows, outdir, a.group_by):
            print("wrote", fn)
    print("wrote", os.path.join(outdir, "summary.csv"))


if __name__ == "__main__":
    main()
