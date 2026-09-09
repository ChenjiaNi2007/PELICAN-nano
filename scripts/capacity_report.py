#!/usr/bin/env python
"""
scripts/capacity_report.py -- render results/capacity_retrain/ into one HTML report page.

Reads summary.csv + best_by_config.csv (written by scripts/training_curves.py) and embeds
the PNG figures as data URIs, so the page is self-contained and can be published as an
artifact / mailed / dropped in the paper folder. Re-run after training_curves.py to refresh.

  .venv/bin/python scripts/training_curves.py --group-by h && .venv/bin/python scripts/capacity_report.py
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import os
import statistics as st
from collections import defaultdict

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "capacity_retrain")

# The rows that prompted this (workspace results/roc_summary.csv), for the side-by-side.
OLD_ROWS = [
    ("h=2", "quant w6a6i6 pmu12", "fpga_model_qat_w6a6i6p12_best.pt", 0.9519, 67.2, "8 ep, seed 42, best ep 3"),
    ("h=4", "quant w6a6i6 pmu12", "nhid4_best.pt", 0.9573, 87.0, "8 ep, seed 42, best ep 3"),
    ("h=2", "“float” = 24-bit QAT, quantizers off", "fpga_model_qat_best.pt", 0.9639, 110.0, "8 ep, best ep 5"),
    ("h=4", "“float” = 6-bit master weights, quantizers off", "nhid4_best.pt", 0.9267, 25.0, "never a trained model"),
]

MODE_LABEL = {"float": "float (no quantizers)", "qat": "QAT w6 a6 i6 pmu12", "q24": "24-bit QAT (old “float” recipe)",
              "qatf": "QAT w6 a6 i6 pmu12, scales frozen in cooldown (ep 18–20)", "q24f": "24-bit QAT, scales frozen",
              "qatf12": "QAT w6 a6 i6 pmu12, scales frozen from epoch 12"}


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def read_csv(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def f4(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "–"


def f1(v):
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "–"


def main():
    rows = read_csv("summary.csv")
    # scripts/capacity_eval.py: AUC + 1/eps_B at exactly eps_S = 0.30 (interpolated the same way as
    # the roc_summary rows). When present for a run, it replaces the trainer's nearest-ROC-point BR.
    ev = {(e["prefix"], e["which"]): e for e in read_csv("eval.csv")}
    for r in rows:
        b, f = ev.get((r["prefix"], "best")), ev.get((r["prefix"], "final"))
        r["br_src"] = "eval" if b else "trainer"
        if b:
            r["test_auc_best"], r["test_br03_best"] = b["AUC"], b["inv_eps_b_03"]
        if f:
            r["test_auc_final"], r["test_br03_final"] = f["AUC"], f["inv_eps_b_03"]
    done = [r for r in rows if r["status"] == "done"]
    n_eval = sum(1 for r in done if r["br_src"] == "eval")
    running = [r for r in rows if r["status"] != "done"]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- aggregate per config (mean over seeds) for the headline table ----
    groups = defaultdict(list)
    for r in done:
        groups[(r["mode"], r["h"], r["lr"], r["epochs"])].append(r)
    order_mode = {"float": 0, "q24": 1, "q24f": 2, "qat": 3, "qatf": 4, "qatf12": 5}
    cfg_rows = []
    for (mode, h, lr, ep), rs in sorted(groups.items(), key=lambda kv: (order_mode.get(kv[0][0], 9), int(kv[0][3]), float(kv[0][2]), int(kv[0][1]))):
        aucb = [float(r["test_auc_best"]) for r in rs]
        aucf = [float(r["test_auc_final"]) for r in rs]
        brb = [float(r["test_br03_best"]) for r in rs]
        brf = [float(r["test_br03_final"]) for r in rs if r.get("test_br03_final") not in ("", None)]
        be = [int(r["best_epoch"]) for r in rs]
        sd = lambda x: st.stdev(x) if len(x) > 1 else None
        cfg_rows.append(dict(mode=mode, h=h, lr=lr, ep=ep, n=len(rs),
                             auc_b=st.mean(aucb), auc_b_sd=sd(aucb), auc_f=st.mean(aucf), auc_f_sd=sd(aucf),
                             br=st.mean(brb), br_sd=sd(brb), brf=st.mean(brf) if brf else None, brf_sd=sd(brf) if len(brf) > 1 else None,
                             be=st.mean(be), seeds=",".join(r["seed"] for r in rs),
                             src="eval" if all(r["br_src"] == "eval" for r in rs) else "trainer"))

    def pm(m, s):
        return f"{m:.4f}" + (f" <span class='sd'>±{s:.4f}</span>" if s is not None else "")

    def pm1(m, s):
        return f"{m:.1f}" + (f" <span class='sd'>±{s:.1f}</span>" if s is not None else "")

    cfg_html = "".join(
        f"<tr class='m-{c['mode']}'><td><span class='dot'></span>h={c['h']}</td><td>{html.escape(MODE_LABEL.get(c['mode'], c['mode']))}</td>"
        f"<td class='num'>{c['ep']}</td><td class='num'>{c['lr']}</td><td class='num'>{c['n']}</td>"
        f"<td class='num'>{pm(c['auc_b'], c['auc_b_sd'])}</td><td class='num'>{pm(c['auc_f'], c['auc_f_sd'])}</td>"
        f"<td class='num'>{pm1(c['br'], c['br_sd'])}{'' if c['src'] == 'eval' else '*'}</td>"
        f"<td class='num'>{pm1(c['brf'], c['brf_sd']) if c['brf'] is not None else '–'}</td><td class='num'>{c['be']:.0f}</td></tr>"
        for c in cfg_rows) or "<tr><td colspan='10' class='empty'>no finished runs yet</td></tr>"

    old_html = "".join(
        f"<tr><td>{h}</td><td>{html.escape(what)}</td><td class='mono'>{ck}</td><td class='num'>{auc:.4f}</td><td class='num'>{br:.1f}</td><td class='note'>{html.escape(note)}</td></tr>"
        for h, what, ck, auc, br, note in OLD_ROWS)

    run_html = "".join(
        f"<tr class='m-{r['mode']}'><td class='mono'>{html.escape(r['prefix'])}</td><td>{r['status']}</td>"
        f"<td class='num'>{r['epochs_done']}/{r['epochs']}</td><td class='num'>{f1(r['sec_per_epoch'])}</td>"
        f"<td class='num'>{r['best_epoch'] or '–'}</td><td class='num'>{f4(r['best_valid_loss'])}</td><td class='num'>{f4(r['best_valid_auc'])}</td>"
        f"<td class='num'>{f4(r['last_valid_auc'])}</td><td class='num'>{f4(r['test_auc_best'])}</td><td class='num'>{f1(r['test_br03_best'])}</td><td class='num'>{f4(r['test_auc_final'])}</td></tr>"
        for r in rows) or "<tr><td colspan='11' class='empty'>nothing launched yet</td></tr>"

    # ---- headline numbers: best finished config per (h, mode) at production lr ----
    def best_of(h, mode):
        cs = [c for c in cfg_rows if c["h"] == h and c["mode"] == mode]
        return max(cs, key=lambda c: (int(c["ep"]), c["auc_f"])) if cs else None

    def tile(label, mode, h):
        c = best_of(h, mode)
        if c is None:
            return f"<div class='tile m-{mode}'><div class='lab'>{html.escape(label)}</div><div class='val pending'>running</div><div class='sub'>no finished run yet</div></div>"
        return (f"<div class='tile m-{mode}'><div class='lab'>{html.escape(label)}</div><div class='val'>{c['auc_f']:.4f}</div>"
                f"<div class='sub'>test AUC, final annealed epoch · {c['ep']} ep · {c['n']} seed{'s' if c['n'] > 1 else ''} · 1/ε<sub>B</sub>@0.3 = {c['br']:.0f}</div></div>")

    def qtile(label, h):
        for mode, suffix in (("qatf12", " (scales frozen from ep 12)"), ("qatf", " (scales frozen in cooldown)"), ("qat", "")):
            if best_of(h, mode) and (mode == "qat" or best_of(h, mode)["n"] > 1):
                return tile(label + suffix, mode, h)
        return tile(label, "qat", h)
    tiles = tile("float h=2", "float", "2") + tile("float h=4", "float", "4") + qtile("QAT h=2", "2") + qtile("QAT h=4", "4")

    figs = ""
    for name, cap in [("curves_2.png", "h = 2: validation loss and AUC, training loss, and the learning rate actually applied, one line per run. Blue float, red QAT with free scales, orange QAT with scales frozen for the cooldown, purple QAT with scales frozen from epoch 12, green 24-bit; line style = seed. Dots mark the checkpoint the trainer keeps (lowest validation loss)."),
                      ("curves_4.png", "h = 4: same panels."),
                      ("lr_schedule_fix.png", "Learning rate per optimizer step for an 8-epoch cos run: the old cooldown (red) halved the LR every minibatch and was below 1e-30 within one epoch; the fix (blue) decays geometrically to lr_final over the three cooldown epochs.")]:
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            figs += f"<figure><img src='{b64(p)}' alt='{html.escape(cap)}'><figcaption>{html.escape(cap)}</figcaption></figure>"

    # ---- data-driven sentence for the 20-epoch, multi-seed float / QAT comparison ----
    def cfg(h, mode, ep="20"):
        cs = [c for c in cfg_rows if c["h"] == h and c["mode"] == mode and c["ep"] == ep and c["lr"] == "0.0025"]
        return cs[0] if cs else None

    def seed_list(h, mode, ep="20"):
        rs = sorted((r for r in done if r["h"] == h and r["mode"] == mode and r["epochs"] == ep and r["lr"] == "0.0025"), key=lambda r: int(r["seed"]))
        return ", ".join(f"{float(r['test_auc_final']):.4f}" for r in rs)

    fl2, fl4 = cfg("2", "float"), cfg("4", "float")
    if fl2 and fl4 and fl2["n"] > 1 and fl4["n"] > 1:
        seed_sentence = (f"At 20 epochs over {fl2['n']} seeds, float h=2 reaches a final-epoch test AUC of {fl2['auc_f']:.4f} ± {fl2['auc_f_sd']:.4f} "
                         f"(seeds: {seed_list('2', 'float')}) and float h=4 {fl4['auc_f']:.4f} ± {fl4['auc_f_sd']:.4f} (seeds: {seed_list('4', 'float')}). "
                         f"The h=2 spread is not noise around one value: two seeds plateau near 0.953 from epoch 6 on, one jumps to 0.962 at epochs 7–8, "
                         f"so h=2 has at least two basins and the single-seed numbers in the old table depended on which one a run landed in. "
                         f"Every h=4 seed lands within 0.001 of the others, {fl4['auc_f'] - fl2['auc_f']:.3f} above the h=2 mean and "
                         f"{fl4['auc_f'] - max(float(r['test_auc_final']) for r in done if r['h'] == '2' and r['mode'] == 'float' and r['epochs'] == '20'):.3f} above the best h=2 seed.")
    else:
        seed_sentence = "The 20-epoch, three-seed runs in the tables below put error bars on that."
    q2, q4 = cfg("2", "qatf12") or cfg("2", "qatf") or cfg("2", "qat"), cfg("4", "qatf12") or cfg("4", "qatf") or cfg("4", "qat")
    if q2 and q4 and q2["n"] > 1 and q4["n"] > 1:
        qat_sentence = (f" For the firmware operating point (6-bit QAT, {MODE_LABEL[q2['mode']].split(', ', 1)[-1]}), 20 epochs over {q2['n']} seeds give "
                        f"h=2 {q2['auc_f']:.4f} ± {q2['auc_f_sd']:.4f} (seeds: {seed_list('2', q2['mode'])}) and h=4 {q4['auc_f']:.4f} ± {q4['auc_f_sd']:.4f} "
                        f"(seeds: {seed_list('4', q4['mode'])}).")
    else:
        qat_sentence = " The corresponding 6-bit QAT seeds are still training."

    # ---- paired freeze effect: identical seed/h/epochs, only the freeze differs ----
    by_key = {(r["h"], r["epochs"], r["lr"], r["seed"], r["mode"]): r for r in done}
    pair_rows = ""
    for (h, ep, lr, seed) in sorted({(r["h"], r["epochs"], r["lr"], r["seed"]) for r in done if r["mode"].startswith("qat")},
                                    key=lambda t: (int(t[1]), int(t[0]), int(t[3]))):
        cells = []
        for mode in ("qat", "qatf", "qatf12"):
            r = by_key.get((h, ep, lr, seed, mode))
            cells.append(f"<td class='num'>{float(r['test_auc_best']):.4f} / {float(r['test_auc_final']):.4f}</td>" if r else "<td class='num'>–</td>")
        if sum(1 for m in ("qat", "qatf", "qatf12") if (h, ep, lr, seed, m) in by_key) >= 2:
            pair_rows += f"<tr><td>h={h}</td><td class='num'>{ep}</td><td class='num'>{seed}</td>{''.join(cells)}</tr>"
    pair_table = (f"<div class='tablewrap'><table><thead><tr><th>h</th><th class='num'>epochs</th><th class='num'>seed</th>"
                  f"<th class='num'>scales free (best / final)</th><th class='num'>frozen ep E−2 (best / final)</th><th class='num'>frozen ep 12 (best / final)</th></tr></thead>"
                  f"<tbody>{pair_rows}</tbody></table></div>") if pair_rows else "<p class='meta'>No completed pairs yet.</p>"

    # ---- peak-LR sensitivity: seed 1, 20 epochs, one cell per (h, mode) x lr ----
    lrs = ["0.001", "0.0025", "0.005"]
    lr_rows = ""
    for (h, mode) in [("2", "float"), ("4", "float"), ("2", "qatf12"), ("4", "qatf12")]:
        cells = []
        for lr in lrs:
            r = by_key.get((h, "20", lr, "1", mode))
            cells.append(f"<td class='num'>{float(r['test_auc_best']):.4f} / {float(r['test_auc_final']):.4f}</td>" if r else "<td class='num'>–</td>")
        if any((h, "20", lr, "1", mode) in by_key for lr in lrs):
            lr_rows += f"<tr class='m-{mode}'><td><span class='dot'></span>h={h}</td><td>{html.escape(MODE_LABEL.get(mode, mode))}</td>{''.join(cells)}</tr>"
    lr_table = (f"<div class='tablewrap'><table><thead><tr><th>h</th><th>mode</th>"
                + "".join(f"<th class='num'>peak LR {lr} (best / final)</th>" for lr in lrs)
                + f"</tr></thead><tbody>{lr_rows}</tbody></table></div>") if lr_rows else "<p class='meta'>No LR-variant runs finished yet.</p>"

    n_done, n_total = len(done), len(rows)
    page = f"""<title>nanoPELICAN Capacity Retrain</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400;600&display=swap">
<style>
:root {{
  --bg:#F6F7F5; --bg2:#EDEFEB; --ink:#1B1F1E; --ink2:#5D6764; --rule:#D3D8D3;
  --float:#0E6E6B; --qat:#B8471A; --q24:#4E7D2A; --warn:#8A5A00; --warn-bg:#FBF3E0;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  --mono:"Source Code Pro", "SF Mono", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#151817; --bg2:#1E2321; --ink:#E4E8E5; --ink2:#9AA49F; --rule:#323A36;
  --float:#4FB8B3; --qat:#E8804F; --q24:#8CC15E; --warn:#E3B45A; --warn-bg:#2A2410; }} }}
:root[data-theme="dark"] {{
  --bg:#151817; --bg2:#1E2321; --ink:#E4E8E5; --ink2:#9AA49F; --rule:#323A36;
  --float:#4FB8B3; --qat:#E8804F; --q24:#8CC15E; --warn:#E3B45A; --warn-bg:#2A2410; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:16px; line-height:1.55; margin:0; }}
main {{ max-width:1060px; margin:0 auto; padding:40px 24px 72px; }}
.col {{ max-width:72ch; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:2.1rem; line-height:1.15; margin:0 0 6px; text-wrap:balance; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:1.35rem; margin:44px 0 10px; text-wrap:balance; }}
h3 {{ font-family:var(--sans); font-weight:600; font-size:1rem; margin:22px 0 6px; }}
p {{ margin:0 0 12px; }}
.eyebrow {{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--ink2); margin-bottom:10px; }}
.meta {{ color:var(--ink2); font-size:.92rem; margin-bottom:26px; }}
.tiles {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:22px 0 8px; }}
.tile {{ background:var(--bg2); padding:14px 16px 12px; border-top:3px solid var(--ink2); }}
.tile.m-float {{ border-top-color:var(--float); }} .tile.m-qat, .tile.m-qatf, .tile.m-qatf12 {{ border-top-color:var(--qat); }} .tile.m-q24, .tile.m-q24f {{ border-top-color:var(--q24); }}
.tile .lab {{ font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; color:var(--ink2); }}
.tile .val {{ font-family:var(--mono); font-size:1.9rem; font-weight:600; margin:4px 0 2px; font-variant-numeric:tabular-nums; }}
.tile .val.pending {{ font-size:1.1rem; font-weight:400; color:var(--ink2); }}
.tile .sub {{ font-size:.8rem; color:var(--ink2); line-height:1.35; }}
.callout {{ background:var(--warn-bg); border-left:3px solid var(--warn); padding:12px 16px; margin:16px 0; }}
.callout p:last-child {{ margin:0; }}
.tablewrap {{ overflow-x:auto; margin:10px 0 6px; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-size:.76rem; letter-spacing:.06em; text-transform:uppercase; color:var(--ink2); font-weight:600; border-bottom:2px solid var(--rule); }}
td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
td.mono {{ font-family:var(--mono); font-size:.84rem; }}
td.note {{ color:var(--ink2); font-size:.86rem; }}
td.empty {{ color:var(--ink2); font-style:italic; }}
.sd {{ color:var(--ink2); font-size:.8rem; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; background:var(--ink2); vertical-align:baseline; }}
tr.m-float .dot {{ background:var(--float); }} tr.m-qat .dot, tr.m-qatf .dot, tr.m-qatf12 .dot {{ background:var(--qat); }} tr.m-q24 .dot, tr.m-q24f .dot {{ background:var(--q24); }}
figure {{ margin:22px 0; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--rule); background:#fff; }}
figcaption {{ font-size:.86rem; color:var(--ink2); margin-top:8px; max-width:80ch; }}
code {{ font-family:var(--mono); font-size:.86em; background:var(--bg2); padding:1px 5px; }}
pre {{ font-family:var(--mono); font-size:.84rem; background:var(--bg2); padding:12px 14px; overflow-x:auto; }}
ul {{ padding-left:22px; }} li {{ margin-bottom:6px; }}
details summary {{ cursor:pointer; color:var(--ink2); font-size:.92rem; margin:8px 0; }}
.legend {{ display:flex; gap:18px; font-size:.84rem; color:var(--ink2); margin:6px 0 0; flex-wrap:wrap; }}
.legend span::before {{ content:""; display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
.legend .lf::before {{ background:var(--float); }} .legend .lq::before {{ background:var(--qat); }} .legend .l24::before {{ background:var(--q24); }}
@media (max-width:720px) {{ .tiles {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} h1 {{ font-size:1.7rem; }} }}
</style>
<main>
<div class="eyebrow">nanoPELICAN · top tagging · capacity check</div>
<h1>Is h=4 really worse than h=2? Retraining both, properly</h1>
<div class="meta">Full top-tagging set (1,211,000 train / 403,000 valid / 404,000 test jets), first 20 constituents, production recipe. Page generated {now}; {n_done} of {n_total} launched runs finished{', ' + str(len(running)) + ' in progress' if running else ''}. Numbers refresh as the queue advances.</div>

<div class="tiles">{tiles}</div>
<div class="legend"><span class="lf">float: no quantizers</span><span class="lq">QAT: 6-bit weights/acts/dots, 12-bit momenta (firmware point)</span><span class="l24">24-bit QAT: the old “float h=2” recipe</span></div>

<div class="col">
<h2>Short answer</h2>
<p>The number that looked wrong was not a float model. The “float h=4” row (AUC 0.9267) is the 6-bit QAT checkpoint <code>nhid4_best.pt</code> evaluated with its quantizers switched off, i.e. the master weights of a 6-bit network, which was never trained as a float model. The “float h=2” row (0.9639) is a <em>24-bit</em> QAT checkpoint, which is float arithmetic in all but name. The two rows are different objects, so their ordering says nothing about capacity.</p>
<p>Retrained under identical conditions at the old 8-epoch budget, a true float h=4 beats float h=2 by 0.010 AUC (0.9623 vs 0.9522, final epoch), and 6-bit QAT h=4 beats QAT h=2 by 0.0055 (0.9576 vs 0.9521), reproducing the original QAT checkpoints (0.9573 / 0.9519) almost exactly.</p>
<p>{seed_sentence}{qat_sentence}</p>
<p>The old 24-bit h=2 figure of 0.964 therefore sits in the upper h=2 basin, which a plain float run also reaches from a favourable seed (0.962); it is not evidence of anything h=4 lacks. The same 24-bit recipe rerun here gave 0.958 at its best epoch and then lost ground to a late scale flip (next section).</p>

<h2>What the old rows were</h2>
</div>
<div class="tablewrap"><table>
<thead><tr><th>model</th><th>what the row actually is</th><th>checkpoint</th><th class="num">test AUC</th><th class="num">1/ε<sub>B</sub>@0.3</th><th>training</th></tr></thead>
<tbody>{old_html}</tbody></table></div>
<div class="col">
<p>All four came from 8-epoch runs on the same 1.211M-jet training set (checkpoint <code>args.num_train</code>). The 24-bit checkpoint’s learned quantizer ranges (read back with <code>scripts/check_scales.py</code>) clip d<sub>ij</sub> at 2048 GeV², the ReLU output at 2.0, the 2→0 aggregate at ±1 and the logit at ±8. Those saturations are nonlinearities a pure float model does not have, which is why “24-bit QAT” is run here as its own control rather than being called float.</p>

<div class="callout"><p><strong>A training bug affected every earlier checkpoint.</strong> The cooldown scheduler in <code>src/trainer/scheduler.py</code> computed <code>lr = lr0 · 0.5<sup>step</sup></code>, halving the learning rate every <em>minibatch</em>. With 4,731 minibatches per epoch the LR was below 1e-30 within the first cooldown epoch, so the last 3 epochs of every <code>cos</code> run changed nothing (the old logs print identical metrics for epochs 6, 7 and 8). An “8-epoch” run was really 4 warmup epochs plus 1 epoch at peak LR, never annealed. That is why every production checkpoint selected its best epoch at 3–5, inside the warmup. Fixed to the intended geometric decay to <code>lr_final</code> over the cooldown window; pinned by <code>tests/test_scheduler_cooldown.py</code>.</p></div>

<h2>What the anneal does to QAT: late scale flips</h2>
<p>With the cooldown working, float runs improve monotonically through it. QAT runs do not always: the 24-bit h=2 control peaked at epoch 5 (test AUC 0.9580) and finished at 0.9512; 6-bit QAT h=4 dipped from 0.957 to 0.951 in epochs 6–7 before recovering at epoch 8. Reading the learned scales of the best and final checkpoints back shows what moved:</p>
</div>
<div class="tablewrap"><table>
<thead><tr><th>run</th><th>scale that changed between the best and final checkpoint</th><th>effect</th></tr></thead>
<tbody>
<tr><td>24-bit h=2</td><td class="mono">input_quant 2⁻¹⁵ → 2⁻¹⁶</td><td>d<sub>ij</sub> clip 256 → 128 GeV², the collapse basin already documented in <code>quant.py</code> (2 of 12 earlier runs)</td></tr>
<tr><td>QAT h=4</td><td class="mono">pmu_quant 0.25 → 0.125; post_agg_quant 0.125 → 0.0625</td><td>momentum clip 512 → 256 GeV (saturates the leading constituents); aggregated-feature clip halved</td></tr>
</tbody></table></div>
<div class="col">
<p>A power-of-two scale is a discrete parameter: it jumps when its continuous log-scale parameter crosses a rounding boundary, and once the learning rate has annealed to 1e-5 the weights cannot compensate. The trainer’s lowest-validation-loss selection hides the damage but wastes the anneal. The remedy is a new trainer flag, <code>--freeze-scales-epoch</code>, which stops the scales from moving from a given epoch while the weights keep annealing. Two settings are run: frozen for the three cooldown epochs only (E−2), and frozen from epoch 12 of 20 after h=4 seed 2 flipped its momentum clip 512 → 256 GeV during epoch 17, one epoch before the cooldown freeze. Same seed, same everything else, test AUC best-by-loss / final epoch:</p>
</div>
{pair_table}
<div class="col">
<p>Where the unfrozen run flips late, the freeze recovers the final-epoch AUC to the run’s best; where nothing flips, the two agree to the third decimal.</p>

<h2>Setup</h2>
<ul>
<li><strong>Data.</strong> <code>data/toptag20/</code>: the full set truncated to the first 20 constituents (the model only ever sees <code>p[:20]</code> at <code>--nobj 20</code>), float32 momenta. Inputs are bit-identical to loading the 200-constituent files, verified on random jets, and each training process needs 0.6 GB instead of 10 GB, which is what lets this run on the 8 GB laptop.</li>
<li><strong>Recipe</strong> (pinned to the production one): nobj 20, N̄ = 49, batch 256, AdamW, weight decay 0.005, dropout 0.05/0.05, ReLU, masked BatchNorm, <code>cos</code> schedule = 4 warmup epochs + cosine + 3 cooldown epochs, peak LR 0.0025 unless stated.</li>
<li><strong>Checkpoint selection.</strong> The trainer keeps the epoch with the lowest validation loss (“best”). With a working anneal the final epoch usually has slightly higher AUC than that pick, so both are reported.</li>
<li><strong>Compute.</strong> CPU only (Brevitas needs named tensors, unsupported on Apple MPS; the float model is slower on MPS than CPU). Three concurrent 2-thread jobs: about 5 min/epoch float h=2, 6.5 min float h=4, 8 min QAT h=2, 10 min QAT h=4. Everything is driven by <code>scripts/capacity_retrain_queue.sh</code> and is resumable.</li>
</ul>

<h2>Matrix</h2>
</div>
<div class="tablewrap"><table>
<thead><tr><th>phase</th><th>h</th><th>mode</th><th class="num">peak LR</th><th class="num">epochs</th><th>seeds</th><th>purpose</th></tr></thead>
<tbody>
<tr><td>1a</td><td>2, 4</td><td>float, QAT</td><td class="num">0.0025</td><td class="num">8</td><td>1</td><td>old budget with the fixed schedule</td></tr>
<tr><td>1b</td><td>2, 4</td><td>float</td><td class="num">0.0025</td><td class="num">20</td><td>1, 2, 3</td><td>headline float comparison with seed spread</td></tr>
<tr><td>3</td><td>2, 4</td><td>QAT, scales frozen in cooldown</td><td class="num">0.0025</td><td class="num">20</td><td>1, 2, 3</td><td>headline QAT comparison with seed spread</td></tr>
<tr><td>1b′ / 3a</td><td>2, 4</td><td>QAT as-is at 20 ep; QAT frozen at 8 ep</td><td class="num">0.0025</td><td class="num">20 / 8</td><td>1</td><td>quantify the freeze effect</td></tr>
<tr><td>2</td><td>2, 4</td><td>float, then QAT frozen</td><td class="num">0.001, 0.005</td><td class="num">20</td><td>1</td><td>learning-rate sensitivity (last in the queue)</td></tr>
<tr><td>ctrl</td><td>2, 4</td><td>24-bit QAT</td><td class="num">0.0025</td><td class="num">8</td><td>1</td><td>reproduce the old “float h=2” recipe; isolates the learned-clip effect</td></tr>
</tbody></table></div>

<div class="col"><h2>Results by configuration</h2>
<p>Mean over finished seeds; ± is the seed standard deviation where more than one seed has finished. Test set = 404,000 jets.</p></div>
<div class="tablewrap"><table>
<thead><tr><th>h</th><th>mode</th><th class="num">epochs</th><th class="num">peak LR</th><th class="num">seeds</th><th class="num">test AUC, best-by-loss</th><th class="num">test AUC, final epoch</th><th class="num">1/ε<sub>B</sub>@0.3, best</th><th class="num">1/ε<sub>B</sub>@0.3, final</th><th class="num">best epoch</th></tr></thead>
<tbody>{cfg_html}</tbody></table></div>
<div class="col"><p class="meta">1/ε<sub>B</sub> is read at exactly ε<sub>S</sub> = 0.30 on the full ROC (same definition as the DeepSet and MLP-Mixer rows); entries marked * still carry the trainer’s nearest-ROC-point value, which can sit at ε<sub>S</sub> 0.26–0.32 and is only indicative. {n_eval} of {n_done} finished runs re-scored so far.</p></div>

<div class="col"><h2>Peak learning rate</h2>
<p>Same seed (1) and 20 epochs, only the peak LR of the warmup–cosine–cooldown schedule changed. For h=2 float the three LRs land on the same 0.953 basin, so the seed, not the LR, decides which basin a run reaches; for h=4 float 0.001 stalls at 0.962 while 0.0025 and 0.005 both reach 0.966.</p></div>
{lr_table}
<div class="col"><h2>Training curves</h2></div>
{figs}

<div class="col"><h2>Every run</h2></div>
<div class="tablewrap"><table>
<thead><tr><th>run</th><th>status</th><th class="num">epochs</th><th class="num">s/epoch</th><th class="num">best ep</th><th class="num">valid loss</th><th class="num">valid AUC</th><th class="num">last valid AUC</th><th class="num">test AUC best</th><th class="num">1/ε<sub>B</sub></th><th class="num">test AUC final</th></tr></thead>
<tbody>{run_html}</tbody></table></div>

<div class="col">
<h2>Files</h2>
<p>PELICAN-nano: <code>docs/CAPACITY_RETRAIN.md</code> (method), <code>scripts/capacity_retrain.py</code> + <code>capacity_retrain_queue.sh</code> (driver), <code>scripts/training_curves.py</code> (this table and the curves), <code>scripts/make_toptag20.py</code> (data), <code>model/cap_*_best.pt</code> (checkpoints), <code>log/cap_*.log</code> (per-epoch metrics). Monitor a live queue with <code>.venv/bin/python scripts/training_curves.py --watch</code>.</p>
</div>
</main>
"""
    out = os.path.join(OUT, "report.html")
    with open(out, "w") as f:
        f.write(page)
    print("wrote", out, f"({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
