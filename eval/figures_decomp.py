"""Decomposition-study figures -> paper/figs/fig_decomp.pdf, fig_decomp_examples.pdf.

fig_decomp: paired dDFD vs B (concat scoring) per dataset x subset, on ids shared by every condition.
fig_decomp_examples: rule-based pick. VABench-V multi-stop sample with the median Agent-Groq(concat) minus B DFD
among common ids; shows the image with GT / B / Agent-local / Agent-Groq paths and both plans.
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decomp_analysis import G, L, SRC, load, paired
from figures import forest
from qualitative import B_COL, INK, median_pick, path, show
from run_traj import load_jsonl

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
OUT = f"{ROOT}/paper/figs"
SUB = json.load(open(f"{ROOT}/data/subsets.json"))
AG_COL, GQ_COL = "#7a5af5", "#d6542a"


def conds(ds):
    gdir, ldir = SRC[ds]
    by = load(f"{gdir}/traj.jsonl")
    by.update({k: v for k, v in load(f"{ldir}/traj.jsonl").items() if k.startswith(L)})
    keep = set.intersection(*(set(by[c]) for c in (L + ":concat", G + ":concat", "C-pred", "B")))
    return by, keep


panels = [("vabench", "multi", "VABench-V, multi-stop"), ("sharerobot", "multi", "ShareRobot, multi-stop"),
          ("sharerobot", "single", "ShareRobot, single-stop")]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
for ax, (ds, s, title) in zip(axes, panels):
    by, keep = conds(ds)
    ids = sorted(i for i in keep if SUB[ds][i] == s)
    rows = []
    for lab, c in (("C-pred (single call)", "C-pred"), ("Agent-local", L + ":concat"),
                   ("Agent-Groq", G + ":concat"), ("Agent-Groq + arm", G + ":concat+arm")):
        r = paired(by[c], by["B"], ids)
        rows.append((lab, r["mean"], r["ci"]))
    forest(ax, rows, f"{title} (n={len(ids)})", r"$\Delta$DFD vs. B  ($\leftarrow$ better)")
    r = paired(by[G + ":concat"], by["C-pred"], ids)
    print(f"{ds} {s}: Groq concat - C-pred {r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.3g} n={r['n']}")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_decomp.pdf")
print("wrote fig_decomp.pdf")

# Qualitative example
ds = "vabench"
by, keep = conds(ds)
gt = {r["id"]: r for r in load_jsonl(f"{ROOT}/data/vabench_v.jsonl")}
gr = {r["id"]: r for r in load_jsonl(f"{SRC[ds][0]}/trajectory_runs.jsonl")}
lr = {r["id"]: r for r in load_jsonl(f"{SRC[ds][1]}/trajectory_runs.jsonl")}
gc, lc, b = by[G + ":concat"], by[L + ":concat"], by["B"]
cands = [(gc[i]["metrics"]["dfd"] - b[i]["metrics"]["dfd"], i) for i in keep
         if SUB[ds][i] == "multi" and gc[i]["valid"] and lc[i]["valid"] and b[i]["valid"]]
pick = median_pick(cands)


def plan_text(title, run):
    lines = [title]
    for k, st in enumerate(run["steps"], 1):
        lines += textwrap.wrap(textwrap.shorten(f"{k}. [{st['task']}] {st['prompt']}", 120), 44, subsequent_indent="      ")
        lines += textwrap.wrap(textwrap.shorten(f"    → {st['raw']}", 60), 44, subsequent_indent="      ")
    return "\n".join(lines)


fig = plt.figure(figsize=(7.0, 2.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
ax = fig.add_subplot(gs[0])
wh = show(ax, gt[pick]["image"])
path(ax, gt[pick]["gt_traj"], wh, INK, "--")
path(ax, b[pick]["pred"], wh, B_COL)
path(ax, lc[pick]["pred"], wh, AG_COL)
path(ax, gc[pick]["pred"], wh, GQ_COL)
d = {k: v[pick]["metrics"]["dfd"] for k, v in (("B", b), ("L", lc), ("G", gc))}
ax.set_title(f"(a) {gt[pick]['instruction']}\nDFD: B {d['B']:.2f}, local {d['L']:.2f}, Groq {d['G']:.2f}",
             fontsize=7, color=INK)
for k, (t, run) in enumerate((("(b) Agent-Groq plan", gr[pick]), ("(c) Agent-local plan", lr[pick])), 1):
    ax = fig.add_subplot(gs[k])
    ax.axis("off")
    ax.text(0, 1, plan_text(f"{t}: {len(run['steps'])} calls, {run['latency']:.1f} s", run), va="top", ha="left",
            fontsize=5.6, family="monospace", color=INK, transform=ax.transAxes)
handles = [plt.Line2D([], [], color=INK, ls="--", label="ground truth"),
           plt.Line2D([], [], color=B_COL, label="B (single call)"),
           plt.Line2D([], [], color=AG_COL, label="Agent-local (concat)"),
           plt.Line2D([], [], color=GQ_COL, label="Agent-Groq (concat)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(f"{OUT}/fig_decomp_examples.pdf")
fig.savefig(f"{ROOT}/results/decomp/fig_decomp_examples_preview.png", dpi=170)
print("pick:", pick, gt[pick]["instruction"])
