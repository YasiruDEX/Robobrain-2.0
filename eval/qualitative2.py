"""Second qualitative figure (rule-based picks) -> paper/figs/fig_examples.pdf.

Selection rules:
  (a) VABench-V, object box contains the GT start point: sample with the median C-pred minus B DFD
  (b)+(c) Agent-local trajectory: sample with the median Agent-local minus B DFD; (c) prints its plan
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qualitative import B_COL, BOX_COL, C_COL, INK, box, inside, median_pick, path, show
from run_traj import load_jsonl

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
R, D = f"{ROOT}/results", f"{ROOT}/data"
AG_COL = "#7a5af5"


def quote(t):
    return f"\"{textwrap.shorten(t.strip(chr(34) + ' '), 40)}\""

# (a) VABench-V
vgt = {r["id"]: r for r in load_jsonl(f"{D}/vabench_v.jsonl")}
vst = {r["id"]: r for r in load_jsonl(f"{R}/vabench/steps.jsonl")}
vtr = {}
for r in load_jsonl(f"{R}/vabench/traj.jsonl"):
    vtr.setdefault(r["cond"], {})[r["id"]] = r
cands = []
for i, st in vst.items():
    b, c = vtr["B"].get(i), vtr["C-pred"].get(i)
    if st.get("box") and b and c and b["valid"] and c["valid"] and inside(st["box"], vgt[i]["gt_traj"][0]):
        cands.append((c["metrics"]["dfd"] - b["metrics"]["dfd"], i))
va = median_pick(cands)

# (b) Agent-local
gt = {r["id"]: r for r in load_jsonl(f"{D}/sharerobot_traj.jsonl")}
tb = {r["id"]: r for r in load_jsonl(f"{R}/traj/traj.jsonl") if r["cond"] == "B"}
runs = {r["id"]: r for r in load_jsonl(f"{R}/agent/trajectory_runs.jsonl")}
ag = median_pick([(r["metrics"]["dfd"] - tb[i]["metrics"]["dfd"], i) for i, r in runs.items()
                  if r["valid"] and tb.get(i, {}).get("valid")])

fig = plt.figure(figsize=(7.0, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
ax = fig.add_subplot(gs[0])
wh = show(ax, vgt[va]["image"])
box(ax, vst[va]["box"], wh, BOX_COL)
path(ax, vgt[va]["gt_traj"], wh, INK, "--")
path(ax, vtr["B"][va]["pred"], wh, B_COL)
path(ax, vtr["C-pred"][va]["pred"], wh, C_COL)
db, dc = vtr["B"][va]["metrics"]["dfd"], vtr["C-pred"][va]["metrics"]["dfd"]
ax.set_title(f"(a) VABench-V: box on the path start\n{quote(vgt[va]['instruction'])}\nDFD {db:.2f} → {dc:.2f}",
             fontsize=7, color=INK)

ax = fig.add_subplot(gs[1])
wh = show(ax, gt[ag]["image"])
path(ax, gt[ag]["gt_traj"], wh, INK, "--")
path(ax, tb[ag]["pred"], wh, B_COL)
path(ax, runs[ag]["pred"], wh, AG_COL)
db, da = tb[ag]["metrics"]["dfd"], runs[ag]["metrics"]["dfd"]
ax.set_title(f"(b) Agent-local, ShareRobot\n{quote(gt[ag]['instruction'])}\nDFD {db:.2f} → {da:.2f}",
             fontsize=7, color=INK)

ax = fig.add_subplot(gs[2])
ax.axis("off")
lines = [f"(c) Agent-local plan for (b), {len(runs[ag]['steps'])} calls, {runs[ag]['latency']:.1f} s"]
for k, st in enumerate(runs[ag]["steps"], 1):
    lines.append(textwrap.shorten(f"{k}. [{st['task']}] {st['prompt']}", 150))
    lines.append(textwrap.shorten(f"    → {st['raw']}", 70))
txt = "\n".join("\n".join(textwrap.wrap(l, 46, subsequent_indent="      ")) if not l.startswith("(c)") else l
                for l in lines)
ax.text(0, 1, txt, va="top", ha="left", fontsize=5.4, family="monospace", color=INK, transform=ax.transAxes)

handles = [plt.Line2D([], [], color=INK, ls="--", label="ground truth"),
           plt.Line2D([], [], color=B_COL, label="B (single-step)"),
           plt.Line2D([], [], color=C_COL, label="C-pred (box injected)"),
           plt.Line2D([], [], color=AG_COL, label="Agent-local"),
           plt.Line2D([], [], color=BOX_COL, label="injected grounding box")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{ROOT}/paper/figs/fig_examples.pdf")
fig.savefig(f"{ROOT}/results/fig_examples_preview.png", dpi=170)
print("picks: vabench", va, "agent", ag)
