"""Qualitative figure with rule-based (not hand-picked) examples -> paper/figs/fig_qualitative.pdf.

Selection rules:
  (a) trajectory, grounding box misses GT endpoint: sample with the median C-pred minus B DFD
  (b) trajectory, grounding box contains GT endpoint: sample with the median C-pred minus B DFD
  (c) affordance: among samples reaching IoU>=0.5 only with C-pred, the one with median IoU gain
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from run_traj import load_jsonl

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
R, D = f"{ROOT}/results", f"{ROOT}/data"
INK, B_COL, C_COL, BOX_COL = "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})


def median_pick(cands):
    cands = sorted(cands, key=lambda t: t[0])
    return cands[len(cands) // 2][1]


gt = {r["id"]: r for r in load_jsonl(f"{D}/sharerobot_traj.jsonl")}
steps = {r["id"]: r for r in load_jsonl(f"{R}/traj/steps.jsonl")}
tr = {}
for r in load_jsonl(f"{R}/traj/traj.jsonl"):
    tr.setdefault(r["cond"], {})[r["id"]] = r


def inside(b, p):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


groups = {True: [], False: []}
for i, st in steps.items():
    b, c = tr["B"].get(i), tr["C-pred"].get(i)
    if st.get("box") and b and c and b["valid"] and c["valid"]:
        d = c["metrics"]["dfd"] - b["metrics"]["dfd"]
        groups[inside(st["box"], gt[i]["gt_traj"][-1])].append((d, i))
picks = [("(a) box misses GT endpoint", median_pick(groups[False])),
         ("(b) box contains GT endpoint", median_pick(groups[True]))]

af_gt = {r["id"]: r for r in load_jsonl(f"{D}/sharerobot_afford.jsonl")}
af = {}
for r in load_jsonl(f"{R}/spatial/affordance.jsonl"):
    af.setdefault(r["cond"], {})[r["id"]] = r
af_steps = {r["id"]: r for r in load_jsonl(f"{R}/spatial/affordance_steps.jsonl")}
gain = [(af["C-pred"][i]["iou"] - af["B"][i]["iou"], i) for i in af["C-pred"]
        if af["C-pred"][i]["iou"] >= .5 and af["B"].get(i, {}).get("iou", 0) < .5]
af_pick = median_pick(gain)


def show(ax, path):
    im = Image.open(path).convert("RGB")
    ax.imshow(im)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d9d8d4")
    return im.size


def box(ax, b, wh, color, ls="-", lw=1.5):
    w, h = wh
    ax.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h,
                           fill=False, ec=color, lw=lw, ls=ls))


def path(ax, pts, wh, color, ls="-", label=None):
    w, h = wh
    p = np.asarray(pts) * [w, h]
    ax.plot(p[:, 0], p[:, 1], ls, color=color, lw=1.6, label=label)
    ax.plot(p[-1, 0], p[-1, 1], "o", color=color, ms=4.5, mec="white", mew=0.8)


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
for ax, (title, i) in zip(axes[:2], picks):
    wh = show(ax, gt[i]["image"])
    box(ax, steps[i]["box"], wh, BOX_COL)
    path(ax, gt[i]["gt_traj"], wh, INK, "--", "ground truth")
    path(ax, tr["B"][i]["pred"], wh, B_COL, "-", "B (single-step)")
    path(ax, tr["C-pred"][i]["pred"], wh, C_COL, "-", "C-pred (box injected)")
    db, dc = tr["B"][i]["metrics"]["dfd"], tr["C-pred"][i]["metrics"]["dfd"]
    ax.set_title(f"{title}\n\"{gt[i]['instruction'].strip(chr(34))}\"  DFD {db:.2f} → {dc:.2f}", fontsize=7, color=INK)

ax, i = axes[2], af_pick
wh = show(ax, af_gt[i]["image"])
box(ax, af_steps[i]["box"], wh, BOX_COL)
box(ax, af_gt[i]["gt_box"], wh, INK, "--")
box(ax, af["B"][i]["box"], wh, B_COL) if af["B"][i].get("box") else None
box(ax, af["C-pred"][i]["box"], wh, C_COL)
ax.set_title(f"(c) affordance\n\"{af_gt[i]['instruction']}\"  IoU {af['B'][i]['iou']:.2f} → "
             f"{af['C-pred'][i]['iou']:.2f}", fontsize=7, color=INK)

handles = [plt.Line2D([], [], color=INK, ls="--", label="ground truth"),
           plt.Line2D([], [], color=B_COL, label="B (single-step)"),
           plt.Line2D([], [], color=C_COL, label="C-pred (grounding injected)"),
           plt.Line2D([], [], color=BOX_COL, label="injected grounding box")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{ROOT}/paper/figs/fig_qualitative.pdf")
print("picks:", picks, "afford:", af_pick)
