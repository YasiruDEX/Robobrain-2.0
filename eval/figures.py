"""Paper figures from results/*.json -> paper/figs/*.pdf (light surface, print)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/jayamadu/Documents/Robobrain-2.0/results"
OUT = "/home/jayamadu/Documents/Robobrain-2.0/paper/figs"
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, MARK = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
LABELS = {"C-text": "C-text (name only)", "C-pred": "C-pred (own box)", "C-point": "C-point (own point)",
          "C-shuffle": "C-shuffle (other image)", "C-goal": "C-goal (end-location box)",
          "C-gtend": "C-gtend (oracle endpoint)"}
ORDER = ["C-text", "C-pred", "C-point", "C-goal", "C-shuffle", "C-gtend"]


def forest(ax, rows, title, xlabel):
    """rows: list of (label, delta, (lo, hi)); top-to-bottom order."""
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for yi, (lab, d, (lo, hi)) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=MARK, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(d, yi, "o", ms=5, color=MARK, mec="white", mew=1.2, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    ax.set_xlabel(xlabel)
    lo_all = min(r[2][0] for r in rows + [("", 0, (0, 0))])
    hi_all = max(r[2][1] for r in rows + [("", 0, (0, 0))])
    pad = 0.08 * (hi_all - lo_all)
    ax.set_xlim(lo_all - pad, hi_all + pad)


def traj_rows(path, metric="dfd"):
    s = json.load(open(path))
    return [(LABELS[c], s[c][f"{metric}_delta"], s[c][f"{metric}_ci"]) for c in ORDER
            if c in s and f"{metric}_delta" in s[c]]


# Fig 1: paired dDFD vs B per condition, one panel per (model, benchmark) that exists
panels = [(f"{R}/traj/summary.json", "RoboBrain2.0-3B\nShareRobot-Bench"),
          (f"{R}/vabench/summary.json", "RoboBrain2.0-3B\nVABench-V"),
          (f"{R}/traj_qwen/summary.json", "Qwen2.5-VL-3B (base)\nShareRobot-Bench")]
panels = [p for p in panels if os.path.exists(p[0])]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.2), sharey=True, squeeze=False)
for ax, (p, t) in zip(axes[0], panels):
    forest(ax, traj_rows(p), t, r"$\Delta$DFD vs. B  ($\leftarrow$ better)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_traj_deltas.pdf")
print("wrote fig_traj_deltas.pdf with", len(panels), "panels")

# Fig 2: mechanism. (a) endpoint attraction, (b) error propagation
rep = json.load(open(f"{R}/report.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.1))
att = [(LABELS[c], v["delta"], v["ci"]) for c, v in rep["attraction"].items()]
forest(a1, att, "(a) Endpoint attraction to\ninjected location", r"$\Delta$ endpoint distance vs. B")
ep = rep["error_prop"]
forest(a2, [(f"box contains GT end (n={ep['yes']['n']})", ep["yes"]["delta"], ep["yes"]["ci"]),
            (f"box misses GT end (n={ep['no']['n']})", ep["no"]["delta"], ep["no"]["ci"])],
       "(b) C-pred effect by\ngrounding correctness", r"$\Delta$DFD vs. B  ($\leftarrow$ better)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_mechanism.pdf")
print("wrote fig_mechanism.pdf")

# Fig 3: planner confusion (row-normalised), one small multiple per router
classes = ["grounding", "affordance", "trajectory", "pointing"]
cols = classes + ["general"]
ABBR = {"grounding": "Gnd", "affordance": "Aff", "trajectory": "Traj", "pointing": "Pnt", "general": "Gen"}
names = {"keyword": "Keyword rules", "rb2": "RoboBrain-3B zero-shot", "rb2fs": "RoboBrain-3B few-shot",
         "groq": "Llama-3.3-70B (Groq)"}
routers = [r for r in ["keyword", "rb2", "rb2fs", "groq"] if r in rep["planner"]]
fig, axes = plt.subplots(1, len(routers), figsize=(2.3 * len(routers), 2.5), squeeze=False)
for ax, r in zip(axes[0], routers):
    conf = rep["planner"][r]["confusion"]
    M = np.array([[conf[t][p] for p in cols] for t in classes], float)
    M = M / M.sum(1, keepdims=True)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [ABBR[c] for c in cols])
    ax.set_yticks(range(len(classes)), [ABBR[c] for c in classes] if r == routers[0] else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{names[r]}\nacc {100 * rep['planner'][r]['acc']:.1f}%", fontsize=8, color=INK)
axes[0][0].set_ylabel("true task (benchmark)")
fig.supxlabel("predicted task", fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_planner.pdf")
print("wrote fig_planner.pdf")
