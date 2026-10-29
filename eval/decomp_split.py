"""Split a traj.jsonl (B + other conditions) by the pre-registered multi/single subset (data/subsets.json).
Paired deltas vs B per subset: mean, bootstrap 95% CI, Wilcoxon p, Holm within each subset."""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from summarize import boot_ci, holm

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", choices=["sharerobot", "vabench"], required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", default=["dfd", "fde"])
    ap.add_argument("--common", action="store_true", help="restrict to ids present in every listed condition")
    a = ap.parse_args()
    sub = json.load(open(f"{ROOT}/data/subsets.json"))[a.dataset]
    by = {}
    for l in open(a.traj):
        r = json.loads(l)
        by.setdefault(r["cond"], {})[r["id"]] = r
    ref = by["B"]
    if a.common:
        keep = set.intersection(*(set(by[c]) for c in a.conds))
        sub = {i: v for i, v in sub.items() if i in keep}
    lines = [f"| subset | cond | n_paired | valid | " + " | ".join(f"d{m.upper()} [95% CI] p_holm" for m in a.metrics) + " |",
             "|---|---|---|---|" + "---|" * len(a.metrics)]
    for s in ("multi", "single"):
        ids = [i for i, v in sub.items() if v == s and i in ref]
        rows, ps = [], {m: [] for m in a.metrics}
        for c in a.conds:
            rs = by[c]
            common = [i for i in ids if i in rs and rs[i]["valid"] and ref[i]["valid"]]
            valid = np.mean([rs[i]["valid"] for i in ids if i in rs])
            cell = {}
            for m in a.metrics:
                d = np.array([rs[i]["metrics"][m] - ref[i]["metrics"][m] for i in common])
                p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else 1.0
                cell[m] = (d.mean(), boot_ci(d) if len(d) >= 2 else (np.nan, np.nan))
                ps[m].append(p)
            rows.append((c, len(common), valid, cell))
        adj = {m: holm(np.array(ps[m])) for m in a.metrics}
        for k, (c, n, valid, cell) in enumerate(rows):
            cols = [f"{cell[m][0]:+.3f} [{cell[m][1][0]:+.3f},{cell[m][1][1]:+.3f}] p={adj[m][k]:.3g}" for m in a.metrics]
            lines.append(f"| {s} (n={len(ids)}) | {c} | {n} | {valid:.2f} | " + " | ".join(cols) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
