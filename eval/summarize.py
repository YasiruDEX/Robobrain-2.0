"""Summarise traj.jsonl: valid rate, means, paired deltas vs B (bootstrap CI + Wilcoxon), Holm."""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["dfd", "hd", "rmse_idx", "rmse", "ade", "fde"]


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="B")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(f"{a.out}/traj.jsonl")]
    by = {}
    for r in recs:
        by.setdefault(r["cond"], {})[r["id"]] = r
    ref = by[a.ref]
    summary = {}
    for cond, rs in by.items():
        n = len(rs)
        valid = [r for r in rs.values() if r["valid"]]
        s = {"n": n, "valid_rate": len(valid) / n,
             "latency_mean": float(np.mean([r["latency"] for r in rs.values() if "latency" in r] or [np.nan]))}
        for k in METRICS:
            s[k] = float(np.mean([r["metrics"][k] for r in valid])) if valid else None
        if cond != a.ref:
            common = [i for i in rs if rs[i]["valid"] and i in ref and ref[i]["valid"]]
            s["n_paired"] = len(common)
            for k in METRICS:
                if len(common) < 5:
                    continue
                d = np.array([rs[i]["metrics"][k] - ref[i]["metrics"][k] for i in common])
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                s[f"{k}_delta"] = float(d.mean())
                s[f"{k}_ci"] = boot_ci(d)
                s[f"{k}_p"] = p
        summary[cond] = s
    others = [c for c in summary if c != a.ref and "dfd_p" in summary[c]]
    for k in METRICS:
        ps = [summary[c][f"{k}_p"] for c in others]
        if ps:
            for c, padj in zip(others, holm(np.array(ps))):
                summary[c][f"{k}_p_holm"] = float(padj)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)

    lines = [f"# Trajectory summary ({a.out})", "",
             f"Paired deltas vs {a.ref} (negative = better); Holm over {len(others)} conditions per metric.", "",
             "| cond | n | valid | DFD | HD | RMSE_idx | RMSE | ADE | FDE | dDFD [95% CI] p_holm | dRMSE_idx [95% CI] p_holm | dFDE [95% CI] p_holm | lat (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    for c, s in summary.items():
        def dcol(k):
            if f"{k}_delta" not in s:
                return "-"
            lo, hi = s[f"{k}_ci"]
            return f"{s[f'{k}_delta']:+.3f} [{lo:+.3f},{hi:+.3f}] p={s.get(f'{k}_p_holm', s[f'{k}_p']):.3g}"
        lines.append(f"| {c} | {s['n']} | {s['valid_rate']:.2f} | " + " | ".join(f(s[k]) for k in METRICS)
                     + f" | {dcol('dfd')} | {dcol('rmse_idx')} | {dcol('fde')} | {s['latency_mean']:.2f} |")
    open(f"{a.out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
