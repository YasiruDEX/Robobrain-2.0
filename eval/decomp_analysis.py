"""Decomposition study, final analysis (results/decomp/report.md + report.json).

Pre-registered headline: multi-stop dDFD(concat) vs B, Holm within each subset over the 2 agent conditions
(Agent-local, Agent-groq-120b). Also: last-step scores, direct Groq - local, C-pred on the same ids, plan/cost stats.
VABench-V: Groq exists for the cached-plan subset only, so every comparison is on the ids all conditions share.
"""
import json

import numpy as np
from scipy.stats import wilcoxon

from summarize import boot_ci, holm

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
R = f"{ROOT}/results"
SRC = {"vabench": (f"{R}/decomp/groq_vabench", f"{R}/decomp/local_vabench"),
       "sharerobot": (f"{R}/decomp/groq_sharerobot", f"{R}/agent")}
L, G = "Agent-local", "Agent-groq-120b"


def load(path):
    by = {}
    for l in open(path):
        r = json.loads(l)
        by.setdefault(r["cond"], {})[r["id"]] = r
    return by


def paired(a, b, ids, m="dfd"):
    """mean(a - b) over ids where both are valid; returns n, mean, CI, Wilcoxon p, per-id deltas."""
    c = [i for i in ids if a[i]["valid"] and b[i]["valid"]]
    d = np.array([a[i]["metrics"][m] - b[i]["metrics"][m] for i in c])
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else 1.0
    return {"n": len(c), "mean": float(d.mean()), "ci": boot_ci(d) if len(d) >= 2 else (np.nan, np.nan), "p": p,
            "deltas": d.tolist()}


def plan_stats(runs_path, cache=None):
    rs = [json.loads(l) for l in open(runs_path)]
    rs = [r for r in rs if not r.get("oom")]
    out = {"n": len(rs), "parsed": float(np.mean([bool(r["plan_parsed"]) for r in rs])),
           "steps": float(np.mean([r["n_steps"] for r in rs])),
           "ge2_traj": float(np.mean([r["tasks"].count("trajectory") >= 2 for r in rs])),
           "latency": float(np.mean([r["latency"] for r in rs]))}
    if cache:
        us = [json.loads(l)["usage"] for l in open(cache)]
        out["groq_tokens"] = float(np.mean([u["total_tokens"] for u in us]))
        out["groq_prompt_tokens"] = float(np.mean([u["prompt_tokens"] for u in us]))
        out["groq_completion_tokens"] = float(np.mean([u["completion_tokens"] for u in us]))
    return out


def main():
    sub_all = json.load(open(f"{ROOT}/data/subsets.json"))
    rep, lines = {}, ["# Decomposition study: final analysis", "",
                      "dDFD vs B (negative = better). Headline = concat on multi-stop; Holm over the 2 agent conditions "
                      "per subset and scoring. All comparisons on ids shared by every condition.", ""]
    for ds, (gdir, ldir) in SRC.items():
        by = load(f"{gdir}/traj.jsonl")
        by.update({k: v for k, v in load(f"{ldir}/traj.jsonl").items() if k.startswith(L)})
        conds = [L, f"{L}:concat", G, f"{G}:concat", "C-pred", "B"]
        keep = set.intersection(*(set(by[c]) for c in conds))
        rep[ds] = {}
        lines += [f"## {ds}", "", "| subset | scoring | cond | n | dDFD [95% CI] | p_holm | dFDE [95% CI] |", "|---|---|---|---|---|---|---|"]
        for s in ("multi", "single"):
            ids = sorted(i for i in keep if sub_all[ds][i] == s)
            rep[ds][s] = {"n_ids": len(ids)}
            for scoring, sfx in (("concat", ":concat"), ("last", "")):
                res = {c: paired(by[c + sfx], by["B"], ids) for c in (L, G)}
                fde = {c: paired(by[c + sfx], by["B"], ids, "fde") for c in (L, G)}
                for c, padj in zip((L, G), holm(np.array([res[c]["p"] for c in (L, G)]))):
                    res[c]["p_holm"] = float(padj)
                    r, f = res[c], fde[c]
                    lines.append(f"| {s} (n={len(ids)}) | {scoring} | {c} | {r['n']} | {r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                                 f" | {padj:.3g} | {f['mean']:+.3f} [{f['ci'][0]:+.3f},{f['ci'][1]:+.3f}] |")
                rep[ds][s][scoring] = {"vsB": res, "fde_vsB": fde,
                                       "groq_minus_local": paired(by[G + sfx], by[L + sfx], ids)}
            rep[ds][s]["cpred_vsB"] = paired(by["C-pred"], by["B"], ids)
            rep[ds][s]["arm"] = {c: paired(by[c + ":concat+arm"], by["B"], ids) for c in (L, G)}
        lines += ["", "| subset | comparison | n | dDFD [95% CI] | p |", "|---|---|---|---|---|"]
        for s in ("multi", "single"):
            for name, r in (("Groq - local (concat)", rep[ds][s]["concat"]["groq_minus_local"]),
                            ("Groq - local (last)", rep[ds][s]["last"]["groq_minus_local"]),
                            ("C-pred - B", rep[ds][s]["cpred_vsB"]),
                            ("local concat+arm - B", rep[ds][s]["arm"][L]),
                            ("Groq concat+arm - B", rep[ds][s]["arm"][G])):
                lines.append(f"| {s} | {name} | {r['n']} | {r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] | {r['p']:.3g} |")
        rep[ds]["plans"] = {"local": plan_stats(f"{ldir}/trajectory_runs.jsonl"),
                            "groq": plan_stats(f"{gdir}/trajectory_runs.jsonl",
                                               f"{R}/groq_cache/openai_gpt-oss-120b.jsonl")}
        lines += ["", "| planner | n | parsed | steps | >=2 traj steps | latency (s) | Groq tokens/plan |", "|---|---|---|---|---|---|---|"]
        for k, p in rep[ds]["plans"].items():
            lines.append(f"| {k} | {p['n']} | {p['parsed']:.2f} | {p['steps']:.2f} | {p['ge2_traj']:.2f} | {p['latency']:.1f} | "
                         f"{p.get('groq_tokens', float('nan')):.0f} |")
        lines.append("")
    json.dump(rep, open(f"{R}/decomp/report.json", "w"), indent=1)
    open(f"{R}/decomp/report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
