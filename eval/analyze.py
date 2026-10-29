"""All paper numbers from results/ -> results/report.md + results/report.json."""
import json
import os

import numpy as np

from summarize import boot_ci
from run_traj import load_jsonl

R = "/home/jayamadu/Documents/Robobrain-2.0/results"
D = "/home/jayamadu/Documents/Robobrain-2.0/data"
out, md = {}, ["# Results report", ""]


def paired(a, b, key):
    """Mean of a-b over ids valid in both, with bootstrap CI."""
    ids = [i for i in a if i in b and a[i].get("valid") and b[i].get("valid")]
    d = np.array([a[i][key] - b[i][key] for i in ids])
    return (float(d.mean()), boot_ci(d), len(ids)) if len(ids) >= 5 else (None, None, len(ids))


# 1. Trajectory: summary + error propagation (does the grounding box contain the GT endpoint?)
if os.path.exists(f"{R}/traj/summary.md"):
    md += [open(f"{R}/traj/summary.md").read(), ""]
    gt = {r["id"]: r for r in load_jsonl(f"{D}/sharerobot_traj.jsonl")}
    steps = {r["id"]: r for r in load_jsonl(f"{R}/traj/steps.jsonl")}
    tr = {}
    for r in load_jsonl(f"{R}/traj/traj.jsonl"):
        if r.get("valid"):
            r["dfd"] = r["metrics"]["dfd"]
        tr.setdefault(r["cond"], {})[r["id"]] = r
    hit, miss = {}, {}
    for i, st in steps.items():
        b, (ex, ey) = st.get("box"), gt[i]["gt_traj"][-1]
        if b is None:
            continue
        (hit if b[0] <= ex <= b[2] and b[1] <= ey <= b[3] else miss)[i] = True
    md += ["## Error propagation (C-pred minus B, DFD; negative = better)", "",
           "| grounding box contains GT endpoint | n | dDFD | 95% CI |", "|---|---|---|---|"]
    out["error_prop"] = {}
    for name, sel in [("yes", hit), ("no", miss)]:
        a = {i: v for i, v in tr.get("C-pred", {}).items() if i in sel}
        b = {i: v for i, v in tr.get("B", {}).items() if i in sel}
        m, ci, n = paired(a, b, "dfd")
        out["error_prop"][name] = {"delta": m, "ci": ci, "n": n}
        md.append(f"| {name} | {n} | {'-' if m is None else f'{m:+.3f}'} | "
                  f"{'-' if ci is None else f'[{ci[0]:+.3f},{ci[1]:+.3f}]'} |")
    goals = {r["id"]: r for r in load_jsonl(f"{R}/traj/goal_steps.jsonl")}
    if goals:  # how often does the end-location box contain the GT endpoint?
        gb = [(g["goal_box"], gt[i]["gt_traj"][-1]) for i, g in goals.items() if g.get("goal_box")]
        g_hit = sum(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b, (x, y) in gb)
        out["goal_hit"] = {"hit": g_hit, "n": len(gb), "n_total": len(goals)}
        md += [f"Goal box contains GT endpoint: {g_hit}/{len(gb)} (object box: {len(hit)}/{len(hit) + len(miss)})", ""]

    # Mechanism: is the predicted endpoint pulled toward the injected location (box centre / point)?
    def inj_xy(cond, i):
        if cond == "C-gtend":
            return gt[i]["gt_traj"][-1]
        if cond == "C-point":
            return steps[i].get("point")
        b = {"C-pred": steps[i].get("box"), "C-goal": goals.get(i, {}).get("goal_box")}.get(cond)
        return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] if b else None
    md += ["", "## Endpoint attraction: distance from predicted endpoint to the injected location", "",
           "| cond | n | dist (cond) | dist (B, same location) | delta [95% CI] |", "|---|---|---|---|---|"]
    out["attraction"] = {}
    for cond in [c for c in ["C-pred", "C-point", "C-goal", "C-gtend"] if c in tr]:
        dc, db = [], []
        for i, r in tr.get(cond, {}).items():
            loc, rb = inj_xy(cond, i), tr["B"].get(i)
            if loc is None or not r.get("valid") or not rb or not rb.get("valid"):
                continue
            dc.append(np.linalg.norm(np.subtract(r["pred"][-1], loc)))
            db.append(np.linalg.norm(np.subtract(rb["pred"][-1], loc)))
        d = np.array(dc) - np.array(db)
        out["attraction"][cond] = {"n": len(d), "cond": float(np.mean(dc)), "B": float(np.mean(db)),
                                   "delta": float(d.mean()), "ci": boot_ci(d)}
        md.append(f"| {cond} | {len(d)} | {np.mean(dc):.3f} | {np.mean(db):.3f} | {d.mean():+.3f} "
                  f"[{boot_ci(d)[0]:+.3f},{boot_ci(d)[1]:+.3f}] |")

    # Failure taxonomy for C-pred samples that got clearly worse than B (dDFD > 0.05)
    def path_len(p):
        return float(np.sum(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1)))
    tax = {"grounding box misses GT endpoint": 0, "degenerate path (len<0.05)": 0,
           "endpoint captured by box (GT end outside)": 0, "other": 0}
    worse = [i for i, r in tr.get("C-pred", {}).items()
             if r.get("valid") and tr["B"].get(i, {}).get("valid") and r["dfd"] - tr["B"][i]["dfd"] > 0.05]
    for i in worse:
        b, p, (ex, ey) = steps[i].get("box"), tr["C-pred"][i]["pred"], gt[i]["gt_traj"][-1]
        inside = lambda x, y: b[0] <= x <= b[2] and b[1] <= y <= b[3]
        if path_len(p) < 0.05:
            tax["degenerate path (len<0.05)"] += 1
        elif inside(*p[-1]) and not inside(ex, ey):
            tax["endpoint captured by box (GT end outside)"] += 1
        elif not inside(ex, ey):
            tax["grounding box misses GT endpoint"] += 1
        else:
            tax["other"] += 1
    out["taxonomy"] = {"n_worse": len(worse), **tax}
    md += ["", f"## Failure taxonomy: C-pred samples with dDFD > 0.05 vs B (n={len(worse)}, first matching rule)",
           ""] + [f"- {k}: {v}" for k, v in tax.items()]

    lat = {c: np.mean([r["latency"] for r in v.values() if "latency" in r]) for c, v in tr.items()}
    step_lat = np.mean([s["lat_plan"] + s["lat_ground"] for s in steps.values()])
    out["latency"] = {"B": lat.get("B"), "C-pred_total": lat.get("C-pred", 0) + step_lat}
    md += ["", f"Latency: B {lat.get('B', 0):.2f} s/sample; C-pred pipeline (plan+ground+traj) "
               f"{out['latency']['C-pred_total']:.2f} s/sample", ""]

# 2. Spatial tasks
sp = f"{R}/spatial"
if os.path.exists(f"{sp}/affordance.jsonl"):
    af = {}
    for r in load_jsonl(f"{sp}/affordance.jsonl"):
        af.setdefault(r["cond"], {})[r["id"]] = r
    md += ["## Affordance (ShareRobot-Bench, IoU)", "",
           "| cond | n | valid | mIoU | IoU>=0.5 | dIoU vs B [95% CI] |", "|---|---|---|---|---|---|"]
    out["affordance"] = {}
    for c, v in af.items():
        ious = np.array([r["iou"] for r in v.values()])
        valid = np.mean([r["valid"] for r in v.values()])
        ids = [i for i in v if i in af["B"]]
        d = np.array([v[i]["iou"] - af["B"][i]["iou"] for i in ids])  # invalid counts as IoU 0
        dc = "-" if c == "B" else f"{d.mean():+.3f} [{boot_ci(d)[0]:+.3f},{boot_ci(d)[1]:+.3f}]"
        out["affordance"][c] = {"n": len(v), "valid": valid, "miou": ious.mean(), "acc50": (ious >= .5).mean()}
        md.append(f"| {c} | {len(v)} | {valid:.2f} | {ious.mean():.3f} | {(ious >= .5).mean():.3f} | {dc} |")
    # exact McNemar on IoU>=0.5, C-pred vs B
    from scipy.stats import binomtest
    hb = {i: af["B"][i]["iou"] >= .5 for i in af["B"]}
    hc = {i: af["C-pred"][i]["iou"] >= .5 for i in af["C-pred"] if i in hb}
    up, down = sum(hc[i] and not hb[i] for i in hc), sum(hb[i] and not hc[i] for i in hc)
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    out["affordance"]["mcnemar_acc50"] = {"B_only": down, "Cpred_only": up, "p": p}
    md += ["", f"McNemar IoU>=0.5 (C-pred vs B): C-pred-only {up}, B-only {down}, exact p={p:.3g}"]
    md.append("")
if os.path.exists(f"{sp}/grounding.jsonl"):
    g = load_jsonl(f"{sp}/grounding.jsonl")
    ious = np.array([r["iou"] for r in g])
    out["grounding"] = {"n": len(g), "miou": ious.mean(), "acc50": (ious >= .5).mean(),
                        "valid": np.mean([r["valid"] for r in g])}
    md += [f"## Grounding (RoboRefIt subset): n={len(g)}, mIoU {ious.mean():.3f}, "
           f"Acc@0.5 {(ious >= .5).mean():.3f}, valid {out['grounding']['valid']:.2f}", ""]
if os.path.exists(f"{sp}/pointing.jsonl"):
    p = load_jsonl(f"{sp}/pointing.jsonl")
    sc = {k: 100 * np.mean([r["score"] for r in p if k == "all" or r["category"] == k]) for k in
          ["seen", "unseen", "all"]}
    out["pointing"] = sc
    md += [f"## Pointing (Where2Place): seen {sc['seen']:.2f}, unseen {sc['unseen']:.2f}, all {sc['all']:.2f}", ""]

# 3. Planner
if os.path.exists(f"{R}/planner/planner.jsonl"):
    pl = load_jsonl(f"{R}/planner/planner.jsonl")
    classes = ["grounding", "affordance", "trajectory", "pointing"]
    md += ["## Task routing (labels = benchmark provenance)", "",
           "| router | n | acc | acc (traj+afford merged) | " + " | ".join(classes) + " | latency (s) |",
           "|---" * (len(classes) + 5) + "|"]
    merge = lambda c: "manipulation" if c in ("trajectory", "affordance") else c
    out["planner"] = {}
    for router in sorted({r["router"] for r in pl}):
        rs = [r for r in pl if r["router"] == router]
        per = {c: np.mean([r["pred"] == c for r in rs if r["label"] == c]) for c in classes}
        acc = np.mean([r["pred"] == r["label"] for r in rs])
        conf = {c: {k: sum(r["pred"] == k for r in rs if r["label"] == c) for k in classes + ["general"]}
                for c in classes}
        acc_m = np.mean([merge(r["pred"]) == merge(r["label"]) for r in rs])
        out["planner"][router] = {"acc": acc, "acc_merged": acc_m, "per_class": per, "confusion": conf,
                                  "latency": np.mean([r["latency"] for r in rs])}
        md.append(f"| {router} | {len(rs)} | {acc:.3f} | {acc_m:.3f} | " + " | ".join(f"{per[c]:.3f}" for c in classes)
                  + f" | {out['planner'][router]['latency']:.3f} |")
    md.append("")

# 4. Generality: where does the object box sit on the GT path? (start vs end), per dataset and model
from scipy.stats import binomtest


def mcnemar(a, b):
    """Exact McNemar on two {id: bool} dicts: (a-only, b-only, p)."""
    ids = [i for i in a if i in b]
    up, down = sum(a[i] and not b[i] for i in ids), sum(b[i] and not a[i] for i in ids)
    return up, down, (binomtest(up, up + down, 0.5).pvalue if up + down else 1.0)


out["anchor"] = {}
md += ["## Where the object box lies on the GT path, and C-pred vs B start/end error", "",
       "| run | n | box contains GT start | box contains GT end | start err B | start err C-pred | "
       "dStart [95% CI] | dDFD C-pred |", "|---|---|---|---|---|---|---|---|"]
for name, data, res in [("ShareRobot / RoboBrain", "sharerobot_traj", "traj"),
                        ("VABench-V / RoboBrain", "vabench_v", "vabench"),
                        ("ShareRobot / Qwen base", "sharerobot_traj", "traj_qwen")]:
    if not os.path.exists(f"{R}/{res}/summary.json"):
        continue
    gt = {r["id"]: r["gt_traj"] for r in load_jsonl(f"{D}/{data}.jsonl")}
    st = {r["id"]: r for r in load_jsonl(f"{R}/{res}/steps.jsonl")}
    tr = {}
    for r in load_jsonl(f"{R}/{res}/traj.jsonl"):
        tr.setdefault(r["cond"], {})[r["id"]] = r
    ins = lambda b, p: b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
    boxed = {i: s["box"] for i, s in st.items() if s.get("box")}
    s_in = sum(ins(b, gt[i][0]) for i, b in boxed.items())
    e_in = sum(ins(b, gt[i][-1]) for i, b in boxed.items())
    ids = [i for i in boxed if tr["B"].get(i, {}).get("valid") and tr["C-pred"].get(i, {}).get("valid")]
    eb = np.array([np.linalg.norm(np.subtract(tr["B"][i]["pred"][0], gt[i][0])) for i in ids])
    ec = np.array([np.linalg.norm(np.subtract(tr["C-pred"][i]["pred"][0], gt[i][0])) for i in ids])
    sm = json.load(open(f"{R}/{res}/summary.json"))
    d_dfd = sm["C-pred"].get("dfd_delta")
    ci = boot_ci(ec - eb)
    out["anchor"][res] = {"n": len(boxed), "start_in": s_in, "end_in": e_in, "n_paired": len(ids),
                          "start_err_B": eb.mean(), "start_err_Cpred": ec.mean(), "d_start": (ec - eb).mean(),
                          "d_start_ci": ci, "summary": sm}
    md.append(f"| {name} | {len(boxed)} | {s_in} ({s_in / len(boxed):.0%}) | {e_in} ({e_in / len(boxed):.0%}) | "
              f"{eb.mean():.3f} | {ec.mean():.3f} | {(ec - eb).mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | "
              f"{'-' if d_dfd is None else f'{d_dfd:+.3f}'} |")
md.append("")

# Base Qwen: many invalid outputs, so report validity changes next to the paired deltas
if os.path.exists(f"{R}/traj_qwen/traj.jsonl"):
    tq = {}
    for r in load_jsonl(f"{R}/traj_qwen/traj.jsonl"):
        tq.setdefault(r["cond"], {})[r["id"]] = bool(r["valid"])
    md += ["## Qwen base: valid-rate change vs B (exact McNemar; invalid = off-frame or <2 points)", "",
           "| cond | valid | valid-only-cond | valid-only-B | p |", "|---|---|---|---|---|"]
    out["qwen_valid"] = {}
    for c, v in tq.items():
        if c == "B":
            continue
        up, down, p = mcnemar(v, tq["B"])
        out["qwen_valid"][c] = {"valid": np.mean(list(v.values())), "cond_only": up, "B_only": down, "p": p}
        md.append(f"| {c} | {np.mean(list(v.values())):.2f} | {up} | {down} | {p:.3g} |")
    md.append("")

# 5. Agent-local: original app layer (multi-turn decomposer) with the local RoboBrain decoder
ag = f"{R}/agent"
if os.path.exists(f"{ag}/affordance_runs.jsonl"):
    out["agent"] = {}
    md += ["## Agent-local (original app layer, local decoder)", ""]
    for task in ["trajectory", "affordance"]:
        rs = load_jsonl(f"{ag}/{task}_runs.jsonl")
        ok = [r for r in rs if not r.get("oom")]
        st = {"n": len(rs), "oom": len(rs) - len(ok), "parsed": sum(bool(r["plan_parsed"]) for r in ok),
              "repaired": sum(bool(r["plan_repaired"]) for r in ok), "valid": sum(bool(r["valid"]) for r in rs),
              "steps_mean": np.mean([r["n_steps"] for r in ok]), "steps_max": max(r["n_steps"] for r in ok),
              "latency": np.mean([r["latency"] for r in ok]),
              "has_task": {t: sum(t in r["tasks"] for r in ok) for t in ["grounding", "affordance", "trajectory", "general"]}}
        out["agent"][task] = st
        md.append(f"- {task}: n={st['n']}, OOM {st['oom']}, plan parsed {st['parsed']} (repaired {st['repaired']}), "
                  f"valid {st['valid']}, steps mean {st['steps_mean']:.1f} max {st['steps_max']}, "
                  f"latency {st['latency']:.1f} s, plans containing task {st['has_task']}")
    if os.path.exists(f"{ag}/summary.json"):
        sm = json.load(open(f"{ag}/summary.json"))
        out["agent"]["traj_summary"] = sm
    af_ag = {r["id"]: r for r in load_jsonl(f"{ag}/affordance_runs.jsonl")}
    if "affordance" in out:
        ious = np.array([r.get("iou", 0.0) for r in af_ag.values()])
        ids = [i for i in af_ag if i in af["B"]]
        d = np.array([af_ag[i].get("iou", 0.0) - af["B"][i]["iou"] for i in ids])
        up, down, p = mcnemar({i: af_ag[i].get("iou", 0.0) >= .5 for i in ids}, {i: af["B"][i]["iou"] >= .5 for i in ids})
        up2, down2, p2 = mcnemar({i: af_ag[i].get("iou", 0.0) >= .5 for i in ids},
                                 {i: af["C-pred"][i]["iou"] >= .5 for i in ids})
        out["agent"]["affordance_scores"] = {"miou": ious.mean(), "acc50": (ious >= .5).mean(), "d_miou_B": d.mean(),
                                             "d_miou_ci": boot_ci(d), "vsB": [up, down, p], "vsCpred": [up2, down2, p2]}
        md += ["", f"Agent-local affordance: mIoU {ious.mean():.3f}, IoU>=0.5 {(ious >= .5).mean():.3f}, "
                   f"dmIoU vs B {d.mean():+.3f} [{boot_ci(d)[0]:+.3f},{boot_ci(d)[1]:+.3f}]; "
                   f"McNemar IoU>=0.5 vs B: agent-only {up}, B-only {down}, p={p:.3g}; "
                   f"vs C-pred: agent-only {up2}, C-pred-only {down2}, p={p2:.3g}", ""]

json.dump(out, open(f"{R}/report.json", "w"), indent=1, default=float)
open(f"{R}/report.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
