"""Grounding-conditioned trajectory experiment (resumable).

Input: JSONL with {id, image, instruction, gt_traj:[[x,y],...] in [0,1], target?, gt_box?}.
Pass 1 (per sample): target extraction (local planner), grounding, pointing.
Pass 2 (per sample x condition): trajectory with the condition's context.
Outputs append to <out>/steps.jsonl and <out>/traj.jsonl; finished keys are skipped on restart.
"""
import argparse
import json
import os
import sys
import traceback

from parse import parse_box, parse_points, valid_coords
from metrics import traj_metrics

CONDS = ["B", "C-text", "C-pred", "C-point", "C-shuffle", "C-gtend", "C-goal"]
PLANNER_Q = ('Instruction: "{}". What is the single target object the robot must reach or manipulate? '
             'Answer with the object name only.')
GOAL_Q = ('Instruction: "{}". Where should the robot end effector be at the end of this task? '
          'Answer with a short object or place name only.')


def clean_target(raw):
    """Planner answer -> plain object phrase (drops markdown escapes, quotes, trailing period)."""
    t = raw.replace("\\_", " ").replace("_", " ").replace("*", "").strip().splitlines()[0]
    return t.strip(" .\"'").lower()[:60]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def to_px(xy, scale):
    return [round(v * s) for v, s in zip(xy, scale * (len(xy) // 2))]


def context_for(cond, s, step, shuffle_step):
    obj = step["target"]
    sc = step["scale"]
    if cond == "B":
        return ""
    if cond == "C-text":
        return f'The target object "{obj}" is located in the image.'
    if cond == "C-pred":
        b = step.get("box")
        return f'The target object "{obj}" is located at bounding box {to_px(b, sc)} in the image.' if b else None
    if cond == "C-point":
        p = step.get("point")
        return f'The target object "{obj}" is located at point {to_px(p, sc)} in the image.' if p else None
    if cond == "C-shuffle":
        b = shuffle_step.get("box") if shuffle_step else None
        return f'The target object "{obj}" is located at bounding box {to_px(b, sc)} in the image.' if b else None
    if cond == "C-gtend":
        return f'The target object "{obj}" is located at point {to_px(s["gt_traj"][-1], sc)} in the image.'
    if cond == "C-goal":  # ground the motion's end location instead of the manipulated object
        b = step.get("goal_box")
        return (f'The motion should end at the "{step["goal"]}", located at bounding box {to_px(b, sc)} '
                f'in the image.') if b else None
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/jayamadu/Documents/Robobrain-2.0/weights/RoboBrain2.0-3B")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    samples = load_jsonl(a.data)
    if a.limit:
        samples = samples[: a.limit]
    conds = a.conds.split(",")
    steps_path, traj_path = f"{a.out}/steps.jsonl", f"{a.out}/traj.jsonl"
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    print(f"=== START n={len(samples)} conds={conds}", flush=True)

    from rb2 import RB2
    m = RB2(a.model, quant=a.quant, max_pixels=a.max_pixels)

    # Pass 1: planner target extraction + grounding + pointing
    steps = {r["id"]: r for r in load_jsonl(steps_path)}
    for s in samples:
        if s["id"] in steps:
            continue
        try:
            if s.get("target"):
                target, plan = s["target"], None
            else:
                plan = m.infer(None, PLANNER_Q.format(s["instruction"]), "general")
                target = clean_target(plan["raw"])
            g = m.infer(s["image"], target, "grounding")
            p = m.infer(s["image"], f"Identify a point on the {target}", "pointing")
            pts = parse_points(p["raw"], p["scale"])
            rec = {"id": s["id"], "target": target, "scale": g["scale"],
                   "box": parse_box(g["raw"], g["scale"]), "point": pts[0] if pts else None,
                   "raw_plan": plan["raw"] if plan else None, "raw_ground": g["raw"], "raw_point": p["raw"],
                   "lat_plan": plan["latency"] if plan else 0.0, "lat_ground": g["latency"],
                   "lat_point": p["latency"]}
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED step {s['id']}", flush=True)
            sys.exit(1)
        append(steps_path, rec)
        steps[s["id"]] = rec
    print("] done (pass1)", flush=True)

    # Pass 1b (only for C-goal): ask for the motion's end location, ground it. Kept in its own file
    # so earlier steps.jsonl files stay untouched.
    if "C-goal" in conds:
        goal_path = f"{a.out}/goal_steps.jsonl"
        goals = {r["id"]: r for r in load_jsonl(goal_path)}
        for s in samples:
            if s["id"] not in goals:
                q = m.infer(None, GOAL_Q.format(s["instruction"]), "general")
                goal = clean_target(q["raw"])
                g = m.infer(s["image"], goal, "grounding")
                goals[s["id"]] = {"id": s["id"], "goal": goal, "goal_box": parse_box(g["raw"], g["scale"]),
                                  "raw_goal": q["raw"], "raw_goal_ground": g["raw"],
                                  "lat_goal": q["latency"] + g["latency"]}
                append(goal_path, goals[s["id"]])
            steps[s["id"]].update(goals[s["id"]])
        print("] done (pass1b goal)", flush=True)

    # Pass 2: trajectory conditions. Shuffle partner is a fixed derangement (i -> i + n//2).
    ids = [s["id"] for s in samples]
    partner = {ids[i]: ids[(i + max(1, len(ids) // 2)) % len(ids)] for i in range(len(ids))}
    done = {(r["id"], r["cond"]) for r in load_jsonl(traj_path)}
    for cond in conds:
        for s in samples:
            if (s["id"], cond) in done:
                continue
            st = steps[s["id"]]
            ctx = context_for(cond, s, st, steps.get(partner[s["id"]]))
            rec = {"id": s["id"], "cond": cond, "context": ctx}
            if ctx is None:  # upstream step produced no coordinates -> condition not runnable
                rec.update(valid=False, reason="no_prior")
            else:
                try:
                    r = m.infer(s["image"], s["instruction"], "trajectory", ctx, s.get("robot"))
                except Exception:
                    traceback.print_exc()
                    print(f"=== FAILED traj {s['id']} {cond}", flush=True)
                    sys.exit(1)
                pts = parse_points(r["raw"], r["scale"])
                ok = len(pts) >= 2 and valid_coords(pts)
                rec.update(raw=r["raw"], pred=pts, latency=r["latency"], valid=ok,
                           reason=None if ok else "parse_or_range")
                if ok:
                    rec["metrics"] = traj_metrics(pts, s["gt_traj"])
            append(traj_path, rec)
        print(f"] done ({cond})", flush=True)
    print("=== RUNNER FINISHED", flush=True)


if __name__ == "__main__":
    main()
