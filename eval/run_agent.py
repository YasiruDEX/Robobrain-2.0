"""Benchmark the original app's agentic layer (scripts/pipeline_chat.py) with the Groq LLM replaced
by RoboBrain2.0-3B's own decoder ("Agent-local").

Faithful to pipeline_chat.py: same decomposition system prompt and task table (loaded from the file),
same JSON parsing/fallback, same short-trajectory-prompt enhancer, multi-turn history prefix
(conversation_memory.get_context_prompt), `(Context: prev[:200])` injection for use_previous steps,
and robot-arm start-point prepending (50 px rule). Deviations: greedy decoding instead of
temperature 0.2/0.3 sampling; the system prompt is sent as the first part of the user message.

Scores: trajectory -> last trajectory step vs GT (cond "Agent-local", and "Agent-local+arm" with the
robot-arm point prepended) and ":concat" (all trajectory steps joined in plan order, pre-registered
headline for multi-stop tasks); affordance -> last affordance step's box IoU vs GT.
--planner groq uses the Groq decomposer (groq_plans.py) instead of the local decoder; --dataset vabench.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import traceback

import torch

from metrics import box_iou, traj_metrics
from parse import parse_box, parse_points, valid_coords
from run_traj import append, load_jsonl

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
SRC = open(f"{ROOT}/scripts/pipeline_chat.py").read()


def _legacy_prompts():
    tree = ast.parse(SRC)
    tasks = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "ROBOBRAIN_TASKS")
    td = "\n".join(f"- {k}: {v['description']} (Output: {v['output']})" for k, v in tasks.items())
    sys_t = re.search(r'system_prompt = f"""(.*?)"""', SRC, re.S).group(1)
    sys_t = sys_t.replace("{task_descriptions}", td).replace("{{", "{").replace("}}", "}")
    enh = re.search(r'"content": """(You are a robot motion planning expert.*?)"""', SRC, re.S).group(1)
    return sys_t, enh


DECOMP_SYS, ENHANCE_SYS = _legacy_prompts()
DECOMP_USER = "Decompose this query into a detailed task pipeline:\n\nQuery: \"{q}\""
ARM_Q = "robot arm end-effector OR robot gripper OR robotic manipulator tip"


class Agent:
    def __init__(self, m):
        self.m = m

    def llm(self, system, user, max_new_tokens):
        old = self.m.max_new_tokens
        self.m.max_new_tokens = max_new_tokens
        try:
            return self.m.infer(None, f"{system}\n\n{user}", "general")
        finally:
            self.m.max_new_tokens = old

    def decompose(self, query):
        r = self.llm(DECOMP_SYS, DECOMP_USER.format(q=query), 1500)  # original app: max_tokens=1500
        mt = re.search(r"\[[\s\S]*\]", r["raw"])
        plan, ok, r["repaired"] = None, False, False
        if mt:
            try:
                plan, ok = json.loads(mt.group()), True
            except json.JSONDecodeError:
                try:  # deviation from the app (user 2026-10-26): local decoder writes markdown "\_" escapes
                    plan, ok, r["repaired"] = json.loads(mt.group().replace("\\_", "_")), True, True
                except json.JSONDecodeError:
                    pass
        if not ok or not isinstance(plan, list) or not all(isinstance(s, dict) for s in plan):
            plan, ok = [{"step": 1, "task": "general", "prompt": query, "use_previous": False}], False
        for st in plan:
            if st.get("task") == "trajectory" and len(st.get("prompt", "")) < 50:
                st["prompt"] = f"generate trajectory waypoints for the robot end-effector to {st.get('prompt', '')}"
        return plan, ok, r

    def enhance(self, prompt):
        r = self.llm(ENHANCE_SYS, f"Enhance this trajectory prompt: \"{prompt}\"", 100)
        e = r["raw"].strip().strip("\"'")
        return (e if len(e) > len(prompt) else prompt), r["latency"]

    def run(self, image, query):
        t0 = time.perf_counter()
        plan, parsed, dr = self.decompose(query)
        turns, steps, prev = [], [], None
        for st in plan:
            task = st.get("task", "general")
            task = task if task in ("general", "grounding", "affordance", "trajectory", "pointing") else "general"
            prompt = str(st.get("prompt", ""))
            if st.get("use_previous") and prev:
                prompt = prompt.replace("{previous_result}", str(prev))
                if "{previous_result}" not in str(st.get("prompt", "")):
                    prompt = f"{prompt} (Context: {prev[:200]})"
            if task == "trajectory" and len(prompt) < 80:
                prompt, _ = self.enhance(prompt)
            ctx = ""
            if turns:  # conversation_memory.get_context_prompt (last 10 turns)
                ctx = "\n".join(["[Previous conversation context]"] +
                                [f"{'User (with image)' if r == 'user' else 'Assistant'}: {c}" for r, c in turns[-10:]] +
                                ["[Current query]"]) + "\n"
            r = self.m.infer(image, f"{ctx}{prompt}", task)
            turns += [("user", prompt), ("assistant", r["raw"])]
            prev = r["raw"]
            steps.append({"task": task, "prompt": prompt, "raw": r["raw"], "scale": r["scale"]})
        return {"plan": plan, "plan_parsed": parsed, "plan_repaired": dr["repaired"], "raw_plan": dr["raw"], "steps": steps, "turns": turns,
                "latency": time.perf_counter() - t0}

    def arm_point(self, image, turns):
        """detect_robot_arm: grounding call with the conversation history, centre of first box."""
        ctx = "\n".join(["[Previous conversation context]"] +
                        [f"{'User (with image)' if r == 'user' else 'Assistant'}: {c}" for r, c in turns[-10:]] +
                        ["[Current query]"]) + "\n"
        r = self.m.infer(image, f"{ctx}{ARM_Q}", "grounding")
        b = re.search(r"\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]", r["raw"])
        if b:
            x1, y1, x2, y2 = map(float, b.groups())
            return ((x1 + x2) / 2, (y1 + y2) / 2), r
        p = re.search(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)", r["raw"])
        return ((float(p.group(1)), float(p.group(2))) if p else None), r


def concat_points(steps):
    """All trajectory steps' points in plan order, consecutive duplicates dropped (fair score for multi-stop plans)."""
    pts = []
    for st in steps:
        if st["task"] == "trajectory":
            for p in parse_points(st["raw"], st["scale"]):
                if not pts or tuple(p) != tuple(pts[-1]):
                    pts.append(tuple(p))
    return pts


def with_arm(pts, arm, scale):
    """Prepend the detected arm point when it is > 50 px (model frame) from the first point (app rule)."""
    if not (pts and arm):
        return pts
    sw, sh = scale
    fx, fy = pts[0][0] * sw, pts[0][1] * sh
    return [(arm[0] / sw, arm[1] / sh)] + list(pts) if ((fx - arm[0]) ** 2 + (fy - arm[1]) ** 2) ** 0.5 > 50 else pts


def export_traj(runs_path, out, b_path, tag):
    """traj.jsonl next to the B rows (paired by id): last / last+arm (app-faithful) and concat / concat+arm."""
    with open(f"{out}/traj.jsonl", "w") as f:
        for r in load_jsonl(b_path):
            if r["cond"] in ("B", "C-pred"):
                f.write(json.dumps(r) + "\n")
        for r in load_jsonl(runs_path):
            gt = r["gt_traj"]
            tr = [st for st in r.get("steps", []) if st["task"] == "trajectory"]
            last = parse_points(tr[-1]["raw"], tr[-1]["scale"]) if tr else []
            cat = concat_points(r.get("steps", []))
            scale = tr[-1]["scale"] if tr else None
            arm = r.get("arm")
            for cond, pts in ((f"Agent-{tag}", last), (f"Agent-{tag}+arm", with_arm(last, arm, scale)),
                              (f"Agent-{tag}:concat", cat), (f"Agent-{tag}:concat+arm", with_arm(cat, arm, scale))):
                ok = len(pts) >= 2 and valid_coords(pts)
                f.write(json.dumps({"id": r["id"], "cond": cond, "valid": ok, "pred": pts,
                                    "metrics": traj_metrics(pts, gt) if ok else None,
                                    "latency": r.get("latency", 0.0)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["trajectory", "affordance"], required=True)
    ap.add_argument("--planner", choices=["local", "groq"], default="local")
    ap.add_argument("--groq-model", default="openai/gpt-oss-120b")
    ap.add_argument("--dataset", choices=["sharerobot", "vabench"], default="sharerobot")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=f"{ROOT}/weights/RoboBrain2.0-3B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", action="store_true", help="only rebuild traj.jsonl from the runs file (no GPU)")
    ap.add_argument("--cached-only", action="store_true", help="groq: skip samples whose decomposition is not cached")
    a = ap.parse_args()
    if a.out is None:  # legacy default: results/agent (Agent-local, ShareRobot)
        a.out = (f"{ROOT}/results/agent" if (a.planner, a.dataset) == ("local", "sharerobot")
                 else f"{ROOT}/results/decomp/{a.planner}_{a.dataset}")
    tag = "local" if a.planner == "local" else "groq-" + a.groq_model.split("/")[-1].replace("gpt-oss-", "")
    data = {"affordance": "sharerobot_afford"}.get(a.task) or {"sharerobot": "sharerobot_traj", "vabench": "vabench_v"}[a.dataset]
    b_path = f"{ROOT}/results/{'traj' if a.dataset == 'sharerobot' else 'vabench'}/traj.jsonl"
    samples = load_jsonl(f"{ROOT}/data/{data}.jsonl")[: a.limit or None]
    gt = {s["id"]: s.get("gt_traj") for s in samples}
    os.makedirs(a.out, exist_ok=True)
    raw_path = f"{a.out}/{a.task}_runs.jsonl"
    if a.rescore:
        runs = load_jsonl(raw_path)
        tmp = f"{a.out}/.runs_gt.jsonl"
        with open(tmp, "w") as f:
            for r in runs:
                f.write(json.dumps({**r, "gt_traj": gt[r["id"]]}) + "\n")
        export_traj(tmp, a.out, b_path, tag)
        os.remove(tmp)
        print(f"] done (rescore {a.out})", flush=True)
        return
    done = {r["id"] for r in load_jsonl(raw_path)}
    print(f"=== START agent {a.task} planner={a.planner} dataset={a.dataset} n={len(samples)}", flush=True)
    from rb2 import RB2
    m = RB2(a.model)
    if a.planner == "groq":
        from groq_plans import GroqAgent, GroqLLM
        ag = GroqAgent(m, GroqLLM(a.groq_model))
        if a.cached_only:  # Groq daily token cap: run only instructions already decomposed (kept as a partial subset)
            n0 = len(samples)
            samples = [s for s in samples if ("decompose", DECOMP_USER.format(q=s["instruction"])) in ag.g.cache]
            print(f"cached-only: {len(samples)}/{n0} samples have a cached Groq plan", flush=True)
    else:
        ag = Agent(m)
    for s in samples:
        if s["id"] in done:
            continue
        try:
            try:
                run = ag.run(s["image"], s["instruction"])
            except torch.OutOfMemoryError:  # long multi-turn context on a 4 GB GPU: retry once, then count as failure
                torch.cuda.empty_cache()
                try:
                    run = ag.run(s["image"], s["instruction"])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"=== OOM agent {s['id']} (counted as failure)", flush=True)
                    append(raw_path, {"id": s["id"], "oom": True, "valid": False, "metrics": None, "pred": [],
                                      "iou": 0.0, "box": None, "plan_parsed": None, "plan_repaired": None, "steps": []})
                    continue
            rec = {"id": s["id"], **{k: run[k] for k in ("plan", "plan_parsed", "plan_repaired", "raw_plan", "steps", "latency")},
                   "n_steps": len(run["steps"]), "tasks": [st["task"] for st in run["steps"]]}
            if a.task == "trajectory":
                tr = [st for st in run["steps"] if st["task"] == "trajectory"]
                pts = parse_points(tr[-1]["raw"], tr[-1]["scale"]) if tr else []
                ok = len(pts) >= 2 and valid_coords(pts)
                rec.update(pred=pts, valid=ok, metrics=traj_metrics(pts, s["gt_traj"]) if ok else None)
                # Arm point whenever any trajectory step exists (concat scoring needs it even if the last step failed).
                arm, ar = ag.arm_point(s["image"], run["turns"]) if tr else (None, None)
                pts_arm = with_arm(pts, arm, tr[-1]["scale"]) if ok else pts
                rec.update(arm=arm, raw_arm=ar["raw"] if ar else None, pred_arm=pts_arm,
                           metrics_arm=traj_metrics(pts_arm, s["gt_traj"]) if ok else None)
            else:
                af = [st for st in run["steps"] if st["task"] == "affordance"]
                b = parse_box(af[-1]["raw"], af[-1]["scale"]) if af else None
                rec.update(box=b, valid=b is not None, iou=box_iou(b, s["gt_box"]) if b else 0.0)
        except Exception:
            traceback.print_exc()
            print(f"=== FAILED agent {s['id']}", flush=True)
            sys.exit(1)
        append(raw_path, rec)
    print(f"] done (agent {a.task} {a.planner} {a.dataset})", flush=True)
    if a.task == "trajectory":
        tmp = f"{a.out}/.runs_gt.jsonl"
        with open(tmp, "w") as f:
            for r in load_jsonl(raw_path):
                f.write(json.dumps({**r, "gt_traj": gt[r["id"]]}) + "\n")
        export_traj(tmp, a.out, b_path, tag)
        os.remove(tmp)
    print("=== AGENT FINISHED", flush=True)


if __name__ == "__main__":
    main()
