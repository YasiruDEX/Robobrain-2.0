"""Task-routing evaluation. Labels come from benchmark provenance (no manual labelling):
trajectory <- ShareRobot traj, affordance <- ShareRobot afford, grounding <- RoboRefIt, pointing <- Where2Place.

Routers: rb2 (local RoboBrain2.0-3B, text-only), keyword (rule baseline), groq (Llama-3.3-70B, only if
GROQ_API_KEY is set). Output: <out>/planner.jsonl, one line per (id, router).
"""
import argparse
import json
import os
import re
import time

from run_traj import append, load_jsonl

CLASSES = ["general", "grounding", "affordance", "trajectory", "pointing"]
SYSTEM = (
    "Classify a robot instruction into exactly one task type:\n"
    "general - questions or descriptions that need no coordinates\n"
    "grounding - locate a named object; answer is a bounding box\n"
    "affordance - where the gripper should contact/grasp an object to do an action; answer is a box\n"
    "trajectory - how to move the end effector to complete a manipulation task; answer is waypoints\n"
    "pointing - pick free-space spots or points satisfying a spatial relation; answer is points\n"
    "Answer with the task type only.")
# Hand-written few-shot examples (not drawn from any test set).
FEWSHOT = ("Examples:\n"
           "Instruction: \"What objects are on the table?\" -> general\n"
           "Instruction: \"Find the blue mug\" -> grounding\n"
           "Instruction: \"Where is the screwdriver?\" -> grounding\n"
           "Instruction: \"Where should the gripper hold the kettle to lift it?\" -> affordance\n"
           "Instruction: \"Grasp the drawer handle\" -> affordance\n"
           "Instruction: \"Open the top drawer\" -> trajectory\n"
           "Instruction: \"Put the apple into the bowl\" -> trajectory\n"
           "Instruction: \"Mark free spots on the shelf left of the box\" -> pointing\n")
KEYWORDS = [("pointing", r"\b(spot|spots|vacant|free space|empty|between|point to|point at|place where)\b"),
            ("trajectory", r"\b(move|put|push|pull|open|close|assemble|wipe|pour|stack|place|turn|fold|insert|slide|"
                           r"lift|drag|go to|reach)\b"),
            ("affordance", r"\b(grasp|grip|hold|handle|throw|cut with|use the)\b"),
            ("grounding", r"\b(where|find|locate|pick up|pass|give|need|want|bring|fetch|the .+)\b")]


def keyword_route(q):
    for task, pat in KEYWORDS:
        if re.search(pat, q.lower()):
            return task
    return "general"


def norm_label(raw):
    raw = raw.lower()
    hits = [c for c in CLASSES if c in raw]
    return hits[0] if hits else "general"


def build_items():
    d = "/home/jayamadu/Documents/Robobrain-2.0/data"
    items = []
    for r in load_jsonl(f"{d}/sharerobot_traj.jsonl"):
        items.append({"id": f"traj/{r['id']}", "label": "trajectory", "q": r["instruction"]})
    for r in load_jsonl(f"{d}/sharerobot_afford.jsonl"):
        items.append({"id": f"afford/{r['id']}", "label": "affordance", "q": r["instruction"]})
    for r in load_jsonl(f"{d}/roborefit.jsonl"):
        items.append({"id": f"ground/{r['id']}", "label": "grounding", "q": r["instruction"]})
    for r in load_jsonl(f"{d}/where2place.jsonl"):
        items.append({"id": f"point/{r['id']}", "label": "pointing", "q": r["question"].split(" Your answer")[0]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--routers", default="keyword,rb2,groq")
    ap.add_argument("--model", default="/home/jayamadu/Documents/Robobrain-2.0/weights/RoboBrain2.0-3B")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/planner.jsonl"
    done = {(r["id"], r["router"]) for r in load_jsonl(path)}
    items = build_items()
    print(f"=== START planner n={len(items)}", flush=True)
    for router in a.routers.split(","):
        if router == "groq" and not os.environ.get("GROQ_API_KEY"):
            print("] skipped (groq: no GROQ_API_KEY)", flush=True)
            continue
        if router in ("rb2", "rb2fs"):
            from rb2 import RB2
            m = RB2(a.model)
        if router == "groq":
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
        for it in items:
            if (it["id"], router) in done:
                continue
            t0 = time.perf_counter()
            if router == "keyword":
                raw = keyword_route(it["q"])
            elif router == "rb2":
                raw = m.infer(None, f"{SYSTEM}\nInstruction: \"{it['q']}\"", "general")["raw"]
            elif router == "rb2fs":
                raw = m.infer(None, f"{SYSTEM}\n{FEWSHOT}Instruction: \"{it['q']}\" ->", "general")["raw"]
            else:
                raw = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", temperature=0, max_tokens=10,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"Instruction: \"{it['q']}\""}]).choices[0].message.content
            pred = norm_label(raw)
            append(path, {"id": it["id"], "router": router, "label": it["label"], "pred": pred, "raw": raw,
                          "latency": time.perf_counter() - t0})
        print(f"] done ({router})", flush=True)
        if router in ("rb2", "rb2fs"):
            del m
    print("=== PLANNER FINISHED", flush=True)


if __name__ == "__main__":
    main()
