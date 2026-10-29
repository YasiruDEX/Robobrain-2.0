"""Grounding / pointing / affordance evaluation (resumable), incl. grounding-conditioned affordance.

--task grounding  : RoboRefIt (data/roborefit.jsonl)          cond B             metric IoU
--task pointing   : Where2Place (data/where2place.jsonl)      cond B             metric frac. points in mask
--task affordance : ShareRobot-Bench (data/sharerobot_afford) conds B,C-text,C-pred,C-shuffle   metric IoU
Affordance C-* conditions first ground the planner-extracted target, then inject the box.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
from PIL import Image

from metrics import box_iou
from parse import parse_box, parse_points
from run_traj import PLANNER_Q, append, clean_target, load_jsonl, to_px

DATA = {"grounding": "roborefit", "pointing": "where2place", "affordance": "sharerobot_afford"}


def mask_score(pts, mask_path):
    m = np.array(Image.open(mask_path).convert("L")) > 0
    h, w = m.shape
    if not pts:
        return 0.0
    hit = [0 <= x < 1 and 0 <= y < 1 and m[int(y * h), int(x * w)] for x, y in pts]
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/jayamadu/Documents/Robobrain-2.0/weights/RoboBrain2.0-3B")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = "/home/jayamadu/Documents/Robobrain-2.0/data"
    samples = load_jsonl(f"{root}/{DATA[a.task]}.jsonl")[: a.limit or None]
    os.makedirs(a.out, exist_ok=True)
    out_path = f"{a.out}/{a.task}.jsonl"
    json.dump(vars(a), open(f"{a.out}/config_{a.task}.json", "w"), indent=1)
    done = {(r["id"], r["cond"]) for r in load_jsonl(out_path)}
    print(f"=== START {a.task} n={len(samples)}", flush=True)

    from rb2 import RB2
    m = RB2(a.model, quant=a.quant, max_pixels=a.max_pixels)

    try:
        if a.task == "grounding":
            for s in samples:
                if (s["id"], "B") in done:
                    continue
                r = m.infer(s["image"], s["instruction"], "grounding")
                b = parse_box(r["raw"], r["scale"])
                append(out_path, {"id": s["id"], "cond": "B", "raw": r["raw"], "box": b, "valid": b is not None,
                                  "iou": box_iou(b, s["gt_box"]) if b else 0.0, "latency": r["latency"]})
        elif a.task == "pointing":
            for s in samples:
                if (s["id"], "B") in done:
                    continue
                r = m.infer(s["image"], s["question"], "general")
                pts = parse_points(r["raw"], r["scale"])
                append(out_path, {"id": s["id"], "cond": "B", "raw": r["raw"], "pred": pts, "valid": bool(pts),
                                  "score": mask_score(pts, s["mask"]), "category": s["category"],
                                  "latency": r["latency"]})
        else:
            # pass 1: planner target + grounding box (shared by C-* conditions)
            steps_path = f"{a.out}/affordance_steps.jsonl"
            steps = {r["id"]: r for r in load_jsonl(steps_path)}
            for s in samples:
                if s["id"] in steps:
                    continue
                p = m.infer(None, PLANNER_Q.format(s["instruction"]), "general")
                target = clean_target(p["raw"])
                g = m.infer(s["image"], target, "grounding")
                rec = {"id": s["id"], "target": target, "scale": g["scale"], "box": parse_box(g["raw"], g["scale"]),
                       "raw_plan": p["raw"], "raw_ground": g["raw"], "lat_plan": p["latency"],
                       "lat_ground": g["latency"]}
                append(steps_path, rec)
                steps[s["id"]] = rec
            print("] done (pass1)", flush=True)
            ids = [s["id"] for s in samples]
            partner = {ids[i]: ids[(i + max(1, len(ids) // 2)) % len(ids)] for i in range(len(ids))}
            for cond in ["B", "C-text", "C-pred", "C-shuffle"]:
                for s in samples:
                    if (s["id"], cond) in done:
                        continue
                    st = steps[s["id"]]
                    box = {"C-pred": st["box"], "C-shuffle": steps[partner[s["id"]]]["box"]}.get(cond)
                    if cond == "B":
                        ctx = ""
                    elif cond == "C-text":
                        ctx = f'The target object "{st["target"]}" is located in the image.'
                    elif box is None:
                        append(out_path, {"id": s["id"], "cond": cond, "valid": False, "reason": "no_prior",
                                          "iou": 0.0})
                        continue
                    else:
                        ctx = (f'The target object "{st["target"]}" is located at bounding box '
                               f'{to_px(box, st["scale"])} in the image.')
                    r = m.infer(s["image"], s["instruction"], "affordance", ctx)
                    b = parse_box(r["raw"], r["scale"])
                    append(out_path, {"id": s["id"], "cond": cond, "context": ctx, "raw": r["raw"], "box": b,
                                      "valid": b is not None, "iou": box_iou(b, s["gt_box"]) if b else 0.0,
                                      "latency": r["latency"]})
                print(f"] done ({cond})", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"=== FAILED {a.task}", flush=True)
        sys.exit(1)
    print("=== RUNNER FINISHED", flush=True)


if __name__ == "__main__":
    main()
