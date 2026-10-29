"""BAAI/ShareRobot-Bench trajectory/test.json -> harness JSONL (gt_traj normalised to [0,1])."""
import argparse
import ast
import json
import os
import re

from huggingface_hub import snapshot_download
from PIL import Image

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ROOT}/data/sharerobot_bench")
    ap.add_argument("--out", default=f"{ROOT}/data/sharerobot_traj.jsonl")
    a = ap.parse_args()
    snapshot_download("BAAI/ShareRobot-Bench", repo_type="dataset", local_dir=a.src,
                      allow_patterns=["trajectory/*"])
    recs = json.load(open(f"{a.src}/trajectory/test.json"))
    n_bad = 0
    with open(a.out, "w") as f:
        for r in recs:
            q = next(c["value"] for c in r["conversations"] if c["from"] == "human")
            ans = next(c["value"] for c in r["conversations"] if c["from"] == "gpt")
            m = re.search(r"You are (.+?)\. The task is (.+?)\. Please predict", q, re.S)
            if not m:
                n_bad += 1
                continue
            img = os.path.join(a.src, "trajectory", r["image"][0])
            w, h = Image.open(img).size
            pts = [(x / w, y / h) for x, y in ast.literal_eval(ans)]
            f.write(json.dumps({"id": str(r["id"]), "image": img, "robot": m.group(1).strip(),
                                "instruction": m.group(2).strip(), "gt_traj": pts,
                                "img_wh": [w, h]}) + "\n")
    print(f"wrote {len(recs) - n_bad} samples to {a.out}; skipped {n_bad} unparsable prompts")


if __name__ == "__main__":
    main()
