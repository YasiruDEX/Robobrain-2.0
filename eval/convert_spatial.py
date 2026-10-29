"""Convert the spatial benchmarks to harness JSONL (coordinates normalised to [0,1]).

affordance: BAAI/ShareRobot-Bench affordance/test.json (200)       -> data/sharerobot_afford.jsonl
pointing:   wentao-yuan/where2place point_questions.jsonl (100)    -> data/where2place.jsonl
grounding:  IffYuan/Roborefit test shard 0, seeded subsample (300) -> data/roborefit.jsonl
"""
import argparse
import io
import json
import os
import random
import re

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
D = f"{ROOT}/data"


def affordance():
    src = f"{D}/sharerobot_bench"
    snapshot_download("BAAI/ShareRobot-Bench", repo_type="dataset", local_dir=src, allow_patterns=["affordance/*"])
    out = []
    for r in json.load(open(f"{src}/affordance/test.json")):
        q = next(c["value"] for c in r["conversations"] if c["from"] == "human")
        a = next(c["value"] for c in r["conversations"] if c["from"] == "gpt")
        m = re.search(r'The task is "(.+?)"', q)
        img = f"{src}/affordance/{r['image'] if isinstance(r['image'], str) else r['image'][0]}"
        w, h = Image.open(img).size
        x1, y1, x2, y2 = json.loads(a)
        out.append({"id": str(r["id"]), "image": img, "instruction": m.group(1),
                    "gt_box": [x1 / w, y1 / h, x2 / w, y2 / h], "img_wh": [w, h]})
    return out


def pointing():
    src = f"{D}/where2place"
    snapshot_download("wentao-yuan/where2place", repo_type="dataset", local_dir=src)
    out = []
    for l in open(f"{src}/point_questions.jsonl"):
        r = json.loads(l)
        stem = os.path.splitext(r["image"])[0]
        mask = next(f"{src}/masks/{f}" for f in os.listdir(f"{src}/masks") if os.path.splitext(f)[0] == stem)
        out.append({"id": str(r["question_id"]), "image": f"{src}/images/{r['image']}", "question": r["text"],
                    "mask": mask, "category": r["category"]})
    return out


def grounding(n=300, seed=0):
    src = f"{D}/roborefit"
    os.makedirs(f"{src}/images", exist_ok=True)
    pq = hf_hub_download("IffYuan/Roborefit", "data/test-00000-of-00002.parquet", repo_type="dataset",
                         local_dir=src)
    df = pd.read_parquet(pq)
    idx = sorted(random.Random(seed).sample(range(len(df)), n))
    out = []
    for i in idx:
        r = df.iloc[i]
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        path = f"{src}/images/{r['question_id']}.jpg"
        img.save(path, quality=95)
        w, h = img.size
        x1, y1, x2, y2 = [float(v) for v in r["bbox"]]
        out.append({"id": str(r["question_id"]), "image": path, "instruction": r["question"],
                    "gt_box": [x1 / w, y1 / h, x2 / w, y2 / h], "img_wh": [w, h]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="affordance,pointing,grounding")
    a = ap.parse_args()
    names = {"affordance": "sharerobot_afford", "pointing": "where2place", "grounding": "roborefit"}
    for w in a.which.split(","):
        recs = globals()[w]()
        with open(f"{D}/{names[w]}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{w}: wrote {len(recs)} -> {D}/{names[w]}.jsonl", flush=True)


if __name__ == "__main__":
    main()
