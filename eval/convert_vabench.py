"""IffYuan/vabench-v (300, object-centric visual traces) -> data/vabench_v.jsonl (gt_traj in [0,1])."""
import io
import json
import os
import re

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"


def main():
    src = f"{ROOT}/data/vabench_v"
    snapshot_download("IffYuan/vabench-v", repo_type="dataset", local_dir=src, allow_patterns=["*.parquet"])
    pqs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(src) for f in fs if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    os.makedirs(f"{src}/images", exist_ok=True)
    n_bad = 0
    with open(f"{ROOT}/data/vabench_v.jsonl", "w") as f:
        for _, r in df.iterrows():
            m = re.search(r"task instruction is:\s*(.+?)\.\s*Use 2D points", r["problem"], re.S)
            if not m:
                n_bad += 1
                continue
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            path = f"{src}/images/{r['id']}.jpg"
            img.save(path, quality=95)
            w, h = img.size
            ans = r["answer"] if not isinstance(r["answer"], str) else json.loads(r["answer"])
            pts = [(float(x) / w, float(y) / h) for x, y in ans]
            f.write(json.dumps({"id": str(r["id"]), "image": path, "instruction": m.group(1).strip(),
                                "gt_traj": pts, "img_wh": [w, h]}) + "\n")
    print(f"wrote {len(df) - n_bad} samples; skipped {n_bad}; columns {list(df.columns)}")


if __name__ == "__main__":
    main()
