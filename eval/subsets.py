"""Pre-registered multi-stop / single-stop split (frozen 2026-10-28, before any decomposition numbers).

Multi-stop = the instruction moves an object to a destination: a transport verb AND a destination phrase.
Single-stop = everything else (reach / approach / pick X / open / close / wipe / turn / lift / interact ...).
v2 (2026-10-28, after the 30-label hand check, BEFORE any split numbers): "move towards X" is end-effector
reaching -> single; destinations add behind/above/below/beside/across/forward/over/under; stack/destack
need no destination phrase (the other cube is the destination).
Writes data/subsets.json {dataset: {id: "multi"|"single"}} and prints counts + 30 random labels for the hand check.
"""
import json
import random
import re

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
DATASETS = {"sharerobot": "sharerobot_traj", "vabench": "vabench_v"}

VERB = re.compile(r"\b(put|puts|place|places|move|moves|moved|moving|stack|stacks|insert|inserts|drag|drags|slide|slides"
                  r"|push|pushes|take|takes|separate|separates|destack|destacks)\b")
PICK_PLACE = re.compile(r"\bpick(s|ed|ing)?\b.*\b(place|places|placed|put|puts)\b")
DEST = re.compile(r"\b(in|into|on|onto|to|towards|near|next to|inside|on top of|between|left|right"
                  r"|behind|above|below|beside|across|forward|over|under)\b|\bfrom .+ to\b")
STACK = re.compile(r"\b(de)?stacks?\b")
REACH = re.compile(r"^(move|moving|moves) towards\b")


def label(instruction):
    t = instruction.lower().strip().strip("\"'")
    if REACH.search(t):
        return "single"
    if STACK.search(t):
        return "multi"
    verb = bool(VERB.search(t) or PICK_PLACE.search(t))
    return "multi" if verb and DEST.search(t) else "single"


def main():
    out = {}
    for name, f in DATASETS.items():
        rows = [json.loads(l) for l in open(f"{ROOT}/data/{f}.jsonl")]
        out[name] = {r["id"]: label(r["instruction"]) for r in rows}
        n_multi = sum(v == "multi" for v in out[name].values())
        print(f"{name}: n={len(rows)} multi={n_multi} single={len(rows) - n_multi}")
    json.dump(out, open(f"{ROOT}/data/subsets.json", "w"), indent=1)

    rng = random.Random(0)
    pool = [(n, json.loads(l)) for n, f in DATASETS.items() for l in open(f"{ROOT}/data/{f}.jsonl")]
    for n, r in rng.sample(pool, 30):
        print(f"{out[n][r['id']]:6s} | {n:10s} | {r['instruction']}")


if __name__ == "__main__":
    main()
