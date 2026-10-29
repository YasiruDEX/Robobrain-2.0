"""Groq decomposer for the Agent benchmark (the app's planner, scripts/pipeline_chat.py), with a JSONL cache.

The app's model llama-3.3-70b-versatile is no longer served for our key (404), so the default is
openai/gpt-oss-120b (reasoning_effort low; JSON parsed from `content` only). Deviations from the app:
temperature 0 (app 0.2 / 0.3); the enhancer gets max_tokens 300 (app 100) because reasoning tokens
count towards the limit. Same system prompts, same user messages.

Run as a script, it pre-computes the decomposition for every unique instruction of both datasets plus
the enhancer calls whose prompts are known before the GPU run. Enhancer calls that depend on RoboBrain
output are made live during run_agent.py and cached the same way. Resumable: cached keys are skipped.
"""
import argparse
import json
import os
import time

import requests

from run_agent import DECOMP_SYS, ENHANCE_SYS, Agent
from run_traj import append, load_jsonl

ROOT = "/home/jayamadu/Documents/Robobrain-2.0"
URL = "https://api.groq.com/openai/v1/chat/completions"
DATASETS = {"sharerobot": "sharerobot_traj", "vabench": "vabench_v"}
MAX_TOKENS = {"decompose": 1500, "enhance": 300}


def cache_path(model, kind):
    tag = model.replace("/", "_")
    return f"{ROOT}/results/groq_cache/{tag}.jsonl" if kind == "decompose" else f"{ROOT}/results/groq_cache/{tag}.{kind}.jsonl"


class GroqLLM:
    """chat(kind, system, user) -> {"raw", "latency"}; every call is cached under (kind, user)."""

    def __init__(self, model, offline=False):
        self.model, self.offline = model, offline
        self.key = os.environ.get("GROQ_API_KEY")
        self.cache = {}
        os.makedirs(f"{ROOT}/results/groq_cache", exist_ok=True)
        for kind in MAX_TOKENS:
            for r in load_jsonl(cache_path(model, kind)):
                self.cache[(kind, r["user"])] = r

    def chat(self, kind, system, user):
        hit = self.cache.get((kind, user))
        if hit:
            return {"raw": hit["raw"], "latency": hit["latency"]}
        if self.offline:
            raise KeyError(f"groq cache miss ({kind}): {user[:80]}")
        body = {"model": self.model, "temperature": 0, "max_tokens": MAX_TOKENS[kind], "reasoning_effort": "low",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for attempt in range(30):
            t0 = time.perf_counter()
            try:
                r = requests.post(URL, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=120)
            except requests.RequestException as e:
                print(f"groq network error {e!r}, retry", flush=True)
                time.sleep(min(60, 2 ** attempt))
                continue
            latency = time.perf_counter() - t0
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 2 ** attempt)
                print(f"groq {r.status_code}, waiting {wait:.0f}s", flush=True)
                time.sleep(wait + 1)
                continue
            r.raise_for_status()
            j = r.json()
            rec = {"kind": kind, "user": user, "raw": (j["choices"][0]["message"].get("content") or "").strip(),
                   "finish": j["choices"][0]["finish_reason"], "latency": latency, "usage": j.get("usage")}
            append(cache_path(self.model, kind), rec)
            self.cache[(kind, user)] = rec
            # Stay under the tokens-per-minute limit instead of hitting 429s.
            if int(r.headers.get("x-ratelimit-remaining-tokens", 10 ** 6)) < 2500:
                time.sleep(_secs(r.headers.get("x-ratelimit-reset-tokens", "5s")) + 0.5)
            return {"raw": rec["raw"], "latency": latency}
        raise RuntimeError("groq: too many retries")


def _secs(s):
    """Groq reset headers look like '4.69s', '1m26.4s' or '250ms'."""
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    total, num = 0.0, ""
    for ch in s:
        if ch == "m":
            total, num = total + 60 * float(num), ""
        elif ch == "s":
            total, num = total + float(num), ""
        else:
            num += ch
    return total


class GroqAgent(Agent):
    """The app's agent loop with Groq as planner/enhancer and RoboBrain2.0-3B as executor."""

    def __init__(self, m, llm):
        super().__init__(m)
        self.g = llm

    def llm(self, system, user, max_new_tokens):
        return self.g.chat("decompose" if system is DECOMP_SYS else "enhance", system, user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    os.makedirs(f"{ROOT}/results/groq_cache", exist_ok=True)
    instr = []
    for f in DATASETS.values():
        for r in load_jsonl(f"{ROOT}/data/{f}.jsonl"):
            if r["instruction"] not in instr:
                instr.append(r["instruction"])
    print(f"=== START groq-plans {a.model} unique instructions={len(instr)}", flush=True)
    ag = GroqAgent(None, GroqLLM(a.model))
    n_enh = 0
    for k, q in enumerate(instr, 1):
        plan, ok, _ = ag.decompose(q)
        for st in plan:  # enhancer prompts that do not depend on RoboBrain output (see Agent.run)
            p = str(st.get("prompt", ""))
            if st.get("task") == "trajectory" and not st.get("use_previous") and len(p) < 80:
                ag.enhance(p)
                n_enh += 1
        if k % 25 == 0:
            print(f"groq-plans {k}/{len(instr)} enhance={n_enh}", flush=True)
    rows = load_jsonl(cache_path(a.model, "decompose"))
    print(f"] done (groq {a.model}) decompositions={len(rows)} truncated={sum(r['finish'] != 'stop' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
