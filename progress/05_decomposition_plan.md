# Plan: task-decomposition layer on multi-stop (complex) tasks — Groq vs inbuilt decoder
Written 2026-10-28. NOT STARTED (user: "only create a plan, I will run this later").

## Question
Earlier, RoboBrain plus the Groq decomposer seemed more accurate on multi-stop tasks
("pick the apple and put it in the basket"). We test that claim on public GT:
- H1 (Groq): Agent-Groq beats B (single call, no decomposition) on the MULTI-STOP subset.
- H2 (inbuilt Qwen): Agent-local (RoboBrain2.0-3B's own Qwen2.5-VL decoder as decomposer) beats B there.
- H3: the gain (if any) is specific to multi-stop tasks (interaction with subset).
Existing evidence (results/agent, ShareRobot all 200): Agent-local DFD +0.063 (worse). This was never
split by task type, and it scores only the LAST trajectory step, which is unfair to multi-step plans.

## Fixed facts found while planning
- The Groq key works, but `llama-3.3-70b-versatile` (the app's model, pipeline_chat.py:104) returns
  404 model_not_found for this key. Available text models are openai/gpt-oss-120b, openai/gpt-oss-20b
  and qwen/qwen3.8-27b. Primary choice: **openai/gpt-oss-120b** (closest in strength to 70B; set
  `reasoning_effort: "low"`, and the JSON is parsed from `content` only). Optional second decomposer:
  qwen/qwen3.8-27b. The paper must state that the app's model is no longer available.
- Key stored at ~/.config/groq/key (chmod 600, outside the repo). Read it via
  `GROQ_API_KEY=$(cat ~/.config/groq/key)`. Never write it into the repo.
  It was pasted in chat, so rotate it after the run.
- The `groq` package is not installed in rb2eval. Use plain `requests`/`urllib` to the
  OpenAI-compatible endpoint, or `pip install groq`.
- Unique instructions: ShareRobot 127, VABench-V 264, so we need about 391 decompositions per Groq model
  (+ enhancer calls). This is small, so cache everything.

## Step 0 — pre-register the subset BEFORE looking at any new numbers
Multi-stop = the instruction moves an object to a destination. Freeze this rule in `eval/subsets.py`:
- verb in {put, place, move, moved, moving, stack, insert, drag, slide, push, take, pick…place,
  separate, destack} AND a destination phrase matching
  `\b(in|into|on|onto|to|towards|near|next to|inside|on top of|between|from .+ to|left|right)\b`
- single-stop = everything else (reach/approach/pick X/open/close/wipe/turn/lift/interact).
Write `data/subsets.json` {id: "multi"|"single"} for both datasets and print the counts.
Expected: VABench-V almost all multi, ShareRobot mixed. Hand-check 30 random labels and log the check.

## Step 1 — free analysis of existing results (CPU, ~5 min)
Split results/agent (Agent-local, ShareRobot) and results/traj B by subset. Report dDFD with a CI per subset.
This already answers H2 for ShareRobot under "last-step" scoring.

## Step 2 — code changes (no GPU)
1. `eval/groq_plans.py`: for each unique instruction × Groq model, call the decomposer with the
   app's exact system prompt (reuse `_legacy_prompts()` from run_agent.py) at temperature 0
   (deviation from 0.2; document it) and max_tokens 1500. Cache to `results/groq_cache/<model>.jsonl`
   {instruction, raw, plan, parsed, latency, usage}. Retry with backoff on 429/5xx, read the
   `x-ratelimit-*` headers, and resume by skipping cached rows. The enhancer calls (max_tokens 100) are cached the same way,
   keyed by prompt text. End markers `] done (groq <model>)` / `=== FAILED`.
   This runs BEFORE the GPU job, so API problems never waste GPU time.
2. `eval/run_agent.py`: add `--planner {local,groq}`, `--groq-model`, `--dataset {sharerobot,vabench}`.
   With the groq planner, decompose() and enhance() read from the cache (and fail loudly on a miss).
   Executor stays RoboBrain2.0-3B, 4-bit NF4, greedy. Output dir per (planner, dataset).
3. Scoring: keep "last" (app-faithful) and add **"concat"**: all trajectory steps' points in plan
   order, joined at step boundaries, with consecutive duplicate points dropped. Also "+arm" as before.
   Concat is the fair score for "go to apple, then to basket" plans. Pre-register concat as the
   headline metric for the multi-stop subset; last-step stays secondary.
4. Log per sample: n_steps, tasks, plan_parsed, plan_repaired, OOM.
5. `summarize.py` / `analyze.py`: new section 6 with a table of
   subset × {B, Agent-local, Agent-Groq-120b[, Agent-Groq-qwen27b]} × {last, concat}.
   Use paired bootstrap 95% CI and Holm within each subset (family = number of agent conditions × 1 headline metric).
   Report the DFD headline plus FDE (did it reach the goal?) and plan stats (mean steps, % with ≥2 trajectory steps).

## Step 3 — runner `eval/run_decomp.sh` (one detached job, resumable, priority order)
```
=== START groq-plans        (CPU, ~10-20 min, rate-limit bound)
=== START smoke             (3 samples each planner, sanity: plans parsed >=2/3, valid traj)
=== START groq-vabench      (headline: multi-stop dataset)      ~300 x ~8 s  = ~40 min
=== START local-vabench     (inbuilt decoder)                   ~300 x 16.6 s = ~85 min
=== START groq-sharerobot                                       ~200 x ~8 s  = ~30 min
(local-sharerobot: reuse results/agent; only re-score with concat)
=== START summary
=== DECOMP RUNNER FINISHED
```
Total ≈ 3 h GPU (4 GB RTX 3050 Ti, one model at a time). Use the same OOM retry as run_afford.sh and set
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. B rows are reused from results/traj and results/vabench
(same model/settings), so B is not re-run.

## Step 4 — orchestration (research-agents skill, per user preferences)
Launch (from the repo root):
```
GROQ_API_KEY=$(cat ~/.config/groq/key) nohup setsid bash eval/run_decomp.sh > logs/run_decomp.log 2>&1 &
```
- Main confirms the smoke step is sane (plans parsed, coordinates valid) before leaving the job alone.
- Zero-token watcher (one Bash call with run_in_background):
  `until grep -qE "DECOMP RUNNER FINISHED|FAILED|Traceback|Killed|out of memory" logs/run_decomp.log || ! pgrep -f run_decomp.sh >/dev/null; do sleep 60; done; grep -E "\] done \(|FAILED|FINISHED" logs/run_decomp.log; tail -n 8 logs/run_decomp.log`
- Haiku cache keepalive: ping every 50 min, max 6 pings, stop-file `results/.keepalive_stop`. On each ping, run one
  status grep and give a short status line.
- Workers: none during the run (user prefers minimal agents). After results, optionally use ONE sonnet worker
  that owns only paper/main.tex for the new subsection, while main checks the numbers.
- Checkpoint/resume: the runner skips finished ids, so the existing `scripts/checkpoint_state.sh`
  pattern applies. Add run_decomp to resume_jobs.sh.

## Decision rule (pre-registered)
- Agent-X "improves complex tasks" only if multi-stop dDFD(concat) < 0 with the 95% CI excluding 0
  (Holm-adjusted p < .05) on VABench-V. ShareRobot multi-stop is the replication.
- If Agent-Groq improves but Agent-local does not, the paper says: the decomposition gain needs a
  strong external planner, and the inbuilt 3B decoder is not enough (report the parse/repair rate).
- If neither improves, report that honestly, with a plan-quality breakdown (steps, parse failures, wrong targets).
- Never compare against a "last-step" result while claiming a "concat" result. Report both.

## Paper impact
New subsection "Decomposition on multi-stop tasks", a Table (subset × planner), a cost row (Groq latency +
tokens), a limitation (gpt-oss-120b replaces the deprecated Llama-3.3-70B; temperature 0).

## PROGRESS 2026-10-28 (evening)
- Step 0 DONE: eval/subsets.py -> data/subsets.json. v1 rule hand check (30 random, seed 0) found systematic errors:
  "move towards X" (reaching) labelled multi; behind/above/beside/across/forward missing; "stack X Y cube" has no
  preposition. Fixed as v2 BEFORE any split numbers. Counts v2: ShareRobot multi 54 / single 146; VABench-V 285 / 15.
  Known edge case left as is: "pick the carrot from the cloth to the top right side of the table" -> single.
- Step 1 DONE (eval/decomp_split.py, ShareRobot, LAST-step scoring, Holm over 3 conds per subset):
  multi n=53: Agent-local dDFD +0.017 [-0.025,+0.058] p=.56; +arm +0.027 n.s.; C-pred +0.013 n.s.
  single n=144: Agent-local +0.079 [+0.050,+0.111] p=1e-5; +arm +0.075 p=2e-6; C-pred +0.048 p=.006.
  -> the overall Agent-local harm comes from single-stop tasks; multi-stop is n.s. (underpowered, n=53).
  Plan stats: multi 4.6 steps, 39% plans with >=2 trajectory steps (concat scoring matters); single 3.5 steps, 12%.
- NEXT: Step 2 code (groq_plans.py, run_agent.py --planner/--dataset, concat scoring).
- Step 2 DONE (2026-10-28 ~21:00): eval/groq_plans.py (GroqLLM cache results/groq_cache/, GroqAgent), run_agent.py
  --planner/--groq-model/--dataset/--rescore + concat scoring (backup run_agent.py.bak_decomp). Rescore of results/agent
  reproduces all 800 old rows exactly (old traj.jsonl kept as traj.jsonl.bak_pre_concat).
  Deviations: Groq temperature 0; enhancer max_tokens 300 (reasoning tokens count); enhancer calls that depend on
  RoboBrain output are made LIVE during the GPU run (cached), not pre-computed. Groq limit 8k tokens/min, ~2.2k/decomp.
  Agent-local ShareRobot concat (no GPU): multi +0.017 [-0.025,+0.056] n.s.; single +0.078 p=3e-5. Concat ~ last.
- Step 3 LAUNCHED: eval/run_decomp.sh, log logs/run_decomp.log (groq pre-pass: logs/groq_plans.log).
  Order changed: groq-plans in background || smoke-local + local-vabench on GPU; then smoke-groq, groq-vabench,
  groq-sharerobot, summary -> results/decomp/summary.md. Status: bash eval/progress.sh
  Resume after a restart: relaunch the same command (all steps skip finished ids / cached calls).

## USER INSTRUCTION 2026-10-28 22:40 — after run_decomp finishes
Rewrite paper/main.tex to a better standard (arXiv, no page limit): integrate the decomposition results
(new section + table), add more result figures and qualitative examples where they help, "but don't go
overboard". Use the research-writing skill; back up main.tex first; number-preservation check; rebuild clean.

## GROQ QUOTA (2026-10-28 23:30)
Free tier gpt-oss-120b = 200k tokens/DAY (TPD, ~2.2k per decomposition -> ~90/day). Cached: ShareRobot 127/127,
VABench-V 170/264 (94 missing). USER DECISION: check after midnight whether the limit resets; if not, stop the
Groq pre-pass and drop Groq VABench-V, keep everything else (Agent-local VABench-V runs in parallel); Groq ShareRobot
can still run from the cache.
USER (23:40): if Groq is dropped, KEEP everything already finished. Added run_agent.py --cached-only (groq planner runs
only samples with a cached plan). Coverage: ShareRobot 200/200 samples, VABench-V 199/300 -> Groq VABench-V reported
on that 199-sample subset, paired with B and Agent-local on the same ids (state this in the paper).
- 00:05 2026-10-29: Groq limit did NOT reset (rolling TPD; cache 305 rows). Groq pre-pass STOPPED; cache kept.
  run_decomp.sh will log "=== FAILED groq-plans" after local-vabench (expected). Follow-on eval/run_decomp_partial.sh
  (log logs/run_decomp_partial.log) waits for it, then smoke-groq, groq-vabench --cached-only, groq-sharerobot, summary.
  ANALYSIS NOTE: compare Agent-local vs Groq on VABench-V on COMMON ids (Groq subset), and also Agent-local on all 300.
- USER (00:10): do NOT mention the Groq API/quota issue in the paper. Describe Groq VABench-V neutrally as a fixed subset
  (first unique instructions in dataset order, n stated, no outcome-based selection); decomposer = gpt-oss-120b,
  temperature 0, as a method detail (no need to discuss the deprecated Llama model).
- 00:20 local-vabench DONE (300, 0 OOM, plans parsed 100%, 4.3 steps, 27% >=2 traj steps, 18.4 s/sample).
  VABench-V multi n=285 vs B: last dDFD -0.035 [-0.054,-0.015] p_holm=.005; +arm +0.002 n.s.;
  concat (HEADLINE) -0.018 [-0.038,+0.002] p=.28 (valid .91); C-pred -0.051 p=1e-5. single n=15 all n.s.
  -> by the pre-registered rule Agent-local does NOT improve multi-stop (concat CI includes 0); last-step gain is secondary.
- 01:10 groq-vabench DONE (206/300 cached-plan subset, 0 OOM, parsed 100%, 4.4 steps, 24% >=2 traj, 8.6 s/sample).
  COMMON ids, multi n=202 vs B (Holm over 6 conds): Groq concat -0.037 [-0.056,-0.017] p=.009 (PASSES rule);
  Groq last -0.042 p=.004; Groq +arm / concat+arm ~0 (arm prepend cancels gain). Local concat -0.006 n.s., last -0.020 n.s.
  TODO analysis: direct paired Groq-vs-local; C-pred on the same ids; Holm with pre-registered family (2 headline conds).

## FINAL RESULTS (2026-10-29 01:20) — results/decomp/report.md|json (eval/decomp_analysis.py), all on common ids
VABench-V multi n=202: Groq concat -0.037 [-0.057,-0.017] Holm(2) p=.0036 -> PASSES pre-registered rule;
  local concat -0.006 n.s.; Groq-local concat -0.029 p=.004; C-pred (single call) -0.041 p=9e-4 on same ids (>= Groq);
  arm prepend cancels gains (Groq concat+arm +0.001, local +0.012). FDE n.s. for all.
ShareRobot (replication) multi n=54: Groq concat +0.039 n.s. (p=.08), local +0.017 n.s. -> NOT replicated.
  single n=146: both agents worse (local +0.078, Groq +0.043, both p<1e-3); Groq-local -0.039 p=.04.
Plans: parsed 100% both; ~4 steps; 18-27% with >=2 traj steps; latency local 16.6-18.4 s, Groq 5.9-8.6 s; 2217 tok/plan.
INTERPRETATION: decomposition helps only where grounding-conditioning helps (VABench-V, anchor on object) and is
not better than single-call C-pred there -> consistent with the paper's anchor thesis. Strong planner > 3B decoder.
NEXT: paper rewrite (user instruction above). Keepalive stopped (3 pings).

## PAPER UPDATED (2026-10-29 01:30)
paper/main.tex (backup main.tex.bak_decomp_20261029): abstract/intro/contributions; Fig 1 pipeline+agent TikZ;
method paras Agent-Groq / multi-stop subset / concat scoring; setup + Holm family (2 planners); new Results 6.6
"Decomposition helps multi-stop tasks only where conditioning helps" (Table 5, Fig 6 fig_decomp, Fig 7
fig_decomp_examples; eval/figures_decomp.py); cost; discussion 7.2 retitled; limitation; conclusion; bib gptoss.
Removed the optional Llama router TBD. Number check: only 8.9 (abstract dup), 70, 3.3 (router TBD) removed.
12 pages, compiles clean (tectonic, env tex). upload/paper.pdf + paper_source.zip refreshed (backups *.bak_20261029).
Groq quota NOT mentioned (user). Remaining author items: acknowledgements TBD; rotate the Groq key (pasted in chat).
