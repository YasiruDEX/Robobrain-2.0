# CHECKPOINT 2 — 2026-10-26 ~08:50 (user: stop after VABench-V, has a separate task)

- VABench-V (RoboBrain2.0-3B, 300 x 7 conds): FINISHED -> `results/vabench/summary.md`.
  (`eval/stop_after_vabench.sh` stopped run_extra.sh right after the VABench-V summary; log line
  `=== STOPPED for checkpoint after VABench-V`.)
- NOT STARTED: base Qwen2.5-VL-3B (weights ready) and the Agent-local benchmark (`run_agent.sh`
  was cancelled while still waiting; nothing from it is on disk).
- Resume both, unattended, in this order (run_agent.sh waits for run_extra.sh by itself):
  ```bash
  cd ~/Documents/Robobrain-2.0
  (nohup setsid bash eval/run_extra.sh > logs/run_extra.log 2>&1 &)   # skips everything done -> Qwen
  (nohup setsid bash eval/run_agent.sh > logs/run_agent.log 2>&1 &)   # starts when run_extra ends
  ```
  Then arm one watcher on `logs/run_agent.log` for `ALL AGENT JOBS FINISHED|FAILED|Traceback`.
  No guard. Expected: Qwen ~1 h, agent ~1-1.5 h.
- After both: paper pass per `progress/03_next_steps.md` ("Pending paper edits") + new Agent-local
  subsection.

---

# UPDATE 2026-10-26 07:55 — runs continue UNATTENDED (guard removed by user request)

`eval/run_extra.sh` was relaunched 07:25 and keeps running after the Claude tab is closed
(no guard). At 07:50 VABench-V was at 688/2100; then base Qwen (~1 h). Expected end ~09:50.

**ALSO QUEUED (08:03): `eval/run_agent.sh`** -> `logs/run_agent.log`. Starts automatically when
run_extra.sh ends: benchmarks the ORIGINAL app layer (scripts/pipeline_chat.py decomposer + enhancer +
multi-turn history + `(Context: prev[:200])` + robot-arm prepend) with Groq replaced by RoboBrain's own
Qwen decoder ("Agent-local"). Smoke 3 -> ShareRobot traj 200 (+summary vs B, C-pred in results/agent) ->
affordance 200. End marker `ALL AGENT JOBS FINISHED`. ~1-1.5 h after run_extra. Paper needs a new
subsection for it (planner = original system, now local).

**On resume (Claude):**
1. `grep -E "ALL JOBS FINISHED|FAILED|Traceback|Killed|out of memory" logs/run_extra.log`
   - `ALL JOBS FINISHED` -> skip §2 below, go straight to §3 and `progress/03_next_steps.md`
     ("Pending paper edits").
   - runner still alive (`ps -eo pid,args | grep "bash eval/run_extra.sh"`) -> arm one watcher, wait.
   - dead without the end marker -> relaunch per §2 (resumable). Do NOT start `eval/guard.sh`
     (user: closing Claude must not stop GPU jobs).
2. The C-goal results and six-condition Holm p-values are already in `paper/main.tex`.

---

# CHECKPOINT — 2026-10-26 06:50

To resume, open Claude Code in `~/Documents/Robobrain-2.0` and say:

> **Continue the research from `progress/CHECKPOINT.md`.**

All work was stopped cleanly; nothing is running and the GPU is free. Every finished
(sample, condition) result is on disk, and the runner skips finished work, so nothing is redone.

---

## 1. Job state

| Job | State | Saved |
|---|---|---|
| `eval/run_all.sh` (ShareRobot traj 6 conds, affordance, grounding, pointing, planner) | FINISHED | `results/traj`, `results/spatial`, `results/planner` |
| Few-shot local router (`rb2fs`) | FINISHED | `results/planner/planner.jsonl` |
| C-goal on ShareRobot | FINISHED | `results/traj/goal_steps.jsonl`, `traj.jsonl` (1400 lines) |
| VABench-V (300 x 7 conds) | UNFINISHED: pass 1 + goal pass done (300/300), trajectories 80/2100 | `results/vabench/` |
| Base Qwen2.5-VL-3B on ShareRobot (7 conds) | NOT STARTED (weights downloaded, 7.1 GB) | `weights/Qwen2.5-VL-3B-Instruct` |

Estimated remaining GPU time: ~1 h VABench-V + ~1 h Qwen.

## 2. Resume procedure (for Claude)

1. Launch the runner (it skips finished work; the Qwen download step is a no-op now):
   ```bash
   cd ~/Documents/Robobrain-2.0 && (nohup setsid bash eval/run_extra.sh > logs/run_extra.log 2>&1 &)
   ```
2. Arm ONE zero-token watcher **first** (it writes `logs/watcher.pid`), watching
   `logs/run_extra.log` for `ALL JOBS FINISHED|FAILED|Traceback|Killed|out of memory|STOPPED`.
3. Only after `logs/watcher.pid` holds the live watcher PID, start the guard:
   `(nohup setsid bash eval/guard.sh logs/run_extra.log > logs/guard.log 2>&1 &)`.
   (Starting the guard with a stale PID kills the run — happened once.)
4. No subagents, no polling (user preference). Kill processes by PID, never with
   `pkill -f <pattern>` from a shell whose own command line contains the pattern.

## 3. When the runs finish

1. `python eval/analyze.py && python eval/figures.py && python eval/qualitative.py`
   (env: `/home/jayamadu/miniconda3/envs/rb2eval/bin/python`, run from `eval/`).
2. Edit `paper/main.tex`:
   - Fill the three `\tbd{}` blocks: C-goal row of Table 2 + "Grounding the end location"
     paragraph; "Generality" subsection (VABench-V and Qwen).
   - **Holm family is now SIX conditions** (C-goal added): replace every p-value in Table 2 and
     the text with the values from `results/traj/summary.json` (`*_p_holm`), and change
     "family of five"/"five conditions" to six. Re-check the abstract's p = 0.012.
   - Update the Figure 1 caption once it has 3 panels.
3. Number check (research-writing skill §6): every decimal in `main.tex` traceable to
   `results/report.md` or `results/*/summary.md`.
4. Compile in the scratchpad: `tectonic -k main.tex` (env `tex`), check 0 overfull/undefined.
5. Update `progress/00_HUB.md` (results, decision) and `progress/01_status.md`.

## 4. Results so far (ShareRobot-Bench trajectory, RoboBrain2.0-3B, 4-bit, n = 200)

| Cond | DFD | dDFD vs B [95% CI] | note |
|---|---|---|---|
| B | 0.241 | — | |
| C-text | 0.258 | +0.017 [+0.003, +0.031] | name only hurts |
| C-pred | 0.282 | +0.039 [+0.016, +0.061] | own box hurts (headline) |
| C-point | 0.266 | +0.024 [+0.004, +0.044] | |
| C-shuffle | 0.322 | +0.080 [+0.057, +0.105] | model uses coordinates |
| C-goal | 0.255 | +0.014 [−0.004, +0.032] n.s. | end-location box: harm mostly removed, no gain |
| C-gtend | 0.238 | −0.003 n.s.; FDE −0.045 | oracle moves endpoint only |

- Endpoint attraction under C-pred −0.076; object box contains GT endpoint 50/200 (25%),
  goal box 70/200 (35%).
- Affordance: IoU≥0.5 13.5% → 20.5% with C-pred (McNemar p = 0.016).
- Grounding (RoboRefIt) mIoU 0.580; Where2Place 68.00; routing: keyword 68.9%, RoboBrain
  zero-shot 45.6%, few-shot 49.2%.
- Thesis (title approved): grounding-conditioning helps when the output lies on the grounded
  object (affordance) and hurts when it does not (trajectory).

## 5. Paper state

`paper/main.tex` — 7 pages, compiles; approved title; repo URL
`https://github.com/YasiruDEX/Robobrain-2.0`; figures in `paper/figs/` (traj deltas, mechanism,
planner, qualitative). Open items: the `\tbd{}` blocks above; bibliography spot-check
(arXiv API works over **https** only: `https://export.arxiv.org/api/query?id_list=...`).

## 6. Author-only (not for Claude)

- Acknowledgements / course note (`\tbd{acknowledgements ...}` in `main.tex`).
- Approve pushing `eval/`, `results/`, `paper/` to the YasiruDEX repo (`results/` is gitignored;
  keep `weights/`, `data/` out).
- arXiv endorsement (cs.RO), course-policy check, co-author sign-off.
- Rotate the old Groq key (in git history) and the HF token pasted in chat.
- Optional: `GROQ_API_KEY` for the Llama-3.3-70B router comparison.
