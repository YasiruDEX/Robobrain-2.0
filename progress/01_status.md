# Status (2026-10-26)

## Done
- Env rb2eval (torch 2.5.1+cu124, transformers 4.50.0, bnb 0.45.5) - CUDA OK.
- HF access to BAAI/RoboBrain2.0-3B granted (user token in ~/.cache/huggingface/token).
- Benchmarks converted to data/*.jsonl: sharerobot_traj (200), sharerobot_afford (200),
  where2place (100), roborefit (300 seeded subsample of shard 0).
- Harness in eval/: rb2.py (8-bit, greedy, resized-frame scale), parse.py, metrics.py
  (+test_metrics.py, all pass), run_traj.py (6 conds), run_spatial.py (afford 4 conds, grounding,
  pointing), run_planner.py (keyword / rb2 / groq), summarize.py, analyze.py, run_all.sh.

## Running
- logs/setup_env.log: 3B weight download (watcher armed). Next: launch eval/run_all.sh detached.

## Notes
- Planner labels = benchmark provenance; keyword rules were written after seeing some
  instructions -> keyword baseline slightly optimistic (state in paper).
- Routing cost model: wrong route -> task output unusable (score 0); state as assumption.

## 2026-10-26 (later)
- run_all.sh finished 04:28 (traj 6 conds, afford 4 conds, grounding, pointing, planner keyword+rb2); rb2fs few-shot router added.
- results/report.md + figures (paper/figs/*.pdf via eval/figures.py) + paper/main.tex rewritten (7 pp, compiles
  with tectonic from env `tex`; number check: all decimals traced). Title change flagged for author approval.
- run_extra.sh running: C-goal (ShareRobot), VABench-V (300, 7 conds), base Qwen2.5-VL-3B (ShareRobot).
  Paper has \tbd{} for these. Guard lesson: start guard only AFTER a fresh watcher wrote logs/watcher.pid.
- Legacy app (scripts/utils.py) calls a UnifiedInference(load_in_8bit=...) API that exists in no upstream
  release -> documented in eval/README.md, not fixed.
