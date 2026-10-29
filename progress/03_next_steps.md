# Next steps (ordered)

1. RUNNING since 02:57 (4-bit NF4; sanity passed 3/3 boxes, 18/18 trajectories).
   eval/guard.sh stops the run if the Claude watcher (logs/watcher.pid) dies (session ended).
   Resume (skips finished work): `cd ~/Documents/Robobrain-2.0 && nohup setsid bash eval/run_all.sh > logs/run_all.log 2>&1 &`
   then `nohup setsid bash eval/guard.sh > logs/guard.log 2>&1 &` and re-arm a watcher that writes logs/watcher.pid.
2. `python eval/analyze.py` -> results/report.md; apply decision rule in 00_HUB.md.
3. Optional generalisation: base Qwen2.5-VL-3B-Instruct, same trajectory conditions (3B-only scope).
4. Figures (load dataviz skill): per-condition dDFD with CIs; error-propagation bars; planner confusion.
5. Paper rewrite -> paper/main.tex: correct RoboBrain 2.0 background, new related work,
   new results, limitations; fix refs (SigLIP, GIGA, RoboVLMs id, RoboBrain 2.5); one repo URL.
6. Author-only: arXiv endorsement (cs.RO), course policy check, co-author sign-off,
   rotate Groq key (in git history) and the HF token pasted in chat, optional GROQ_API_KEY for router.

## Pending paper edits (do AFTER run_extra finishes — user 2026-10-26)
- DONE already: C-goal row + paragraph; Holm family = 6 (p-values in abstract, contributions, Table 2, text);
  C-point now n.s. (p=0.063). Backup: paper/main.tex.bak_holm6_20261026.
- Discussion lines ~493-495 ("should therefore ground the end location"): soften — C-goal removed most of
  the penalty (+0.014 n.s.) but gave no gain; say gains need more accurate goal grounding.
- Fill "Generality" \tbd (VABench-V, Qwen); Fig 1 caption; rerun analyze/figures/qualitative; number check; compile.

## VABench-V finding (2026-10-26 10:05) — paper edits wait until run_extra + run_agent BOTH finish (user)
- VABench-V reverses the sign: C-pred dDFD -0.053, C-text -0.047, C-point -0.062 (all better), C-shuffle 0.000,
  C-gtend -0.052, C-goal -0.021 n.s. (results/vabench/summary.md).
- Mechanism (scratch analysis, not yet in analyze.py): VABench-V GT paths START on the object
  (object box contains GT start 88%, GT end 4%); ShareRobot starts elsewhere (start 10%, end 25%).
  Start-point error B -> C-pred: VABench-V 0.272 -> 0.137 (fixed), ShareRobot 0.083 -> 0.166 (pulled off).
  => Consistent with thesis: conditioning helps when the output (here the path start) lies on the grounded object.
- TODO in paper pass: add start-containment + start-error rows to eval/analyze.py for both datasets;
  Generality subsection reports this as support (not contradiction) of the thesis.

## Agent-local fixes (2026-10-26 12:10) — relaunched 12:12
- Port bug: decomposer max_new_tokens was 600 (original app 1500) -> truncated plans. Fixed to 1500.
- User decision: repair local decoder's markdown "\_" escapes before JSON parse; logged per sample as
  `plan_repaired` -> paper must state this deviation + repair rate.
- Old outputs moved to results/_stale_agent_600tok/; backups eval/run_agent.{py,sh}.bak_600tok.
- Smoke 3/3 parsed (1 repaired), all valid. Qwen base run FINISHED 12:07 (results/traj_qwen).
- 13:11 agent-traj FINISHED (results/agent/summary.md). agent-afford OOM'd at sample 2420 (4 GB GPU, long multi-turn
  context). Fix: retry once after empty_cache, else record {"oom": true, iou 0} and continue; expandable_segments.
  Relaunched 13:13 via eval/run_afford.sh (log logs/run_agent.log, end marker ALL AGENT JOBS FINISHED).
  Paper must report the OOM count. Then do the full paper pass (user: rewrite paper after inference ends; add examples).

## DONE 2026-10-26 14:10 — paper pass
- analyze.py: new sections 4 (anchor/start error, Qwen validity McNemar) and 5 (Agent-local); figures regenerated;
  new eval/qualitative2.py -> paper/figs/fig_examples.pdf (VABench-V start anchor + Agent-local plan).
- main.tex rewritten: abstract, intro, contributions (+Generality, +Agent), Agent-local method, Table 3 (generality,
  table*), Agent row in Table 4, new Agent subsection + Fig 4, cost, discussion (softened goal claim), limitations
  (+Qwen validity, +agent planner), conclusion. Number check: all new numbers trace to results (derived: 0.91, 9.5, 98.5, 8.9).
- Remaining author-only: acknowledgements \tbd; optional Llama-3.3-70B router \tbd; repo push approval; arXiv items.
