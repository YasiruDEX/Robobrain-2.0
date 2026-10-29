# Research hub: RoboBrain 2.0 grounding-conditioned trajectory (arXiv)

Status (2026-10-29 01:30): decomposition study DONE and integrated; paper 12 pp compiles clean. Nothing running.
Paper rewritten with all results (paper/main.tex, 9 pp, compiles clean; backup main.tex.bak_rewrite_20261026).
Next: author review of the rewrite + author-only items (see 03_next_steps.md). Nothing is running.

## Nodes
- 01_status.md: what is done / running, with dates
- 02_benchmarks.md: public benchmark survey (sources, formats, metrics, reported numbers)
- 03_next_steps.md: ordered task list
- 04_findings.md: code/paper discrepancies found while auditing (must be fixed in paper)
- 05_decomposition_plan.md: decomposition study DONE 2026-10-29 01:16 (results/decomp/report.md); next: paper rewrite

## Core question
Does conditioning RoboBrain 2.0 trajectory prediction on its own grounding output
(structured bbox/point injected in the prompt) reduce trajectory error, and is the gain
due to spatial content (vs. prompt length / object re-mention)?

## Conditions (trajectory)
B single-step | C-pred (own grounding bbox) | C-point (own pointing) | C-gt (oracle target)
| C-shuffle (bbox from another image) | C-text (object re-mention, no coords)

## Decision rule (pre-registered)
- If C-pred beats B with paired bootstrap 95% CI excluding 0 on DFD/RMSE -> positive paper.
- If C-pred ~ B but C-gt beats B -> paper about grounding error as bottleneck (error-propagation analysis).
- If neither beats B -> negative-result / analysis paper. Report honestly.
- C-shuffle must be worse than C-pred for the "spatial content" claim.

## RESULT + DECISION (2026-10-26 04:30, results/report.md)
ShareRobot-Bench traj, 3B 4-bit, n=200: B DFD 0.241. C-pred WORSE (+0.039 [+.016,+.061], Holm p=.012);
C-text +0.017; C-point +0.024; C-shuffle +0.080; C-gtend DFD -0.003 n.s. but FDE -0.045 (p~1e-6).
-> Decision rule: neither C-pred nor C-gt beats B on DFD => ANALYSIS / NEGATIVE-RESULT paper.
Affordance: C-pred mIoU +0.018 n.s., IoU>=.5 0.135->0.205; C-shuffle -0.095. Grounding RoboRefIt mIoU .58;
Where2Place 68.0. Local rb2 zero-shot router 45.6% (traj 2.5%), keyword 68.9% (paper claimed 90%).
Caveat: our B DFD 0.241 << reported 7B 0.5512 -> protocol not identical; do not compare directly.
Mechanism (analyze.py): conditioning DOES attract the endpoint to the injected location (C-pred -0.076),
but only 25% (50/199) of grounded target boxes contain the GT trajectory endpoint -> wrong anchor.
Affordance IoU>=0.5: C-pred 22 gained / 8 lost, McNemar p=0.016 (contact region lies ON the object).
Thesis: grounding-conditioning helps when the output lies on the grounded object (affordance) and
hurts when it doesn't (trajectory goal != manipulated object). Few-shot local router 49.2% (merged 62.5%).

## Scope (user decision 2026-10-26)
- RoboBrain2.0-3B ONLY. The tech report gives no 3B numbers, so our condition B is the
  reference; 7B DFD 0.5512 (ShareRobot-Bench Traj, Table 3) is only a loose sanity anchor.
- Primary benchmark: BAAI/ShareRobot-Bench trajectory (200 samples), benchmark-native prompt
  wording; headline metric DFD (+ HD, official index-matched RMSE as rmse_idx).

## Conventions
- Greedy decoding (do_sample=False), 8-bit, fixed max_pixels; every raw output logged to JSONL.
- All metrics on coordinates normalised to [0,1] of the ORIGINAL image.
- Env: /home/jayamadu/miniconda3/envs/rb2eval ; weights: weights/RoboBrain2.0-3B

## UPDATE 2026-10-26 14:10 — generality + agent (results/report.md, sections 4-5)
- VABench-V: sign reverses, C-pred dDFD -0.053 (p=4e-6); object box contains GT START 88% (ShareRobot 10%);
  start error 0.272 -> 0.137. Thesis holds: helps when output lies on grounded object.
- Qwen base: valid only 53% under B; all priors help incl. C-shuffle (-0.062, valid 68%) -> format cue, indicative only.
- Agent-local (original app layer, local 3B planner): traj DFD +0.063, afford mIoU -0.075, IoU>=.5 9.5%, 16.6 s/sample.
