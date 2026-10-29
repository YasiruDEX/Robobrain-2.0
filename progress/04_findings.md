# Code/paper audit findings (2026-10-26)

1. Planner in code is Groq Llama-3.3-70B (backend.py, scripts/pipeline_chat.py, smart_chat.py); no local Qwen planner exists.
2. Propagation is free text truncated to 200 chars (pipeline_chat.py:503-508), not structured bbox injection.
3. Trajectory output gets a detected robot-arm point prepended (pipeline_chat.py:454) — affects ADE; not in paper.
4. No benchmark JSON, annotate_gt.py or metric code in either branch -> all paper numbers unreproducible.
5. GT built from RoboBrain demo predictions -> circular.
6. RoboBrain2.0-3B has NO thinking mode (rb_v2.0/inference_rb2.py: thinking only 7B/32B) -> thinking ablation in paper impossible on 3B.
7. Upstream default decoding do_sample=True, T=0.7 -> earlier results nondeterministic.
8. scripts/utils.py imports `inference` from upstream repo root, which is now RoboBrain 2.5 code -> fresh clone broken; must use rb_v2.0/inference_rb2.py.
9. Paper background describes RoboBrain 1.0 (A-LoRA/T-LoRA, SigLIP, ShareRobot phase-2) as 2.0.
10. Grounding IoU "improves" via propagation although grounding is step 1.
11. RoboBrain 2.0 trajectory prompt asks for up to 10 2D points; 2.5 asks for 3D (x,y,d) on 0-1000 scale — don't mix.
12. RoboBrain2.0-3B in 8-bit does NOT fit the 4 GB RTX 3050 Ti (OOM while loading, 3.68 GiB usable) -> paper's '8-bit on 4 GB' claim false unless CPU offload; we use 4-bit NF4.
