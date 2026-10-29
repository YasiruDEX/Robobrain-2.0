# Evaluation code for the paper

Everything reported in the paper is produced by this directory. The Flask/React app and
`scripts/` at the repository root are the earlier interactive demo; they are **not** used for
any number in the paper (see "Legacy demo" below).

## Setup

```bash
bash eval/setup_env.sh          # conda env rb2eval + BAAI/RoboBrain2.0-3B weights (gated: accept terms, `huggingface-cli login`)
python eval/convert_sharerobot_traj.py   # ShareRobot-Bench trajectory (200)
python eval/convert_spatial.py           # ShareRobot-Bench affordance (200), Where2Place (100), RoboRefIt subset (300)
python eval/convert_vabench.py           # VABench-V (300)
python eval/test_metrics.py              # unit tests for metrics and parsers
```

## Run

```bash
nohup setsid bash eval/run_all.sh   > logs/run_all.log 2>&1 &   # main experiments
nohup setsid bash eval/run_extra.sh > logs/run_extra.log 2>&1 & # C-goal, VABench-V, base Qwen2.5-VL-3B
python eval/analyze.py    # -> results/report.md, results/report.json
python eval/figures.py    # -> paper/figs/*.pdf
```

All runners are resumable: results are appended per (sample, condition) and finished keys are
skipped on restart.

## Protocol

- Model: RoboBrain2.0-3B, 4-bit NF4 (bitsandbytes), fp16 compute, `max_pixels = 512*28*28`,
  greedy decoding, `max_new_tokens = 256`. 8-bit does not fit a 4 GB GPU.
- Prompts: RoboBrain 2.0 templates (`RoboBrain2.0_lib/rb_v2.0/inference_rb2.py`); ShareRobot-Bench
  trajectory uses the benchmark's own wording. Conditioning text is inserted between the task and
  the output-format instruction (`rb2.build_prompt`).
- Coordinates: the model answers in pixels of its resized input; we divide by the resized size
  (`image_grid_thw * 14`) to get [0, 1] coordinates of the original image.
- Trajectory metrics (`metrics.py`): DFD and HD on raw polylines; `rmse_idx` = ShareRobot-Bench's
  index-matched RMSE; RMSE/ADE on arc-length-resampled polylines (20 points); FDE on endpoints.
- Statistics (`summarize.py`): paired differences vs. B on samples valid in both, 10k-sample
  bootstrap 95% CI, Wilcoxon signed-rank, Holm correction across conditions per metric.

| Condition | Text injected into the trajectory prompt |
|---|---|
| B | nothing |
| C-text | `The target object "<obj>" is located in the image.` |
| C-pred | `... is located at bounding box [x1, y1, x2, y2] ...` (model's own grounding) |
| C-point | `... is located at point [x, y] ...` (model's own pointing) |
| C-shuffle | C-pred box taken from a different image |
| C-goal | `The motion should end at the "<place>", located at bounding box [...]` |
| C-gtend | ground-truth trajectory endpoint as a point (oracle) |

## Legacy demo

`backend.py`, `frontend/` and `scripts/` route tasks with Groq Llama-3.3-70B and call a
`UnifiedInference(model_path, load_in_8bit=...)` interface that does not exist in either
upstream RoboBrain release (the upstream repo root is now RoboBrain 2.5; the 2.0 class lives in
`RoboBrain2.0_lib/rb_v2.0/inference_rb2.py` and takes `(model_id, device_map)`). The demo is kept
for reference only.
