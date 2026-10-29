# 02 - Public benchmarks for RoboBrain 2.0-3B (verified 2026-10-26)

Legend: VERIFIED = read from primary source (HF API/README, GitHub, arXiv PDF text). UNVERIFIED = not confirmed.
All HF repos below are `gated: False` (no token) unless stated. Model repos BAAI/RoboBrain2.0-3B and -7B are `gated: auto` (HF login + click-to-accept; apache-2.0).

## 0. Key findings / corrections to the brief

1. **The RoboBrain 2.0 tech report (arXiv:2507.02029 v5, 14 Sep 2025) reports NO 3B numbers at all.** Only 7B and 32B (Table 2 p.13, Table 3 p.14 of the PDF). Text grep for "3B" finds nothing. The 3B model card on HF is not readable without auth; 7B card has no benchmark table. So there is nothing to "reproduce" for 3B: compare against 7B as an upper reference. (3B numbers in any other source: UNVERIFIED / not found.)
2. The report evaluates only these of our targets: **Where2Place, ShareRobot-Bench Affordance, ShareRobot-Bench Trajectory (DFD only)**, plus RefSpatial-Bench (pointing, not requested). It does **not** report VABench-V, VABench-P/Point, RoboRefIt, RefCOCO, HD or RMSE.
3. **VABench-P / VABench-V are not RoboPoint benchmarks.** They come from the FSD paper "From Seeing to Doing" (Yuan et al., arXiv:2505.08548) and are re-used by Embodied-R1 (arXiv:2508.13998). RoboPoint's benchmarks are Where2Place (and RoboRefIt use). HF release by Embodied-R1 authors (IffYuan), March 2026. RoboBrain 2.0 never evaluates on them.
4. **The public ShareRobot-Bench is only 200 trajectory + 200 affordance samples**, not the 870 trajectory / 522 affordance test split described in RoboBrain 1.0 (arXiv:2502.21257 sec 3, "6000 train / 870 test" trajectory; "6000 / 522" affordance). Whether the tech report's Table 3 used the 200+200 BAAI/ShareRobot-Bench is UNVERIFIED but likely (FlagEvalMM PR #84 points at it; the HF README says "Testset of ShareRobot ... refer to FlagEvalMM PR 84 for evaluation code").
5. **Official ShareRobot-Bench eval code is NOT merged into FlagEvalMM main** (PR #84 `xiaow190:main`, state open, unmerged as of today; `tasks/share_robot` absent from main tree). Code is quoted below from the PR diff (commit 24655d28).
6. RoboBrain 1.0 and 2.0 trajectory normalisation differ: 1.0 paper says waypoints "normalized to [0, 1000)"; the FlagEvalMM PR (the 2.0 evaluation framework) normalises to [0,1] by image width/height. DFD values in the 2.0 table (0.2-1.2) are consistent with the [0,1] scale.

## 1. TRAJECTORY

### 1a. ShareRobot-Bench trajectory (BEST CHOICE; same benchmark RoboBrain 2.0 reports)
- Repo: https://huggingface.co/datasets/BAAI/ShareRobot-Bench (apache-2.0, not gated, last modified 2025-08-17). README is 1 line: "Testset of ShareRobot."
- Files: `trajectory/test.json` (120,179 B), `trajectory/images/*` (200 images, 22.4 MB total); `affordance/test.json` (79,852 B), `affordance/images/*` (200 images, 36.6 MB). Whole repo ~ 59 MB. Downloaded only the two JSONs to `data/raw/sharerobot_bench_traj/test.json` and `data/raw/sharerobot_bench_afford/test.json`.
- Size: **200 samples** (VERIFIED by loading json). Number of GT points per sample: 3 pts:13, 4:72, 5:110, 6:3, 7:2 (so GT is 3-7 points, prompt says "up to 10").
- Image sizes vary (sampled 10 imgs: 640x480, 224x224, 424x240, 256x256, 640x360, 640x512). Max GT coordinate 590 -> **absolute pixels in the original image frame**.
- Schema (one record, VERBATIM from trajectory/test.json):
```json
{"id": 12329935,
 "image": ["images/12329935_167.jpg"],
 "conversations": [
  {"value": "<image>\n You are a Franka robot using the end effector control. The task is assemble one_leg. Please predict up to 10 key trajectory points to complete the task.", "from": "human"},
  {"from": "gpt", "value": "[(170, 47), (176, 77), (180, 84), (183, 90), (172, 84)]"}]}
```
  GT answer is a Python-literal string of (x, y) tuples in pixels. No width/height stored; the evaluator reads them from the image file.
- Full ShareRobot (training source, for README schema): https://huggingface.co/datasets/BAAI/ShareRobot (not gated, 13,501 files, 17k downloads; license field empty = UNVERIFIED). README schema for trajectory: `{"id","meta_data":{"original_dataset","original_width","original_height"},"instruction","points":[[x,y],...]}` with absolute pixel coords (e.g. [265.45, 120.0] on a 640x480 bridge frame). 6,870 trajectory images and 6,522 affordance images in total. Only use for train/other data; ShareRobot-Bench is the test split.
- Mirror (third party, do NOT treat as primary): `IffYuan/sharerobot_trajectory` (200 rows, split called "train", 22.5 MB parquet, same ids; images there appear resized to 224x224 in the sampled row, so coordinates there may NOT match BAAI originals - use BAAI/ShareRobot-Bench). Other mirrors `FedorX8/ShareRobot-test`, etc.: UNVERIFIED.
- Metrics (from FlagEvalMM PR #84, `tasks/share_robot/trajectory/evaluate.py`, https://github.com/flageval-baai/FlagEvalMM/pull/84/files ; raw: https://github.com/flageval-baai/FlagEvalMM/raw/24655d281c9d082592942cd18168cca5d0951fdb/tasks%2Fshare_robot%2Ftrajectory%2Fevaluate.py):
  - Parse model answer with `ast.literal_eval`; fallback regex. For each point: if x>1 or y>1 -> divide by (image width, height) -> coordinates in [0,1]; else treated as already normalised. GT is parsed with the same function (so GT pixel -> [0,1]).
  - Empty prediction -> `[(0,0)]`.
  - **DFD**: discrete Frechet distance (recursive DP, Euclidean `cdist`) between pred and GT point arrays, in normalised units, no resampling.
  - **HD**: `max(directed_hausdorff(P,Q)[0], directed_hausdorff(Q,P)[0])` (scipy), normalised units.
  - **RMSE**: sqrt(mean of squared Euclid distances) of index-matched points over `min(len(P), len(Q))` (truncation, no interpolation); 0.0 if empty.
  - Result = mean over samples of each; failures in a sample are skipped (not penalised, except empty->(0,0)). Keys: `average_discrete_frechet_distance`, `average_hausdorff_distance`, `average_root_mean_square_error`.
  - Prompt used (`process.py`): question text with `Please predict up to 10 key trajectory points to complete the task.` replaced by `... Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.` Model is asked for pixel (or normalised) coordinates; evaluator accepts either.
  - Config: `tasks/share_robot/trajectory/share_robot_trajectory.py` (dataset_path BAAI/ShareRobot-Bench, split test, VqaBaseDataset, BaseEvaluator eval_func evaluate.py).
  - Caveat: the PR code prints per item and uses `eval` in the fallback; copy the maths, not the code verbatim. Whether the exact tech-report run used this version: UNVERIFIED.
  - RoboBrain 1.0 paper (arXiv:2502.21257, p.8 sec 5.2 + Table 3): metrics DFD, HD, RMSE; waypoints normalised to [0,1000) following Qwen2-VL. RoboBrain-1.0 values there: DFD 0.109 / HD 0.010 / RMSE 0.091 (final row; HD 0.010 vs 0.157-0.171 in other rows looks suspicious, treat as the paper's claim only) - on its 870-image test split, not the 200 public one.
- RoboBrain 2.0 reported numbers (tech report Table 3, PDF page 14, "ShareRobot-Bench Traj. (DFD down)"): **7B-2.0 = 0.5512; 32B-2.0 = 0.2368; 3B = not reported.** Baselines: RoboBrain-7B-1.0 0.6248, Qwen2.5-VL-72B 0.5034, GPT-o4-mini 0.5726, Gemini-2.5-Pro 0.7666, GPT-4o 0.6850, Qwen2.5-VL-32B 0.9222. HD / RMSE not reported for 2.0.
- Compute: 200 images, one forward each; trivial for 4 GB with 3B in 4-bit/bf16 offload.

### 1b. VABench-V (visual trace) - NOT reported by RoboBrain 2.0; optional extra
- Repo: https://huggingface.co/datasets/IffYuan/vabench-v (not gated; licence field empty = UNVERIFIED; made by Embodied-R1 authors; README of Embodied-R1 dated 2026-03 announces it). Split named `train` (misnomer, it is the test set), **300 rows**, 27.0 MB parquet.
- Schema (datasets-server row 0): `{"id":"fsd_pointbridge_17150","image":<640x480 image>,"problem":"You are currently a robot performing robotic manipulation tasks. The task instruction is: Move the yellow block in the middle of the table. Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. You must provide the points in the order of the trajectory, and the number of points must be 8.","answer":[[158,59],[164,64],...,[260,132]],"data_type":"fsd_visual_trace"}` - absolute pixels, 8 points.
- Metric (Embodied-R1 `eval/hf_inference_vabench_visual_trace.py`, https://github.com/pickxiguapi/Embodied-R1/blob/main/eval/hf_inference_vabench_visual_trace.py): both pred and GT scaled by `(x/width)*(1000-1)`, `(y/height)*(1000-1)` (i.e. 0-999 space); both resampled by linear interpolation (`interp1d`) to `max(len(pred), len(gt))` points; RMSE = sqrt(mean(dx^2+dy^2)), MAE = mean((|dx|+|dy|)/2). Samples with <=1 point are skipped (None). Paper also reports an LLM score (not reproducible offline).
- Reference numbers (Embodied-R1 paper Table 3, PDF page 8; RoboBrain **1.0** not 2.0): RoboBrain RMSE 121.6 / MAE 103.8 / LLM 4.5; Embodied-R1 77.8 / 45.0 / 7.3; FSD 78.3/63.4/6.2; GPT-4o 136.1/113.5/4.4. No RoboBrain 2.0 number exists in a primary source (UNVERIFIED if any).
- Scale differs from ShareRobot DFD (0-999 vs 0-1), do not compare across benchmarks.

### 1c. Other 2D trajectory benchmarks reported by RoboBrain 2.0
None. Only ShareRobot-Bench Traj (Table 3). (Checked full text of the tech report for "VABench", "trace", "trajectory".)

## 2. GROUNDING

### 2a. RoboRefIt - not reported by RoboBrain 2.0 (used by RoboPoint-style/Embodied-R1 pointing evals)
- Mirror used by Embodied-R1 authors: https://huggingface.co/datasets/IffYuan/Roborefit (not gated, licence empty = UNVERIFIED). Split `test`, **2000 samples**, 2 parquet shards, **383 MB download** (703 MB uncompressed). Over my 50 MB limit so not downloaded. Schema (datasets-server): `{"question_id":0,"image":<640x480>,"question":"will you please pass me the glue stick","bbox":[175,168,233,217]}`; bbox = absolute pixel [x1,y1,x2,y2] (640x480 frame).
- Annotation-only alternative (1.4 MB, downloaded to `data/raw/roborefit_embodiedr1_json/roborefit_test.json`): https://github.com/pickxiguapi/Embodied-R1/blob/main/eval/roborefit_test.json, 2000 records with keys `id, problem, image ("image_0.png"), instruction, instruction_2, bbox [175.0,168.0,233.0,217.0]`. Images are NOT in the repo; they must come from IffYuan/Roborefit parquet.
- Original dataset: VL-Grasp (RoboRefIt) - original repo/licence UNVERIFIED (not checked).
- Metric in Embodied-R1 eval (`eval/hf_inference_roborefit.py`): model outputs points; per-sample score = fraction of predicted points inside GT bbox; reported = mean (also overall point accuracy). This is a pointing-accuracy protocol, not box IoU. For RoboBrain 2.0 you could instead ask for a bbox and use IoU (see RefCOCO).
- Reference (Embodied-R1 Table 2, p.8): RoboBrain-1.0 10.10, Qwen2.5-VL 74.90, Embodied-R1 85.58. RoboBrain 2.0: not reported anywhere verified.
- 2000 images x 383 MB: feasible on disk and compute (can subsample first 200-500).

### 2b. RefCOCO val (FlagEvalMM evaluates it)
- FlagEvalMM `tasks/refcoco/refcoco_val.py`: dataset_path `lmms-lab/RefCOCO`, split `val` (8,811 samples; split sizes val 8811 / test 5000 / testA 1975 / testB 1810; HF metadata num_bytes val ~1.55 GB). Not gated. Licence: UNVERIFIED. Fields: question_id, image, question, answer, segmentation, bbox, iscrowd, file_name.
- Metric: `tasks/refcoco/evaluate.py` (https://github.com/flageval-baai/FlagEvalMM/blob/main/tasks/refcoco/evaluate.py): IoU, ACC@0.1/0.3/0.5/0.7/0.9, Center_ACC; `optimize_bbox` for "qwen2.5" renormalises boxes outside [0,1] by dividing by image width/height (so Qwen2.5-VL absolute pixel boxes are handled).
- RoboBrain 2.0 reports no RefCOCO number. Small subset recommended (stream first 500 of val), full val 1.5 GB is acceptable disk-wise.
- Smaller alternative mirror `jxu124/refcoco` validation parquet is 3.0 MB but has no images (image paths only; UNVERIFIED usable).

## 3. POINTING

### 3a. Where2Place (BEST CHOICE; reported by RoboBrain 2.0)
- Two HF copies:
  - Original: https://huggingface.co/datasets/wentao-yuan/where2place (RoboPoint authors, apache-2.0, not gated, folders images/, masks/, `point_questions.jsonl`, `bbox_questions.jsonl`; 100 images).
  - FlagEval (BAAI) copy used by FlagEvalMM and thus likely by RoboBrain 2.0: https://huggingface.co/datasets/FlagEval/Where2Place (not gated, licence empty, single parquet `data/test-00000-of-00001.parquet`, **4.36 MB**, **100 samples**; README says adapted from wentaoyuan/RoboPoint). Fields: `question_id (str), question, question_type ("seen"/"unseen"), image, mask`.
- Split: 100 total = 70 seen + 30 unseen (counted from point_questions.jsonl `category`). RoboBrain 2.0 footnote: "test set included several incorrect cases, which were manually screened and excluded" (which ones: UNVERIFIED), so numbers are on slightly fewer than 100.
- Schema, original `point_questions.jsonl` row 0 (downloaded to `data/raw/where2place_robopoint/`): `{"question_id": 0, "image": "00.jpg", "text": "Identify several spots within the vacant space that's between the two mugs. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.", "category": "unseen"}`; GT = binary mask image per question (masks/NN.jpg).
- Coordinates: RoboPoint prompt asks **normalised 0-1**. FlagEvalMM has two variants: `where2place` (0-1 prompt) and `where2place_norm1k` (0-1000 prompt, `text2pts_norm1k`). Which one was used for RoboBrain 2.0 numbers: UNVERIFIED. Parser `text2pts` also accepts absolute integer pixels (ints are treated as pixels, floats as normalised) and bboxes (expanded to all pixels inside).
- Metric: https://github.com/flageval-baai/FlagEvalMM/blob/main/tasks/where2place/evaluate.py and `flagevalmm/evaluator/point_utils.py::calculate_mask_score`: per sample score = fraction of predicted points that land on mask>0 (out-of-range points count as misses; no points -> 0); accuracy = mean over samples x100, also reported per `question_type` (seen/unseen). Only the last line of the answer is parsed.
- Embodied-R1 variant: `eval/hf_inference_where2place.py` same idea (points in mask, `x<=1` normalised else pixel).
- Official RoboPoint eval code: could not list repo tree via API (rate/response issue); path in wentaoyuan/RoboPoint UNVERIFIED.
- RoboBrain 2.0 reported (Table 3, PDF p.14, "Where2Place*"): **7B: Seen 64.33 / Unseen 61.88 / All 63.59; 32B: 73.95 / 72.74 / 73.59; 3B not reported.** Baselines: RoboBrain-7B-1.0 54.58/49.45/53.04, Qwen2.5-VL-72B 35.74/49.65/39.92, Gemini-2.5-Pro 42.92/41.13/42.38. RoboPoint 46.01 and Embodied-R1 69.50 on their own run (Embodied-R1 Table 2).
- Cost: 4.4 MB. Run first.

### 3b. VABench-Point (VABench-P)
- Repo: https://huggingface.co/datasets/IffYuan/VABench-P (not gated, licence UNVERIFIED), split `test`, **300 samples**, 27.1 MB. Schema: `{"id":"fsd_pointbridge_17150","problem":"You are currently a robot performing robotic manipulation tasks. The task instruction is: Move the yellow block in the middle of the table. Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved.","image":<640x480>,"mask":<640x480 mask>}`. Metric: fraction of predicted points inside mask (`eval/hf_inference_vabench_point.py`). Not reported by RoboBrain 2.0. Embodied-R1 Table 2: RoboBrain-1.0 7.00, RoboPoint 19.09, FSD 61.82, Embodied-R1 66.00. A third-party mirror `thomas-yanxin/vabench-p-mirror` etc. exists (ignore).
- Note: it is NOT from RoboPoint; origin is FSD (arXiv:2505.08548) per Embodied-R1 citations.

## 4. AFFORDANCE

### 4a. ShareRobot-Bench affordance (BEST CHOICE; reported by RoboBrain 2.0)
- Same repo as 1a: `affordance/test.json` (79.9 KB) + 200 images (36.6 MB). **200 samples**. Downloaded to `data/raw/sharerobot_bench_afford/test.json`.
- Schema (verbatim row 0):
```json
{"id": 1982, "image": "images/1982_testset_0000394.jpg",
 "conversations": [
  {"value": "<image>\nYou are a robot using the joint control. The task is \"throw the soccer_ball\". Please predict a possible affordance area of the end effector?", "from": "human"},
  {"from": "gpt", "value": "[161, 256, 183, 281]"}]}
```
  Box format `[x1, y1, x2, y2]`, **absolute pixels** (max value 2560 in the set, so some high-res images; images here include AGD20K-like names e.g. `..._testset_...`, `bicycle`, `motorcycle`; these differ from the full ShareRobot README which has bridge-style {x,y,width,height}). Image filename `0_8.png` types also exist.
- Metric (FlagEvalMM PR #84 `tasks/share_robot/affordance/evaluate.py`): IoU between predicted and GT box; reports average IoU, fraction with IoU >= 0.5 / 0.75 / 0.9, and "average_precision" which is literally the mean IoU (code comment: "Simplified mAP"). Prompt suffix: `Only return the box in format: [x1, y1, x2, y2] with no other output.` No normalisation is applied (pred compared in pixel space directly; a model emitting normalised or 0-1000 boxes would score ~0). Which of these numbers the tech report's "Afford." column is: UNVERIFIED (RoboBrain 1.0 paper used AP over IoU thresholds, sec 5.2 / p.7).
- RoboBrain 2.0 reported (Table 3, p.14 "Afford."): **7B 28.05; 32B 35.28; 3B not reported.** Baselines: RoboBrain-7B-1.0 10.20, Qwen2.5-VL-72B 23.80, Gemini-2.5-Pro 10.26, GPT-4o 6.00, Qwen2.5-VL-32B 11.97.
- RoboBrain 1.0 paper: affordance test 522 images; public bench has 200.

### 4b. Part-Afford (Embodied-R1; not reported by RoboBrain 2.0)
- https://huggingface.co/datasets/IffYuan/Part-Affordance-2K (not gated, licence UNVERIFIED), 2000 samples, one parquet **396 MB**, split named `train`; fields `question_id, problem, image, mask, category_type`. Derived from RGBD-Part-Affordance. Metric = points-in-mask. Too large to be worth it (optional).

## 5. Other things RoboBrain 2.0 reports (pointing, not requested)
- RefSpatial-Bench (Table 2, p.13): 7B Loc 36.00 / Pla 29.00 / All 32.50; 32B 54.00/54.00/54.00. Repo/size UNVERIFIED (not looked up).
- Table 2 also: BLINK 83.95, CV-Bench 85.75, EmbSpatial 76.32, RoboSpatial 54.23 (7B); SAT 75.33, VSI-Bench 36.10 (7B, Table 3).

## 6. Official code locations
| Item | Location | Status |
|---|---|---|
| ShareRobot traj+afford eval | FlagEvalMM PR #84 (https://github.com/flageval-baai/FlagEvalMM/pull/84), `tasks/share_robot/{trajectory,affordance}/{evaluate,process}.py` | open PR, NOT in main |
| Where2Place eval | https://github.com/flageval-baai/FlagEvalMM/tree/main/tasks/where2place, `flagevalmm/evaluator/point_utils.py` | in main |
| RefCOCO eval | https://github.com/flageval-baai/FlagEvalMM/tree/main/tasks/refcoco | in main |
| RoboRefIt / VABench-P / VABench-V / Part-Afford / Where2Place alt | https://github.com/pickxiguapi/Embodied-R1/tree/main/eval | in main |
| RoboBrain2.0 repo | https://github.com/FlagOpen/RoboBrain2.0 README: inference only, points to FlagEvalMM; names no benchmarks | no eval scripts |
| RoboPoint eval | https://github.com/wentaoyuan/RoboPoint (path UNVERIFIED) | not checked |

## 7. Gating / tokens
- No benchmark dataset above is gated. No HF token needed for data.
- Model repos BAAI/RoboBrain2.0-3B and 7B: `gated: auto` (needs HF login + accept terms; raw README returns "restricted" without auth).
- Licences: ShareRobot-Bench apache-2.0 (VERIFIED), Where2Place original apache-2.0 (VERIFIED); everything else UNVERIFIED (empty licence metadata).

## 8. Files written
- /home/jayamadu/Documents/Robobrain-2.0/data/raw/sharerobot_bench_traj/test.json (120 KB)
- /home/jayamadu/Documents/Robobrain-2.0/data/raw/sharerobot_bench_afford/test.json (80 KB)
- /home/jayamadu/Documents/Robobrain-2.0/data/raw/where2place_robopoint/point_questions.jsonl
- /home/jayamadu/Documents/Robobrain-2.0/data/raw/roborefit_embodiedr1_json/roborefit_test.json (1.4 MB)
