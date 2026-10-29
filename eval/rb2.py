"""Deterministic RoboBrain 2.0 wrapper for evaluation.

Prompt templates are copied verbatim from RoboBrain2.0_lib/rb_v2.0/inference_rb2.py
(the 2.0 conventions); `context` lets a pipeline step inject spatial priors.
"""
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

POINT_SUFFIX = ("Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
                "where each tuple contains the x and y coordinates of a point satisfying the conditions "
                "above. The coordinates should indicate the normalized pixel locations of the points in the image.")
TRAJ_SUFFIX = ("Please predict up to 10 key trajectory points to complete the task. Your answer should be "
               "formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains "
               "the x and y coordinates of a point.")


def build_prompt(task, text, context="", robot=None):
    ctx = f" {context}" if context else ""
    if task == "trajectory" and robot:  # benchmark-native wording (ShareRobot-Bench)
        return f"You are {robot}. The task is {text}.{ctx} {TRAJ_SUFFIX}"
    if task == "pointing":
        return f"{text}.{ctx} {POINT_SUFFIX}"
    if task == "affordance":
        return (f"You are a robot using the joint control. The task is \"{text}\".{ctx} "
                "Please predict a possible affordance area of the end effector.")
    if task == "trajectory":
        return f"You are a robot using the joint control. The task is \"{text}\".{ctx} {TRAJ_SUFFIX}"
    if task == "grounding":
        return f"Please provide the bounding box coordinate of the region this sentence describes: {text}.{ctx}"
    return f"{text}{ctx}"


class RB2:
    def __init__(self, model_path, quant="4bit", max_pixels=512 * 28 * 28, max_new_tokens=256):
        qc = {"8bit": BitsAndBytesConfig(load_in_8bit=True),
              "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16),
              "fp16": None}[quant]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, quantization_config=qc, device_map="cuda:0")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.quant = quant

    @torch.inference_mode()
    def infer(self, image_path, text, task="general", context="", robot=None):
        prompt = build_prompt(task, text, context, robot)
        content = [{"type": "text", "text": prompt}]
        if image_path:  # image_path=None -> text-only call (planner)
            content.insert(0, {"type": "image", "image": f"file://{image_path}", "max_pixels": self.max_pixels})
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages) if image_path else (None, None)
        inputs = self.processor(text=[chat], images=images, videos=videos, padding=True,
                                return_tensors="pt").to("cuda:0")
        scale = None
        if image_path:
            _, gh, gw = inputs["image_grid_thw"][0].tolist()
            scale = (gw * 14, gh * 14)  # resized (W, H) the model sees and answers in
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return {"raw": raw, "prompt": prompt, "scale": scale, "latency": latency,
                "n_tokens": int(out.shape[1] - inputs.input_ids.shape[1])}
