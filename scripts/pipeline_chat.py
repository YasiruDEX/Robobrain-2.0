#!/usr/bin/env python3
"""
Pipeline Chat with RoboBrain 2.0

This script uses Groq LLM to decompose complex user queries into a pipeline
of simple RoboBrain tasks (general, grounding, affordance, trajectory, pointing)
and executes them sequentially, passing results between steps.

Usage:
    python scripts/pipeline_chat.py
    python scripts/pipeline_chat.py --image path/to/image.jpg
"""

import argparse
import sys
import pathlib
import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from utils import get_model
from conversation_memory import MultiTurnInference

# Groq API for task decomposition
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Error: groq package required for pipeline chat. Install with: pip install groq")
    sys.exit(1)

# Try to import visualization libraries
try:
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Warning: cv2/matplotlib not available. Visualization disabled.")

# Results directory
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "pipeline"

# Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")



ROBOBRAIN_TASKS = {
    "general": {
        "description": "General visual question answering - describe scenes, count objects, identify colors, explain content",
        "output": "Text description or answer",
        "examples": ["What is in this image?", "How many objects are there?", "What color is the car?"]
    },
    "grounding": {
        "description": "Locate and find objects in the image, returns bounding box coordinates [x1,y1,x2,y2]",
        "output": "Bounding box coordinates",
        "examples": ["Where is the cup?", "Find the apple", "Locate the chair"]
    },
    "affordance": {
        "description": "Detect graspable areas and manipulation affordances, returns bounding box of graspable region",
        "output": "Bounding box of graspable area",
        "examples": ["Where to grasp the cup?", "How to pick up this object?", "Find graspable area"]
    },
    "trajectory": {
        "description": "Plan motion paths and trajectories, returns sequence of waypoints [(x1,y1), (x2,y2), ...]",
        "output": "List of waypoint coordinates",
        "examples": ["Plan path to the cup", "How to reach the object?", "Generate trajectory to target"]
    },
    "pointing": {
        "description": "Point to specific locations or identify what's at a point, returns single (x,y) coordinate",
        "output": "Single point coordinate",
        "examples": ["Point to the button", "Click on the handle", "What is at (100,200)?"]
    }
}


# ============================================================================
# GROQ PIPELINE DECOMPOSER
# ============================================================================

class GroqPipelineDecomposer:
    """Uses Groq LLM to decompose complex queries into simple task pipelines."""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def decompose(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Decompose a complex query into a pipeline of simple RoboBrain tasks.
        
        Returns a list of task steps, each with:
        - task: one of [general, grounding, affordance, trajectory, pointing]
        - prompt: the specific prompt for RoboBrain
        - description: human-readable description of this step
        - use_previous: whether to use results from previous step
        """
        
        task_descriptions = "\n".join([
            f"- {name}: {info['description']} (Output: {info['output']})"
            for name, info in ROBOBRAIN_TASKS.items()
        ])
        
        system_prompt = f"""You are an expert task decomposition system for a robot vision model called RoboBrain.

Your job is to break down complex user queries into a PIPELINE of simple, atomic tasks that RoboBrain can execute sequentially.

AVAILABLE TASKS:
{task_descriptions}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES FOR TASK DECOMPOSITION:
═══════════════════════════════════════════════════════════════════════════════

1. **ALWAYS GROUND OBJECTS FIRST**: Before any manipulation (affordance) or movement (trajectory), 
   you MUST first locate the object using "grounding" task.

2. **TRAJECTORY PROMPTS MUST BE SPECIFIC**: 
   - BAD: "move to the cup" (too vague)
   - GOOD: "move the robot end-effector from current position to reach the red cup on the table"
   - GOOD: "generate trajectory waypoints to navigate from the current gripper position to the target object location"
   
3. **TRAJECTORY TASK DETAILS**:
   - The trajectory task generates a sequence of (x, y) waypoints for robot motion
   - Always specify: START point (current position / gripper / robot arm)
   - Always specify: END point (the target object or location)
   - Always specify: The ACTION (reach, move to, navigate, approach)
   - Include context about what the robot is doing
   
4. **AFFORDANCE BEFORE MANIPULATION**:
   - Before picking up, always find the grasp point using "affordance"
   - Affordance returns WHERE on the object to grasp
   
5. **LOGICAL ORDERING**:
   - Understand scene → grounding (find object) → trajectory (move to it) → affordance (grasp it)
   - For pick-and-place: ground source → affordance → ground destination → trajectory to destination

═══════════════════════════════════════════════════════════════════════════════
TASK-SPECIFIC PROMPT GUIDELINES:
═══════════════════════════════════════════════════════════════════════════════

**GENERAL**: Use for scene understanding, descriptions, counting
  - "Describe all objects visible in this scene"
  - "What objects are on the table?"
  - "How many cups are there?"

**GROUNDING**: Use to LOCATE objects (returns bounding box [x1,y1,x2,y2])
  - "the red cup"
  - "the door handle"
  - "the leftmost object on the table"

**AFFORDANCE**: Use to find WHERE to GRASP (returns grasp bounding box)
  - "grasp the cup by its handle"
  - "find the graspable region on the bottle"
  - "where should the robot gripper contact to pick up the mug"

**TRAJECTORY**: Use to plan MOTION PATH (returns waypoints [[x1,y1], [x2,y2], ...])
  MUST include these elements in the prompt:
  - What is moving: "robot arm", "end-effector", "gripper"
  - Action verb: "reach", "move to", "approach", "navigate toward"
  - Target description: specific object or location
  - Context: what task is being performed
  
  Example prompts:
  - "move the robot gripper to reach the cup located on the table"
  - "generate trajectory for the end-effector to approach the door handle"
  - "plan motion path from current position to the target object for grasping"
  - "move toward the red cup to prepare for picking it up"

**POINTING**: Use to identify a SINGLE POINT (returns (x,y) coordinate)
  - "point to the center of the button"
  - "click location on the screen"

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON array):
═══════════════════════════════════════════════════════════════════════════════
[
  {{
    "step": 1,
    "task": "task_type",
    "prompt": "detailed, specific prompt for RoboBrain",
    "description": "human-readable description",
    "use_previous": false
  }},
  {{
    "step": 2,
    "task": "task_type", 
    "prompt": "detailed prompt with context from previous step",
    "description": "what this step accomplishes",
    "use_previous": true
  }}
]

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════════════════════════

Query: "Pick up the red cup"
Pipeline:
[
  {{"step": 1, "task": "grounding", "prompt": "the red cup", "description": "Locate the red cup in the scene", "use_previous": false}},
  {{"step": 2, "task": "trajectory", "prompt": "move the robot end-effector from its current position to approach and reach the red cup for grasping", "description": "Generate motion trajectory to reach the cup", "use_previous": true}},
  {{"step": 3, "task": "affordance", "prompt": "find the optimal grasp region on the red cup where the robot gripper should make contact to securely pick it up", "description": "Determine grasp point on the cup", "use_previous": true}}
]

Query: "Move the bottle to the plate"
Pipeline:
[
  {{"step": 1, "task": "grounding", "prompt": "the bottle", "description": "Locate the bottle", "use_previous": false}},
  {{"step": 2, "task": "affordance", "prompt": "find where to grasp the bottle securely for manipulation", "description": "Find grasp point on bottle", "use_previous": true}},
  {{"step": 3, "task": "grounding", "prompt": "the plate", "description": "Locate the destination plate", "use_previous": false}},
  {{"step": 4, "task": "trajectory", "prompt": "plan the robot arm trajectory to move from the bottle location toward the plate, carrying the grasped object to place it near the plate", "description": "Generate path to move bottle to plate", "use_previous": true}}
]

Query: "Navigate to the door and open it"
Pipeline:
[
  {{"step": 1, "task": "grounding", "prompt": "the door handle", "description": "Locate the door handle", "use_previous": false}},
  {{"step": 2, "task": "trajectory", "prompt": "generate trajectory waypoints for the robot to navigate and move its end-effector from current position toward the door handle location", "description": "Plan approach path to door handle", "use_previous": true}},
  {{"step": 3, "task": "affordance", "prompt": "identify the graspable region on the door handle where the robot should grip to turn and open the door", "description": "Find grasp point on door handle", "use_previous": true}}
]

Query: "What objects can I interact with?"
Pipeline:
[
  {{"step": 1, "task": "general", "prompt": "List and describe all objects visible in this scene that could potentially be manipulated or interacted with by a robot", "description": "Identify all objects in scene", "use_previous": false}},
  {{"step": 2, "task": "affordance", "prompt": "identify all graspable regions and manipulation affordances for the objects in the scene", "description": "Find interactable/graspable areas", "use_previous": true}}
]

IMPORTANT: Return ONLY the JSON array, no other text. Make trajectory prompts DETAILED and SPECIFIC."""

        context_info = f"\nPrevious context: {context}" if context else ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this query into a detailed task pipeline:{context_info}\n\nQuery: \"{query}\""}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                pipeline = json.loads(json_match.group())
                # Post-process: enhance trajectory prompts if they're too short
                for step in pipeline:
                    if step.get("task") == "trajectory":
                        prompt = step.get("prompt", "")
                        if len(prompt) < 50:  # Too short, enhance it
                            step["prompt"] = f"generate trajectory waypoints for the robot end-effector to {prompt}"
                return pipeline
            else:
                # Fallback: single general task
                return [{
                    "step": 1,
                    "task": "general",
                    "prompt": query,
                    "description": "Process query directly",
                    "use_previous": False
                }]
                
        except Exception as e:
            print(f"⚠️  Groq decomposition error: {e}")
            # Fallback to single task
            return [{
                "step": 1,
                "task": "general",
                "prompt": query,
                "description": "Fallback: process as general query",
                "use_previous": False
            }]
    
    def classify_single(self, query: str) -> str:
        """Classify a simple query into a single task type."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Classify this query into exactly ONE task type for a robot vision system.

Available tasks:
- general: questions about scene content, descriptions, counting, identifying objects
- grounding: finding/locating WHERE an object is (returns bounding box coordinates)
- affordance: grasping/manipulation queries - WHERE to grasp an object (returns grasp region)
- trajectory: path planning, motion planning, navigation, reaching - HOW to move to a target (returns waypoints)
- pointing: point to a specific location (returns single point coordinate)

CLASSIFICATION RULES:
- "where is X" or "find X" or "locate X" → grounding
- "grasp X" or "pick up X" or "hold X" → affordance  
- "move to X" or "reach X" or "go to X" or "path to X" or "navigate to X" → trajectory
- "point to X" or "click X" → pointing
- descriptions, counting, "what is" → general

Respond with ONLY the task name, nothing else."""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            task = response.choices[0].message.content.strip().lower()
            if task in ROBOBRAIN_TASKS:
                return task
            return "general"
            
        except Exception as e:
            print(f"⚠️  Classification error: {e}")
            return "general"
    
    def enhance_trajectory_prompt(self, original_prompt: str) -> str:
        """Enhance a trajectory prompt to be more detailed for better results."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a robot motion planning expert. Your job is to enhance trajectory prompts to be more detailed and specific.

A good trajectory prompt should include:
1. What is moving (robot arm, end-effector, gripper)
2. The action (move, reach, approach, navigate)
3. The target (specific object or location)
4. Context about the task

Transform the input prompt into a detailed trajectory instruction.

Examples:
- Input: "move to the cup"
  Output: "move the robot end-effector from current position to approach and reach the cup for manipulation"
  
- Input: "go to the door"
  Output: "generate trajectory waypoints for the robot to navigate toward the door location"
  
- Input: "reach the bottle"
  Output: "plan motion path for the robot gripper to reach the bottle position for grasping"

Respond with ONLY the enhanced prompt, nothing else."""
                    },
                    {"role": "user", "content": f"Enhance this trajectory prompt: \"{original_prompt}\""}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            enhanced = response.choices[0].message.content.strip()
            # Remove quotes if present
            enhanced = enhanced.strip('"\'')
            return enhanced if len(enhanced) > len(original_prompt) else original_prompt
            
        except Exception as e:
            print(f"⚠️  Prompt enhancement error: {e}")
            return original_prompt


# ============================================================================

class PipelineExecutor:
    """Executes a pipeline of RoboBrain tasks."""
    
    def __init__(self, model, repo_dir, decomposer=None):
        self.chat = MultiTurnInference(model, repo_dir)
        self.results_history = []
        self.decomposer = decomposer  # For trajectory prompt enhancement
    
    def set_image(self, image_path: str):
        """Set the image for all pipeline steps."""
        self.chat.set_image(image_path)
    
    def execute_pipeline(self, pipeline: List[Dict], enable_thinking: bool = False) -> Dict[str, Any]:
        """
        Execute a pipeline of tasks sequentially.
        
        Returns:
        - steps: list of step results
        - final_answer: combined final answer
        - visualizations: list of visualization paths
        """
        results = {
            "steps": [],
            "final_answer": "",
            "visualizations": [],
            "success": True
        }
        
        previous_result = None
        
        for step_info in pipeline:
            step_num = step_info.get("step", len(results["steps"]) + 1)
            task = step_info.get("task", "general")
            prompt = step_info.get("prompt", "")
            description = step_info.get("description", "")
            use_previous = step_info.get("use_previous", False)
            
            # Inject previous result into prompt if needed
            if use_previous and previous_result:
                prompt = prompt.replace("{previous_result}", str(previous_result))
                # Also append context
                if "{previous_result}" not in step_info.get("prompt", ""):
                    prompt = f"{prompt} (Context: {previous_result[:200]})"
            
            # Enhance trajectory prompts for better results
            if task == "trajectory" and self.decomposer and len(prompt) < 80:
                original_prompt = prompt
                prompt = self.decomposer.enhance_trajectory_prompt(prompt)
                if prompt != original_prompt:
                    print(f"   📝 Enhanced prompt: {prompt[:80]}...")
            
            print(f"\n{'='*60}")
            print(f"📍 Step {step_num}: {description}")
            print(f"   Task: {task.upper()}")
            print(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"{'='*60}")
            
            # Execute the task
            try:
                result = self.chat.ask(
                    prompt,
                    task=task,
                    enable_thinking=enable_thinking
                )
                
                answer = result.get("answer", "")
                thinking = result.get("thinking", "")
                
                step_result = {
                    "step": step_num,
                    "task": task,
                    "prompt": prompt,
                    "description": description,
                    "answer": answer,
                    "thinking": thinking,
                    "success": True
                }
                
                print(f"\n✅ Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
                
                # Parse coordinates if spatial task
                coords = self._parse_coordinates(answer, task)
                if coords:
                    step_result["coordinates"] = coords
                    print(f"   📊 Coordinates: {coords[:3]}{'...' if len(coords) > 3 else ''}")
                
                # Visualize if spatial task
                if task in ["grounding", "affordance", "trajectory", "pointing"] and coords:
                    vis_path = self._visualize_step(
                        self.chat.memory.current_image,
                        answer, task, prompt, step_num
                    )
                    if vis_path:
                        step_result["visualization"] = str(vis_path)
                        results["visualizations"].append(str(vis_path))
                
                results["steps"].append(step_result)
                previous_result = answer
                
            except Exception as e:
                print(f"\n❌ Error in step {step_num}: {e}")
                step_result = {
                    "step": step_num,
                    "task": task,
                    "prompt": prompt,
                    "description": description,
                    "error": str(e),
                    "success": False
                }
                results["steps"].append(step_result)
                results["success"] = False
                # Continue with next step anyway
                previous_result = f"Error: {e}"
        
        # Combine results into final answer
        results["final_answer"] = self._combine_results(results["steps"])
        
        return results
    
    def _parse_coordinates(self, answer: str, task: str) -> Optional[List]:
        """Parse coordinates from model answer."""
        try:
            if task == "pointing":
                pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
                matches = re.findall(pattern, answer)
                if matches:
                    return [(float(x), float(y)) for x, y in matches]
            
            elif task in ["affordance", "grounding"]:
                # Bounding box format
                pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
                matches = re.findall(pattern, answer)
                if matches:
                    return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in matches]
                # Fallback: points to bbox
                pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
                matches = re.findall(pattern, answer)
                if matches:
                    result = []
                    for x, y in matches:
                        cx, cy = float(x), float(y)
                        result.append((cx - 30, cy - 30, cx + 30, cy + 30))
                    return result
            
            elif task == "trajectory":
                # List of points
                pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
                matches = re.findall(pattern, answer)
                if matches:
                    return [(float(x), float(y)) for x, y in matches]
                pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
                matches = re.findall(pattern, answer)
                if matches:
                    return [(float(x), float(y)) for x, y in matches]
        except Exception:
            pass
        return None
    
    def _visualize_step(self, image_path: str, answer: str, task: str, prompt: str, step_num: int) -> Optional[pathlib.Path]:
        """Visualize a single step result."""
        if not HAS_VIS:
            return None
        
        try:
            if image_path.startswith("http"):
                import requests
                from io import BytesIO
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(image_path)
            
            if img is None:
                return None
            
            h, w = img.shape[:2]
            coords = self._parse_coordinates(answer, task)
            
            if coords is None:
                return None
            
            vis_img = img.copy()
            
            # Color based on step
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            color = colors[(step_num - 1) % len(colors)]
            
            if task == "pointing":
                for i, (x, y) in enumerate(coords):
                    if x <= 1 and y <= 1:
                        px, py = int(x * w), int(y * h)
                    else:
                        px, py = int(x), int(y)
                    cv2.circle(vis_img, (px, py), 15, color, 3)
                    cv2.circle(vis_img, (px, py), 5, (0, 0, 255), -1)
            
            elif task in ["affordance", "grounding"]:
                for i, (x1, y1, x2, y2) in enumerate(coords):
                    if all(v <= 1 for v in [x1, y1, x2, y2]):
                        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(vis_img, f"Step {step_num}: {task}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            elif task == "trajectory":
                points = []
                for x, y in coords:
                    if x <= 1 and y <= 1:
                        px, py = int(x * w), int(y * h)
                    else:
                        px, py = int(x), int(y)
                    points.append((px, py))
                
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        cv2.line(vis_img, points[i], points[i+1], color, 3)
                    for i, pt in enumerate(points):
                        cv2.circle(vis_img, pt, 8, color, -1)
                        cv2.putText(vis_img, str(i+1), (pt[0]-5, pt[1]+5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Save
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"step{step_num}_{task}_{timestamp}.png"
            
            cv2.imwrite(str(output_path), vis_img)
            return output_path
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            return None
    
    def _combine_results(self, steps: List[Dict]) -> str:
        """Combine step results into a coherent final answer."""
        if not steps:
            return "No results available."
        
        successful_steps = [s for s in steps if s.get("success", False)]
        
        if len(successful_steps) == 1:
            return successful_steps[0].get("answer", "")
        
        summary_parts = []
        for step in successful_steps:
            summary_parts.append(f"Step {step['step']} ({step['task']}): {step.get('answer', '')[:200]}")
        
        return "\n\n".join(summary_parts)
    
    def reset(self):
        """Reset the conversation and results."""
        self.chat.reset()
        self.results_history = []


# ============================================================================
# COMBINED VISUALIZATION
# ============================================================================

def create_pipeline_visualization(image_path: str, pipeline_results: Dict, output_path: str) -> Optional[str]:
    """Create a combined visualization showing all pipeline steps."""
    if not HAS_VIS:
        return None
    
    try:
        if image_path.startswith("http"):
            import requests
            from io import BytesIO
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        h, w = img.shape[:2]
        vis_img = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for step in pipeline_results.get("steps", []):
            if not step.get("success") or "coordinates" not in step:
                continue
            
            step_num = step["step"]
            task = step["task"]
            coords = step["coordinates"]
            color = colors[(step_num - 1) % len(colors)]
            
            if task == "pointing":
                for x, y in coords:
                    if x <= 1 and y <= 1:
                        px, py = int(x * w), int(y * h)
                    else:
                        px, py = int(x), int(y)
                    cv2.circle(vis_img, (px, py), 12, color, 3)
            
            elif task in ["affordance", "grounding"]:
                for x1, y1, x2, y2 in coords:
                    if all(v <= 1 for v in [x1, y1, x2, y2]):
                        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            elif task == "trajectory":
                points = []
                for x, y in coords:
                    if x <= 1 and y <= 1:
                        px, py = int(x * w), int(y * h)
                    else:
                        px, py = int(x), int(y)
                    points.append((px, py))
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        cv2.line(vis_img, points[i], points[i+1], color, 2)
                    for pt in points:
                        cv2.circle(vis_img, pt, 5, color, -1)
            
            # Add legend
            cv2.putText(vis_img, f"S{step_num}: {task}", (10, 25 * step_num),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, vis_img)
        return output_path
        
    except Exception as e:
        print(f"⚠️  Combined visualization error: {e}")
        return None


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🔗 RoboBrain 2.0 PIPELINE Chat (Task Decomposition Engine) 🔗  ║
║                                                                  ║
║  This chat uses GROQ LLM to decompose complex queries into      ║
║  pipelines of simple RoboBrain tasks:                           ║
║                                                                  ║
║  Available Tasks:                                                ║
║    • general    - Scene understanding, Q&A                      ║
║    • grounding  - Object localization (bounding box)            ║
║    • affordance - Grasp detection (bounding box)                ║
║    • trajectory - Path planning (waypoints)                     ║
║    • pointing   - Point identification (single point)           ║
║                                                                  ║
║  Complex Query Examples:                                         ║
║    "Pick up the red cup and put it near the plate"              ║
║    "Navigate to the door and find where to grasp the handle"    ║
║    "What objects can I interact with and how?"                  ║
║                                                                  ║
║  Commands:                                                       ║
║    /image <path>  - Set a new image                             ║
║    /simple        - Toggle simple mode (single task only)       ║
║    /thinking on   - Enable thinking mode                        ║
║    /thinking off  - Disable thinking mode                       ║
║    /results       - Show last pipeline results                  ║
║    /clear         - Clear conversation                          ║
║    /help          - Show this help                              ║
║    /quit          - Exit                                        ║
║                                                                  ║
║  💡 Type complex queries - AI decomposes them automatically!    ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Pipeline chat with RoboBrain 2.0")
    parser.add_argument("--image", "-i", type=str, help="Initial image to analyze")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--simple", action="store_true", help="Simple mode (single task, no decomposition)")
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize Groq decomposer
    try:
        decomposer = GroqPipelineDecomposer(GROQ_API_KEY)
        print("🤖 Groq Pipeline Decomposer initialized (LLaMA 3.3 70B)")
    except Exception as e:
        print(f"❌ Could not initialize Groq: {e}")
        sys.exit(1)
    
    # Load model
    print("\nLoading RoboBrain model...")
    model, repo_dir = get_model()
    
    # Initialize executor with decomposer for trajectory enhancement
    executor = PipelineExecutor(model, repo_dir, decomposer=decomposer)
    
    # Settings
    simple_mode = args.simple
    enable_thinking = not args.no_thinking
    last_results = None
    
    # Set initial image
    if args.image:
        executor.set_image(args.image)
    else:
        demo_image = repo_dir / "assets/demo/grounding.jpg"
        if demo_image.exists():
            executor.set_image(str(demo_image))
            print(f"📷 Using demo image: {demo_image}")
        else:
            print("⚠️  No image set. Use /image <path> to set one.")
    
    print(f"\n🔗 Pipeline mode: {'SIMPLE (single task)' if simple_mode else 'FULL (decomposition)'}")
    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    print("\nType your query (complex queries will be decomposed)!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ["/quit", "/exit", "/q"]:
                print("Goodbye! 👋")
                break
            
            elif cmd == "/help":
                print_banner()
            
            elif cmd == "/image":
                if arg:
                    if pathlib.Path(arg).exists() or arg.startswith("http"):
                        executor.set_image(arg)
                    else:
                        print(f"❌ Image not found: {arg}")
                else:
                    print(f"📷 Current image: {executor.chat.memory.current_image}")
            
            elif cmd == "/simple":
                simple_mode = not simple_mode
                print(f"🔗 Pipeline mode: {'SIMPLE (single task)' if simple_mode else 'FULL (decomposition)'}")
            
            elif cmd == "/thinking":
                if arg.lower() == "on":
                    enable_thinking = True
                    print("🧠 Thinking mode: ON")
                elif arg.lower() == "off":
                    enable_thinking = False
                    print("🧠 Thinking mode: OFF")
                else:
                    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            
            elif cmd == "/results":
                if last_results:
                    print("\n📊 Last Pipeline Results:")
                    print(json.dumps(last_results, indent=2, default=str)[:2000])
                else:
                    print("No results yet.")
            
            elif cmd == "/clear":
                executor.reset()
                last_results = None
                print("🗑️  Conversation and results cleared")
            
            else:
                print(f"❌ Unknown command: {cmd}. Type /help for available commands.")
            
            continue
        
        # Check image
        if not executor.chat.memory.current_image:
            print("⚠️  No image set. Use /image <path> to set one first.")
            continue
        
        # Process query
        if simple_mode:
            # Simple mode: single task classification
            task = decomposer.classify_single(user_input)
            print(f"\n🎯 Task: {task.upper()}")
            
            pipeline = [{
                "step": 1,
                "task": task,
                "prompt": user_input,
                "description": f"Execute as {task} task",
                "use_previous": False
            }]
        else:
            # Full mode: decompose into pipeline
            print("\n🔍 Decomposing query into pipeline...")
            pipeline = decomposer.decompose(user_input)
            
            print(f"\n📋 Pipeline ({len(pipeline)} steps):")
            for step in pipeline:
                print(f"   {step['step']}. [{step['task'].upper()}] {step['description']}")
        
        # Execute pipeline
        print(f"\n🚀 Executing pipeline...")
        results = executor.execute_pipeline(pipeline, enable_thinking=enable_thinking)
        last_results = results
        
        # Show final summary
        print(f"\n{'='*60}")
        print("📊 PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful steps: {sum(1 for s in results['steps'] if s.get('success'))}/{len(results['steps'])}")
        
        if results.get("visualizations"):
            print(f"📸 Visualizations saved: {len(results['visualizations'])}")
            for vis in results["visualizations"]:
                print(f"   - {vis}")
        
        # Create combined visualization
        if len(results.get("steps", [])) > 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_path = RESULTS_DIR / f"pipeline_combined_{timestamp}.png"
            combined_vis = create_pipeline_visualization(
                executor.chat.memory.current_image,
                results,
                str(combined_path)
            )
            if combined_vis:
                print(f"🎨 Combined visualization: {combined_vis}")
        
        print(f"\n📝 Final Answer:\n{results.get('final_answer', 'No answer')[:500]}")
        print()


if __name__ == "__main__":
    main()
