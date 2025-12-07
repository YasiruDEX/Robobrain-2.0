#!/usr/bin/env python3
"""
Smart Interactive Chat with RoboBrain 2.0

This script automatically classifies user prompts and selects the appropriate
task type (general, grounding, affordance, trajectory, pointing) using
Groq LLM with function calling for intelligent ML-based classification.

Usage:
    python scripts/smart_chat.py
    python scripts/smart_chat.py --image path/to/image.jpg
"""

import argparse
import sys
import pathlib
import re
import json
import os
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from utils import get_model
from conversation_memory import MultiTurnInference

# Groq API for ML-based task classification
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: groq package not installed. Using fallback keyword matching.")

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
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "interactive"

# Groq API Key - set via environment variable GROQ_API_KEY or hardcoded fallback
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "groq_your_api_key_here")


# ============================================================================
# GROQ ML-BASED TASK CLASSIFICATION WITH TOOLS
# ============================================================================

# Define tools for each task type
TASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "task_general",
            "description": "Use this for general questions about the image content, descriptions, counting objects, identifying colors, explaining scenes, or any question that asks WHAT something is, WHO is in the image, HOW MANY objects there are, or requests a description. Examples: 'What is in this image?', 'Describe the scene', 'How many people are there?', 'What color is the car?', 'Is there a dog?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this is a general question"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_grounding",
            "description": "Use this ONLY when the user wants to LOCATE or FIND where an object is positioned in the image using a BOUNDING BOX. This returns bounding box coordinates [x1,y1,x2,y2] for visual localization. Use for pure location questions like 'Where is the X?', 'Find the X', 'Locate the X', 'Show me where X is', 'What is the position of X?'. DO NOT use this for grasping, picking up, pointing, or manipulation queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The object the user wants to locate"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this is a grounding/localization task"
                    }
                },
                "required": ["target_object", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_affordance",
            "description": "Use this when the user wants to GRASP, GRAB, PICK UP, HOLD, MANIPULATE, or INTERACT with objects physically. This detects WHERE and HOW a robot can grip/grasp objects and returns a BOUNDING BOX around the graspable area. ALWAYS use this for any query involving: picking up, grasping, grabbing, gripping, holding, lifting, manipulation, robot interaction, end-effector positioning. Examples: 'Grasp the cup', 'Pick up the mug', 'How to grab this?', 'What can I pick up?', 'Grasp the handle', 'Hold the bottle', 'Lift the box'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The object or area to check for grasping/manipulation affordance"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this is an affordance/manipulation task"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_trajectory",
            "description": "Use this when the user wants to PLAN A PATH, generate WAYPOINTS, or create a movement TRAJECTORY for robot motion. This returns a sequence of points for navigation or arm movement. Use for questions about motion planning, reaching objects through space, path planning, robot arm trajectory, or navigation. Examples: 'Plan a path to X', 'How should the robot move to reach X?', 'Generate trajectory', 'Move to the cup', 'Navigate to X', 'What path should I take?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The target location or object for the trajectory"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this is a trajectory/path planning task"
                    }
                },
                "required": ["target", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_pointing",
            "description": "Use this when the user wants to POINT TO, CLICK ON, or IDENTIFY A SINGLE POINT location of something. This returns a SINGLE POINT coordinate (x,y), not a bounding box. Use for: 'Point to X', 'Click on X', 'Tap on X', 'Show me the exact point of X', 'Mark X', 'What is at coordinates (x,y)?'. Also use when user provides specific pixel coordinates and wants to identify what's there.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The object or location to point to"
                    },
                    "coordinates": {
                        "type": "string",
                        "description": "The coordinates mentioned by the user, if any"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this is a pointing task"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]

# Map tool names to task names
TOOL_TO_TASK = {
    "task_general": "general",
    "task_grounding": "grounding",
    "task_affordance": "affordance",
    "task_trajectory": "trajectory",
    "task_pointing": "pointing"
}


class GroqTaskClassifier:
    """ML-based task classifier using Groq API with function calling."""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Fast and good at function calling
    
    def classify(self, prompt: str) -> tuple:
        """
        Classify the user prompt into a task using Groq LLM with tools.
        Returns (task_name, confidence, reason, extra_info)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a task classifier for a robot vision system. 
Your job is to determine what type of task the user wants based on their question about an image.
Choose the most appropriate tool/function based on the user's intent.

CRITICAL DISTINCTION - Output Types:
- AFFORDANCE: Returns BOUNDING BOX [x1,y1,x2,y2] for graspable area
- GROUNDING: Returns BOUNDING BOX [x1,y1,x2,y2] for object location  
- POINTING: Returns SINGLE POINT (x,y) coordinate
- TRAJECTORY: Returns SEQUENCE of points [(x1,y1), (x2,y2), ...]

TASK SELECTION RULES:

1. AFFORDANCE (bounding box): Any query about GRASPING, PICKING UP, GRABBING, HOLDING, LIFTING, or MANIPULATING objects.
   Keywords: grasp, grab, pick up, hold, lift, grip, manipulate, affordance, graspable
   Examples: "Grasp the cup", "Pick up the mug", "What can I grab?", "Hold the handle"
   
2. GROUNDING (bounding box): ONLY for pure LOCATION queries - user wants to FIND/LOCATE an object visually.
   Keywords: where is, find, locate, position of, show me where
   Examples: "Where is the cup?", "Find the apple", "Locate the chair"

3. POINTING (single point): When user wants to POINT TO something or identify what's at a specific point.
   Keywords: point to, point at, click on, tap on, mark, what is at (x,y)
   Examples: "Point to the cup", "Click on the button", "What is at (100,200)?"

4. TRAJECTORY (path/sequence): Path planning, motion planning, navigation.
   Keywords: plan path, move to, navigate, reach, trajectory, route
   Examples: "Plan a path to the cup", "How to reach the bottle?"

5. GENERAL: Questions about content - descriptions, counting, colors, scene understanding.
   Keywords: what is, describe, how many, count, color, explain
   Examples: "What is this?", "Describe the scene", "How many objects?"

REMEMBER:
- "Grasp X" → AFFORDANCE (bounding box of graspable area)
- "Point to X" → POINTING (single point coordinate)
- "Where is X?" → GROUNDING (bounding box of object)

Always call exactly one tool based on the user's primary intent."""
                    },
                    {
                        "role": "user",
                        "content": f"Classify this query for a robot vision system: \"{prompt}\""
                    }
                ],
                tools=TASK_TOOLS,
                tool_choice="required",
                temperature=0.1,
                max_tokens=200
            )
            
            # Extract the tool call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                task = TOOL_TO_TASK.get(tool_name, "general")
                reason = tool_args.get("reason", "ML classification")
                extra_info = {k: v for k, v in tool_args.items() if k != "reason"}
                
                return task, 0.95, reason, extra_info
            else:
                # Fallback if no tool was called
                return "general", 0.5, "No tool selected, defaulting to general", {}
                
        except Exception as e:
            print(f"⚠️  Groq API error: {e}")
            return None, 0, str(e), {}


# Fallback keyword-based classifier
def classify_task_fallback(prompt: str) -> tuple:
    """Fallback keyword-based classification when Groq is unavailable."""
    prompt_lower = prompt.lower().strip()
    
    # Simple keyword matching
    if any(kw in prompt_lower for kw in ["where is", "find", "locate", "show me where"]):
        return "grounding", 0.7, "keyword: location query", {}
    elif any(kw in prompt_lower for kw in ["grasp", "grab", "pick up", "graspable", "affordance"]):
        return "affordance", 0.7, "keyword: manipulation query", {}
    elif any(kw in prompt_lower for kw in ["path", "trajectory", "move to", "reach", "navigate"]):
        return "trajectory", 0.7, "keyword: motion planning query", {}
    elif re.search(r"at\s*\(?\s*\d+", prompt_lower) or "at coordinate" in prompt_lower:
        return "pointing", 0.7, "keyword: coordinate query", {}
    else:
        return "general", 0.6, "default: general question", {}


def parse_coordinates(answer, task):
    """Parse coordinates from model answer based on task type."""
    try:
        if task == "pointing":
            # Look for tuples like (x, y) or [(x, y)]
            pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x), float(y)) for x, y in matches]
        
        elif task == "affordance" or task == "grounding":
            # First try bounding box format [x1, y1, x2, y2]
            pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in matches]
            
            # Fallback: if model returned point coordinates [(x, y)], convert to small bbox
            pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
            matches = re.findall(pattern, answer)
            if matches:
                # Convert points to small bounding boxes (point ± 30 pixels)
                result = []
                for x, y in matches:
                    cx, cy = float(x), float(y)
                    result.append((cx - 30, cy - 30, cx + 30, cy + 30))
                return result
        
        elif task == "trajectory":
            # Look for list of points [[x,y], [x,y], ...] or [(x,y), ...]
            pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x), float(y)) for x, y in matches]
            # Try tuple format
            pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x), float(y)) for x, y in matches]
    except Exception as e:
        print(f"Warning: Could not parse coordinates: {e}")
    
    return None


def visualize_result(image_path, answer, task, query):
    """Visualize the result on the image and save it."""
    if not HAS_VIS:
        print("⚠️  Visualization libraries not available")
        return None
    
    # Load image
    if image_path.startswith("http"):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        print(f"⚠️  Could not load image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    coords = parse_coordinates(answer, task)
    
    if coords is None:
        print("⚠️  Could not parse coordinates from answer")
        return None
    
    vis_img = img.copy()
    
    if task == "pointing":
        for i, (x, y) in enumerate(coords):
            if x <= 1 and y <= 1:
                px, py = int(x * w), int(y * h)
            else:
                px, py = int(x), int(y)
            
            cv2.circle(vis_img, (px, py), 15, (0, 255, 0), 3)
            cv2.circle(vis_img, (px, py), 5, (0, 0, 255), -1)
            cv2.line(vis_img, (px - 20, py), (px + 20, py), (0, 255, 0), 2)
            cv2.line(vis_img, (px, py - 20), (px, py + 20), (0, 255, 0), 2)
            cv2.putText(vis_img, f"P{i+1}({px},{py})", (px + 10, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    elif task == "affordance" or task == "grounding":
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, (x1, y1, x2, y2) in enumerate(coords):
            color = colors[i % len(colors)]
            if all(v <= 1 for v in [x1, y1, x2, y2]):
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            label = f"{task.upper()} {i+1}" if len(coords) > 1 else task.upper()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis_img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    elif task == "trajectory":
        if len(coords) >= 2:
            points = []
            for x, y in coords:
                if x <= 1 and y <= 1:
                    px, py = int(x * w), int(y * h)
                else:
                    px, py = int(x), int(y)
                points.append((px, py))
            
            for i in range(len(points) - 1):
                ratio = i / (len(points) - 1)
                color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
                cv2.line(vis_img, points[i], points[i + 1], color, 3)
            
            for i, (px, py) in enumerate(points):
                if i == 0:
                    cv2.circle(vis_img, (px, py), 12, (0, 255, 0), -1)
                    cv2.putText(vis_img, "START", (px + 15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif i == len(points) - 1:
                    cv2.circle(vis_img, (px, py), 12, (0, 0, 255), -1)
                    cv2.putText(vis_img, "END", (px + 15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.circle(vis_img, (px, py), 8, (255, 165, 0), -1)
                cv2.putText(vis_img, str(i + 1), (px - 5, py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    title = f"Task: {task.upper()} | Query: {query[:50]}{'...' if len(query) > 50 else ''}"
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{task}_{timestamp}.png"
    
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(vis_img_rgb)
    ax.axis('off')
    ax.set_title(f"{task.upper()}: {query}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Result saved to: {output_path}")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(vis_img_rgb)
        ax.axis('off')
        ax.set_title(f"{task.upper()}: {query}", fontsize=12)
        plt.show(block=False)
        plt.pause(0.1)
    except Exception as e:
        print(f"Note: Could not display image ({e})")
    
    return output_path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🧠 RoboBrain 2.0 SMART Chat (ML-Powered Task Selection) 🧠 ║
║                                                              ║
║  This chat uses GROQ LLM to intelligently detect your task:  ║
║    • "Where is the cup?"      → grounding (bounding box)    ║
║    • "What can I grasp here?" → affordance (grasp areas)    ║
║    • "Plan a path to the cup" → trajectory (waypoints)      ║
║    • "What is at (100,200)?"  → pointing (identify point)   ║
║    • "Describe this scene"    → general (free Q&A)          ║
║                                                              ║
║  🤖 Powered by: Groq LLaMA 3.3 70B for task classification  ║
║  📊 Results auto-saved to: results/interactive/             ║
║                                                              ║
║  Commands:                                                   ║
║    /image <path>  - Set a new image                         ║
║    /task <type>   - Force a specific task                   ║
║    /auto on/off   - Toggle ML task detection                ║
║    /thinking on   - Enable thinking mode                    ║
║    /thinking off  - Disable thinking mode                   ║
║    /clear         - Clear conversation memory               ║
║    /help          - Show this help                          ║
║    /quit          - Exit                                    ║
║                                                              ║
║  💡 Just type naturally - AI will figure out what you need! ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Smart chat with RoboBrain 2.0 (ML-powered task selection)")
    parser.add_argument("--image", "-i", type=str, help="Initial image to analyze")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML task detection (use keyword fallback)")
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize Groq classifier if available
    groq_classifier = None
    if HAS_GROQ and not args.no_ml:
        try:
            groq_classifier = GroqTaskClassifier(GROQ_API_KEY)
            print("🤖 Groq ML classifier initialized (LLaMA 3.3 70B)")
        except Exception as e:
            print(f"⚠️  Could not initialize Groq: {e}")
            print("   Falling back to keyword-based classification")
    else:
        print("📝 Using keyword-based classification (--no-ml or groq not available)")
    
    # Load model
    print("\nLoading RoboBrain model...")
    model, repo_dir = get_model()
    
    # Initialize multi-turn chat
    chat = MultiTurnInference(model, repo_dir)
    
    # Settings
    use_ml = groq_classifier is not None
    enable_thinking = not args.no_thinking
    forced_task = None  # If set, overrides auto detection
    
    # Set initial image if provided
    if args.image:
        chat.set_image(args.image)
    else:
        demo_image = repo_dir / "assets/demo/grounding.jpg"
        if demo_image.exists():
            chat.set_image(str(demo_image))
            print(f"📷 Using demo image: {demo_image}")
        else:
            print("⚠️  No image set. Use /image <path> to set one.")
    
    print(f"\n🎯 Task detection: {'ML (Groq)' if use_ml else 'Keyword-based'}")
    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    print("\nJust type your question naturally!\n")
    
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
            
            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                print("Goodbye! 👋")
                break
            
            elif cmd == "/help":
                print_banner()
            
            elif cmd == "/image":
                if arg:
                    if pathlib.Path(arg).exists() or arg.startswith("http"):
                        chat.set_image(arg)
                    else:
                        print(f"❌ Image not found: {arg}")
                else:
                    print(f"📷 Current image: {chat.memory.current_image}")
            
            elif cmd == "/task":
                valid_tasks = ["general", "grounding", "affordance", "trajectory", "pointing"]
                if arg.lower() in valid_tasks:
                    forced_task = arg.lower()
                    print(f"🎯 Task forced to: {forced_task} (auto detection disabled for this)")
                elif arg.lower() == "auto":
                    forced_task = None
                    print("🎯 Task: auto detection enabled")
                else:
                    print(f"❌ Invalid task. Choose from: {', '.join(valid_tasks)}, or 'auto'")
            
            elif cmd == "/auto" or cmd == "/ml":
                if arg.lower() == "on":
                    if groq_classifier:
                        use_ml = True
                        print("🤖 ML task detection: ON (Groq LLaMA 3.3)")
                    else:
                        print("⚠️  Groq not available. Using keyword fallback.")
                elif arg.lower() == "off":
                    use_ml = False
                    print("📝 ML task detection: OFF (using keywords)")
                else:
                    print(f"🎯 Task detection: {'ML (Groq)' if use_ml else 'Keyword-based'}")
            
            elif cmd == "/thinking":
                if arg.lower() == "on":
                    enable_thinking = True
                    print("🧠 Thinking mode: ON")
                elif arg.lower() == "off":
                    enable_thinking = False
                    print("🧠 Thinking mode: OFF")
                else:
                    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            
            elif cmd == "/clear":
                chat.reset()
                forced_task = None
                print("🗑️  Conversation cleared, task reset to auto")
            
            elif cmd == "/history":
                chat.show_history()
            
            elif cmd == "/save":
                filepath = arg or "conversations/smart_chat_history.json"
                chat.save_conversation(filepath)
            
            elif cmd == "/load":
                if arg:
                    try:
                        chat.load_conversation(arg)
                    except FileNotFoundError:
                        print(f"❌ File not found: {arg}")
                else:
                    print("Usage: /load <filepath>")
            
            else:
                print(f"❌ Unknown command: {cmd}. Type /help for available commands.")
            
            continue
        
        # Regular chat message
        if not chat.memory.current_image:
            print("⚠️  No image set. Use /image <path> to set one first.")
            continue
        
        # Determine task (auto or forced)
        extra_info = {}
        if forced_task:
            current_task = forced_task
            reason = "manually forced"
            confidence = 1.0
            print(f"\n🎯 Using forced task: {current_task}")
        elif use_ml and groq_classifier:
            # Use Groq ML classifier
            current_task, confidence, reason, extra_info = groq_classifier.classify(user_input)
            if current_task is None:
                # Fallback on error
                current_task, confidence, reason, _ = classify_task_fallback(user_input)
                print(f"\n🎯 Task (fallback): {current_task.upper()} - {reason}")
            else:
                conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                print(f"\n🤖 ML-detected task: {current_task.upper()} [{conf_bar}]")
                print(f"   💡 Reason: {reason}")
                if extra_info:
                    for k, v in extra_info.items():
                        print(f"   📌 {k}: {v}")
        else:
            # Use keyword fallback
            current_task, confidence, reason, _ = classify_task_fallback(user_input)
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"\n📝 Keyword-detected task: {current_task.upper()} [{conf_bar}] ({reason})")
        
        print(f"🤖 Processing with RoboBrain...")
        
        result = chat.ask(
            user_input,
            task=current_task,
            enable_thinking=enable_thinking
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            if enable_thinking and result.get("thinking"):
                print(f"\n💭 Thinking: {result['thinking'][:300]}{'...' if len(result.get('thinking', '')) > 300 else ''}")
            
            print(f"\n🤖 Answer: {result['answer']}")
            
            if result.get("context_used"):
                print(f"   (Using conversation context from {result['turn_number']} turns)")
            
            # Visualize results for spatial tasks
            if current_task in ["pointing", "affordance", "trajectory", "grounding"]:
                vis_path = visualize_result(
                    chat.memory.current_image,
                    result['answer'],
                    current_task,
                    user_input
                )
                if vis_path:
                    print(f"📊 Visualization saved!")
        
        # Reset forced task after use (one-shot override)
        if forced_task:
            print(f"   (Forced task '{forced_task}' used once, reverting to auto)")
            forced_task = None
        
        print()


if __name__ == "__main__":
    main()
