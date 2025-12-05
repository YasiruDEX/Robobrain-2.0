#!/usr/bin/env python3
"""
Interactive Multi-Turn Chat with RoboBrain 2.0

Run this script to have a conversation with RoboBrain about images.
The model remembers previous exchanges and maintains context.

Usage:
    python scripts/interactive_chat.py
    python scripts/interactive_chat.py --image path/to/image.jpg
"""

import argparse
import sys
import pathlib
import re
import ast
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from utils import get_model
from conversation_memory import MultiTurnInference

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


def parse_coordinates(answer, task):
    """Parse coordinates from model answer based on task type."""
    try:
        # Try to find list patterns in the answer
        # Look for patterns like [(x,y), ...] or [[x,y], ...] or [x1,y1,x2,y2]
        
        if task == "pointing":
            # Look for tuples like (x, y) or [(x, y)]
            pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x), float(y)) for x, y in matches]
        
        elif task == "affordance" or task == "grounding":
            # Look for bounding box [x1, y1, x2, y2]
            pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
            matches = re.findall(pattern, answer)
            if matches:
                return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in matches]
        
        elif task == "trajectory":
            # Look for list of points [[x,y], [x,y], ...] or [(x,y), ...]
            # Try bracket format first
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
    
    # Create a copy for drawing
    vis_img = img.copy()
    
    if task == "pointing":
        # Draw points
        for i, (x, y) in enumerate(coords):
            # Convert normalized coordinates if needed (0-1 range)
            if x <= 1 and y <= 1:
                px, py = int(x * w), int(y * h)
            else:
                px, py = int(x), int(y)
            
            # Draw point with circle and crosshair
            cv2.circle(vis_img, (px, py), 15, (0, 255, 0), 3)
            cv2.circle(vis_img, (px, py), 5, (0, 0, 255), -1)
            cv2.line(vis_img, (px - 20, py), (px + 20, py), (0, 255, 0), 2)
            cv2.line(vis_img, (px, py - 20), (px, py + 20), (0, 255, 0), 2)
            
            # Label
            cv2.putText(vis_img, f"P{i+1}({px},{py})", (px + 10, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    elif task == "affordance" or task == "grounding":
        # Draw bounding boxes
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, (x1, y1, x2, y2) in enumerate(coords):
            color = colors[i % len(colors)]
            
            # Convert coordinates (assuming they might be absolute pixel values)
            if all(v <= 1 for v in [x1, y1, x2, y2]):
                # Normalized coordinates
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{task.upper()} {i+1}" if len(coords) > 1 else task.upper()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis_img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    elif task == "trajectory":
        # Draw trajectory path
        if len(coords) >= 2:
            points = []
            for x, y in coords:
                # Convert normalized coordinates if needed
                if x <= 1 and y <= 1:
                    px, py = int(x * w), int(y * h)
                else:
                    px, py = int(x), int(y)
                points.append((px, py))
            
            # Draw path lines
            for i in range(len(points) - 1):
                # Gradient color from green to red
                ratio = i / (len(points) - 1)
                color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
                cv2.line(vis_img, points[i], points[i + 1], color, 3)
            
            # Draw waypoints
            for i, (px, py) in enumerate(points):
                if i == 0:
                    # Start point - green
                    cv2.circle(vis_img, (px, py), 12, (0, 255, 0), -1)
                    cv2.putText(vis_img, "START", (px + 15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif i == len(points) - 1:
                    # End point - red
                    cv2.circle(vis_img, (px, py), 12, (0, 0, 255), -1)
                    cv2.putText(vis_img, "END", (px + 15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Intermediate points
                    cv2.circle(vis_img, (px, py), 8, (255, 165, 0), -1)
                
                # Point number
                cv2.putText(vis_img, str(i + 1), (px - 5, py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add title and query
    title = f"Task: {task.upper()} | Query: {query[:50]}{'...' if len(query) > 50 else ''}"
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Save result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{task}_{timestamp}.png"
    
    # Convert BGR to RGB for saving
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # Save with matplotlib for better quality
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(vis_img_rgb)
    ax.axis('off')
    ax.set_title(f"{task.upper()}: {query}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Result saved to: {output_path}")
    
    # Also display with matplotlib
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
║         🤖 RoboBrain 2.0 Interactive Chat 🤖                ║
║                                                              ║
║  Tasks (use /task <name> to switch):                        ║
║    general    - Ask ANY question about the image            ║
║    grounding  - Find objects (shows bounding box)           ║
║    affordance - Detect graspable areas (shows bbox)         ║
║    trajectory - Plan robot paths (shows waypoints)          ║
║    pointing   - Identify locations (shows points)           ║
║                                                              ║
║  📊 Results auto-saved to: results/interactive/             ║
║                                                              ║
║  Commands:                                                   ║
║    /image <path>  - Set a new image                         ║
║    /task <type>   - Switch task type                        ║
║    /history       - Show conversation history               ║
║    /clear         - Clear conversation memory               ║
║    /save <file>   - Save conversation to file               ║
║    /load <file>   - Load conversation from file             ║
║    /thinking on   - Enable thinking mode                    ║
║    /thinking off  - Disable thinking mode                   ║
║    /help          - Show this help                          ║
║    /quit          - Exit                                    ║
║                                                              ║
║  💡 Just type any question to chat with the model!          ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with RoboBrain 2.0")
    parser.add_argument("--image", "-i", type=str, help="Initial image to analyze")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()
    
    print_banner()
    
    # Load model
    print("Loading RoboBrain model...")
    model, repo_dir = get_model()
    
    # Initialize multi-turn chat
    chat = MultiTurnInference(model, repo_dir)
    
    # Settings
    current_task = "general"
    enable_thinking = not args.no_thinking
    
    # Set initial image if provided
    if args.image:
        chat.set_image(args.image)
    else:
        # Use a demo image by default
        demo_image = repo_dir / "assets/demo/grounding.jpg"
        if demo_image.exists():
            chat.set_image(str(demo_image))
            print(f"📷 Using demo image: {demo_image}")
        else:
            print("⚠️  No image set. Use /image <path> to set one.")
    
    print(f"\n🎯 Current task: {current_task}")
    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    print("\nType your question or /help for commands.\n")
    
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
                    # Check if file exists
                    if pathlib.Path(arg).exists() or arg.startswith("http"):
                        chat.set_image(arg)
                    else:
                        print(f"❌ Image not found: {arg}")
                else:
                    print(f"📷 Current image: {chat.memory.current_image}")
            
            elif cmd == "/task":
                valid_tasks = ["general", "grounding", "affordance", "trajectory", "pointing"]
                if arg.lower() in valid_tasks:
                    current_task = arg.lower()
                    print(f"🎯 Task set to: {current_task}")
                else:
                    print(f"❌ Invalid task. Choose from: {', '.join(valid_tasks)}")
            
            elif cmd == "/history":
                chat.show_history()
            
            elif cmd == "/clear":
                chat.reset()
            
            elif cmd == "/save":
                filepath = arg or "conversations/chat_history.json"
                chat.save_conversation(filepath)
            
            elif cmd == "/load":
                if arg:
                    try:
                        chat.load_conversation(arg)
                    except FileNotFoundError:
                        print(f"❌ File not found: {arg}")
                else:
                    print("Usage: /load <filepath>")
            
            elif cmd == "/thinking":
                if arg.lower() == "on":
                    enable_thinking = True
                    print("🧠 Thinking mode: ON")
                elif arg.lower() == "off":
                    enable_thinking = False
                    print("🧠 Thinking mode: OFF")
                else:
                    print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            
            elif cmd == "/context":
                if arg.lower() == "on":
                    chat.use_context = True
                    print("📝 Context mode: ON (using conversation history)")
                elif arg.lower() == "off":
                    chat.use_context = False
                    print("📝 Context mode: OFF (each query is independent)")
                else:
                    print(f"📝 Context mode: {'ON' if chat.use_context else 'OFF'}")
            
            else:
                print(f"❌ Unknown command: {cmd}. Type /help for available commands.")
            
            continue
        
        # Regular chat message
        if not chat.memory.current_image:
            print("⚠️  No image set. Use /image <path> to set one first.")
            continue
        
        print(f"\n🤖 Thinking... (task: {current_task})")
        
        result = chat.ask(
            user_input,
            task=current_task,
            enable_thinking=enable_thinking
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Show thinking if available
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
        
        print()


if __name__ == "__main__":
    main()
