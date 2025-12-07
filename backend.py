import sys
import os
import pathlib
import uuid
import signal
import psutil
import atexit

# Set PyTorch memory configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import torch
import gc
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Add scripts to path to import utils and conversation_memory
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from utils import get_model
from conversation_memory import MultiTurnInference

def classify_task_with_groq(prompt):
    """Classify the user prompt into a task using Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found. Defaulting to 'general'.")
        return "general"
    
    try:
        client = Groq(api_key=api_key)
        
        system_prompt = """
        You are an intelligent agent that classifies user prompts into specific robotic tasks.
        Available tasks:
        - general: General Q&A, conversation, describing images.
        - grounding: Finding specific objects in an image (e.g., 'find the apple', 'where is the cup', 'detect the bottle').
        - affordance: Determining how to interact with objects (e.g., 'how do I grasp this?', 'where can I sit?', 'grasping points').
        - trajectory: Planning a path or movement (e.g., 'draw a path to the door', 'how to move to the table', 'navigation').
        - pointing: Pointing to a specific location (e.g., 'point to the handle', 'center of the object').
        
        Return ONLY the task ID (general, grounding, affordance, trajectory, pointing) as a plain string. Do not output JSON or markdown.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=10,
        )
        task = chat_completion.choices[0].message.content.strip().lower()
        valid_tasks = ['general', 'grounding', 'affordance', 'trajectory', 'pointing']
        if task not in valid_tasks:
            return 'general'
        return task
    except Exception as e:
        print(f"Groq classification error: {e}")
        return 'general'

def cleanup_old_processes():
    """Kill any existing backend.py processes to free GPU memory."""
    current_pid = os.getpid()
    parent_pid = os.getppid()
    current_script = os.path.abspath(__file__)
    killed_count = 0
    
    print("Checking for old backend.py processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
        try:
            proc_pid = proc.info['pid']
            proc_ppid = proc.info.get('ppid')
            
            # Skip current process, parent process, and sibling processes (Flask reloader)
            if proc_pid == current_pid or proc_pid == parent_pid or proc_ppid == parent_pid:
                continue
            
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('backend.py' in str(arg) for arg in cmdline):
                print(f"  Killing old backend process: PID {proc_pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_count > 0:
        print(f"  Killed {killed_count} old process(es)")
        # Give GPU time to free memory
        import time
        time.sleep(2)
    else:
        print("  No old processes found")

def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    print("Cleaning GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Show GPU status
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

def cleanup_on_exit():
    """Cleanup function to run when backend exits."""
    print("\nShutting down backend...")
    cleanup_gpu()
    print("GPU memory cleaned. Goodbye!")

# Register cleanup handler
atexit.register(cleanup_on_exit)

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\nReceived interrupt signal...")
    cleanup_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Cleanup before initializing
# cleanup_old_processes()
# cleanup_gpu()

# Global model instance
print("\nInitializing model... This may take a while.")
try:
    # Load model using the utility from scripts/utils.py
    # This will use local weights if available in 'weights/' folder
    # Note: UnifiedInference uses torch_dtype="auto" for optimal memory usage
    model, repo_dir = get_model()
    print("Model initialized successfully.")
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None
    repo_dir = None

# Sessions store: maps session_id to MultiTurnInference instance
sessions = {}

# Inference lock to prevent concurrent inference requests (GPU OOM prevention)
inference_lock = threading.Lock()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "active_sessions": len(sessions)
    })

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new chat session."""
    if model is None:
        return jsonify({"error": "Model not initialized"}), 503
        
    session_id = str(uuid.uuid4())
    sessions[session_id] = MultiTurnInference(model, repo_dir)
    print(f"Created session: {session_id}")
    return jsonify({"session_id": session_id, "sessionId": session_id}) # Return both for compatibility

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session."""
    if session_id in sessions:
        del sessions[session_id]
        print(f"Deleted session: {session_id}")
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/session/<session_id>/reset', methods=['POST'])
def reset_session(session_id):
    """Reset conversation history for a session."""
    if session_id in sessions:
        sessions[session_id].reset()
        return jsonify({"status": "reset"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload an image file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Use UUID to prevent overwrites and ensure uniqueness
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Return absolute path for the model to use
        abs_path = os.path.abspath(filepath)
        return jsonify({
            "path": abs_path, 
            "filename": unique_filename,
            "url": f"/uploads/{unique_filename}" # In case we want to serve it later
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the model."""
    if model is None:
        return jsonify({"error": "Model not initialized"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Handle both camelCase (JS) and snake_case (Python)
    session_id = data.get('session_id') or data.get('sessionId')
    message = data.get('message')
    image_path = data.get('image')
    task = data.get('task', 'general')
    enable_thinking = data.get('enable_thinking', True)
    
    detected_task_source = None
    if task == 'auto':
        print(f"Auto-detecting task for prompt: {message}")
        detected_task = classify_task_with_groq(message)
        print(f"Detected task: {detected_task}")
        task = detected_task
        detected_task_source = "auto"

    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
        
    if session_id not in sessions:
        # Auto-create session if it doesn't exist (optional, but helpful)
        sessions[session_id] = MultiTurnInference(model, repo_dir)
    
    chat_session = sessions[session_id]
    
    # Use lock to ensure only one inference runs at a time (prevent GPU OOM)
    # Use lock to ensure only one inference runs at a time (prevent GPU OOM)
    with inference_lock:
        # Clear GPU cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
        # If image is provided, update the current image for the session
        if image_path:
            # Check if it's a full path or just a filename in uploads
            if not os.path.exists(image_path):
                potential_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
                if os.path.exists(potential_path):
                    image_path = potential_path
                else:
                    # If image not found, but we have a current image, maybe warn? 
                    # For now, if explicit image path fails, return error.
                    return jsonify({"error": f"Image not found: {image_path}"}), 400
            chat_session.set_image(image_path)
        
        # If no image is set in session and none provided, we can't proceed for visual tasks
        if not chat_session.memory.current_image and task != 'general':
            # Note: 'general' task might work without image if it's just text chat, 
            # but RoboBrain is VLM, so it usually expects an image.
            # However, MultiTurnInference.ask checks for image.
            pass

        print(f"Processing chat for session {session_id}: {message[:50]}...")
        
        # Determine if we should plot
        should_plot = task in ['pointing', 'affordance', 'trajectory', 'grounding']

        # Run inference inside the lock to prevent concurrent GPU usage
        try:
            response = chat_session.ask(
                prompt=message,
                task=task,
                enable_thinking=enable_thinking,
                plot=should_plot
            )
        finally:
            # Clean up GPU memory after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
    
    # If there is an output image, convert path to URL
    if response.get('output_image'):
        filename = os.path.basename(response['output_image'])
        response['output_image'] = f"/result/{filename}"

    # Add the task that was actually used
    response['task'] = task
    if detected_task_source:
        response['task_source'] = detected_task_source

    return jsonify(response)

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get conversation history."""
    if session_id in sessions:
        # Convert turns to serializable format
        history = []
        for turn in sessions[session_id].memory.turns:
            history.append({
                "role": turn.role,
                "content": turn.content,
                "image_path": turn.image_path,
                "task": turn.task,
                "timestamp": turn.timestamp,
                "metadata": turn.metadata
            })
        return jsonify(history)
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get available tasks."""
    tasks = [
        {
            "id": "auto",
            "name": "Auto Mode",
            "description": "Automatically detect task type using AI"
        },
        {
            "id": "general",
            "name": "General Chat",
            "description": "General conversation and Q&A about the image"
        },
        {
            "id": "grounding",
            "name": "Grounding",
            "description": "Locate objects in the image with bounding boxes"
        },
        {
            "id": "affordance",
            "name": "Affordance",
            "description": "Predict actionable areas for robot manipulation"
        },
        {
            "id": "trajectory",
            "name": "Trajectory",
            "description": "Plan robot arm movement paths"
        },
        {
            "id": "pointing",
            "name": "Pointing",
            "description": "Identify specific points on objects"
        }
    ]
    return jsonify({"tasks": tasks})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RoboBrain 2.0 Backend Starting")
    print("="*60)
    print(f"Server: http://127.0.0.1:5001")
    print(f"Network: http://192.168.1.166:5001")
    print(f"Model loaded: {model is not None}")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("="*60 + "\n")
    
    # Run without debug mode to avoid reloader issues
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
