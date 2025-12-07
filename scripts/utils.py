import os
import sys
import subprocess
import pathlib
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
import torch

# Load environment variables from .env file
load_dotenv()

# Constants
REPO_URL = "https://github.com/FlagOpen/RoboBrain2.0.git"
# Clone into a 'lib' directory in the project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
REPO_DIR = PROJECT_ROOT / "RoboBrain2.0_lib"
# Local weights folder (pre-downloaded weights)
WEIGHTS_DIR = PROJECT_ROOT / "weights"

def setup_repo():
    """Clones the RoboBrain2.0 repository if it doesn't exist and adds it to sys.path."""
    if not REPO_DIR.exists():
        print(f"Cloning RoboBrain2.0 repo to {REPO_DIR}...")
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
        
        # Install requirements from the cloned repo if needed
        req_path = REPO_DIR / "requirements.txt"
        if req_path.exists():
            print(f"Installing requirements from {req_path}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)], check=False)
    
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    
    return REPO_DIR

def get_model(model_name=None, force_cpu=False, use_local_weights=True, load_in_8bit=True):
    """Loads the UnifiedInference model from local weights or HuggingFace.
    
    Args:
        model_name: HuggingFace model ID or local path. If None and use_local_weights=True,
                    uses the local weights folder.
        force_cpu: Force CPU inference even if CUDA is available.
        use_local_weights: If True and weights folder exists, load from local weights.
        load_in_8bit: Enable 8-bit quantization to reduce memory (6GB -> ~3GB).
    """
    setup_repo()
    
    # Import from the cloned repo
    try:
        from inference import UnifiedInference
    except ImportError:
        # Try adding the path again or check if clone was successful
        sys.path.insert(0, str(REPO_DIR))
        from inference import UnifiedInference

    # Determine model path: use local weights if available and requested
    if use_local_weights and WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "config.json").exists():
        model_path = str(WEIGHTS_DIR)
        print(f"Using local weights from: {model_path}")
    else:
        # Fall back to HuggingFace model
        model_path = model_name or "BAAI/RoboBrain2.0-3B"
        print(f"Using HuggingFace model: {model_path}")
        
        # Login to HuggingFace if token available (needed for gated models)
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Warning: HF_TOKEN not found in environment variables. Please set it in a .env file or export it.")
            print("You might need it for the gated model.")
        else:
            login(token=token)

    device = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"
    device_map = "auto" if device == "cuda" else "cpu"
    print(f"Using device: {device} — model path: {model_path}")

    # Clear GPU cache before loading model to avoid OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"GPU Memory before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    # Note: load_in_8bit is not supported by UnifiedInference
    # The model uses torch_dtype="auto" by default for quantization
    model = UnifiedInference(model_path, device_map=device_map)
    print("✅ Model loaded.")
    
    if torch.cuda.is_available():
        print(f"GPU Memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    return model, REPO_DIR
