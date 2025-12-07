# RoboBrain 2.0

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.42+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Vision-Language Model for Robotic Understanding and Interaction**

[Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Troubleshooting](#troubleshooting)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Method A: Conda Environment (Recommended)](#method-a-conda-environment-recommended)
  - [Method B: Virtual Environment](#method-b-virtual-environment)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
  - [Task Scripts](#task-scripts)
  - [Multi-Turn Conversation](#multi-turn-conversation)
  - [Interactive Chat Interface](#interactive-chat-interface)
  - [Programmatic API](#programmatic-api)
- [Local Model Weights](#local-model-weights)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

RoboBrain 2.0 is a state-of-the-art vision-language model designed for robotic perception and interaction tasks. This repository provides a modular Python implementation with support for:

- Visual question answering
- Object grounding (bounding box detection)
- Affordance prediction
- Trajectory generation
- Pointing tasks
- Multi-turn conversational memory

The implementation supports both cloud-based model inference via Hugging Face and local weight deployment for offline operation.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Task Support** | General QA, grounding, affordance, trajectory, and pointing |
| **Multi-Turn Memory** | Maintains conversation context across multiple queries |
| **Local Inference** | Run entirely offline with pre-downloaded weights |
| **Interactive CLI** | Command-line chat interface for rapid prototyping |
| **Modular Design** | Separate scripts for each task type |
| **Conversation Persistence** | Save and load conversations as JSON files |
| **Thinking Mode** | Optional chain-of-thought reasoning display |

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16 GB+ |
| System RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |

### Software

- Python 3.10 or higher
- CUDA 11.8 or higher (for GPU acceleration)
- Git

### Dependencies

Core dependencies are managed via `requirements.txt`:

```
transformers>=4.42.0
accelerate>=0.30.0
torch>=2.1.0
timm>=0.9.16
pillow
sentencepiece
bitsandbytes
matplotlib
huggingface_hub
python-dotenv
qwen-vl-utils>=0.0.8
```

---

## Installation

### Method A: Conda Environment (Recommended)

This method ensures reproducible environments across different systems.

```bash
# Clone the repository
git clone https://github.com/YasiruDEX/Robobrain-2.0.git
cd Robobrain-2.0

# Create and activate the conda environment
conda env create -f environment.yml --force
conda activate robobrain2-env

# Install remaining pip dependencies
python -m pip install -r requirements.txt
```

Alternatively, use the provided setup script:

```bash
chmod +x scripts/setup_conda_env.sh
./scripts/setup_conda_env.sh
```

### Method B: Virtual Environment

For systems without conda:

```bash
# Clone the repository
git clone https://github.com/YasiruDEX/Robobrain-2.0.git
cd Robobrain-2.0

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verification

Verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

---

## Configuration

### Hugging Face Authentication

The model requires authentication for first-time download from Hugging Face.

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the model license at [BAAI/RoboBrain2.0-3B](https://huggingface.co/BAAI/RoboBrain2.0-3B)
4. Configure the token:

```bash
cp .env.example .env
```

Edit `.env` and add your token:

```
HF_TOKEN=hf_your_token_here
```

**Security Note**: Never commit `.env` files containing tokens. The `.gitignore` is configured to exclude this file.

---

## Quick Start

### Basic Test

Run the general QA script to verify installation:

```bash
python scripts/run_general_qa.py
```

Expected output:
```
Loading RoboBrain 2.0 model...
Model loaded successfully!
Processing image: RoboBrain2.0_lib/assets/demo/navigation.jpg
Question: What do you see in this image?
Answer: [Model response]
```

### All Task Scripts

```bash
# Visual Question Answering
python scripts/run_general_qa.py

# Object Grounding
python scripts/run_visual_grounding.py

# Affordance Prediction
python scripts/run_affordance.py

# Trajectory Generation
python scripts/run_trajectory.py

# Pointing Tasks
python scripts/run_pointing.py
```

Results are saved to the `results/` directory.

---

## Documentation

### Task Scripts

Each script in the `scripts/` directory handles a specific inference task:

| Script | Task | Output |
|--------|------|--------|
| `run_general_qa.py` | Visual question answering | Text response |
| `run_visual_grounding.py` | Object detection | Bounding box `[x1, y1, x2, y2]` |
| `run_affordance.py` | Action affordance | Affordance map |
| `run_trajectory.py` | Motion trajectory | Trajectory points |
| `run_pointing.py` | Object pointing | Point coordinates `[(x, y), ...]` |

### Multi-Turn Conversation

The multi-turn conversation system enables context-aware interactions where the model maintains memory of previous exchanges.

![Multi-Turn Conversation Example](docs/images/multiturn_example.png)

#### Architecture

```
ConversationMemory
    |
    +-- Turn (dataclass)
    |       - question: str
    |       - answer: str
    |       - task: str
    |       - timestamp: datetime
    |       - image: Optional[str]
    |
    +-- Methods
            - add_turn()
            - get_context_prompt()
            - save() / load()
            - get_conversation_summary()

MultiTurnInference
    |
    +-- model: UnifiedInference
    +-- memory: ConversationMemory
    +-- Methods
            - ask()
            - ground()
            - get_affordance()
            - get_trajectory()
            - point_at()
```

### Interactive Chat Interface

Launch the interactive chat:

```bash
python scripts/interactive_chat.py
```

With a specific image:

```bash
python scripts/interactive_chat.py --image /path/to/image.jpg
```

#### Command Reference

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/image` | `<path>` | Set image for analysis |
| `/task` | `<type>` | Switch task type |
| `/history` | - | Display conversation history |
| `/clear` | - | Clear conversation memory |
| `/save` | `<file>` | Save conversation to JSON |
| `/load` | `<file>` | Load previous conversation |
| `/thinking` | `on\|off` | Toggle chain-of-thought display |
| `/context` | `on\|off` | Toggle context injection |
| `/help` | - | Display command reference |
| `/quit` | - | Exit application |

#### Supported Task Types

| Task | Description | Output Format |
|------|-------------|---------------|
| `general` | Visual question answering | Natural language |
| `grounding` | Object bounding box | `[x1, y1, x2, y2]` |
| `affordance` | Action affordance map | Coordinate list |
| `trajectory` | Motion path | Trajectory points |
| `pointing` | Object pointing | `[(x, y), ...]` |

### Programmatic API

#### Basic Usage

```python
from scripts.utils import get_model
from scripts.conversation_memory import MultiTurnInference

# Initialize model
model, repo_dir = get_model()

# Create chat instance
chat = MultiTurnInference(model, repo_dir)

# Set image
chat.set_image("path/to/image.jpg")

# Query the model
response = chat.ask("What objects are visible?")
print(response["answer"])
```

#### Multi-Turn Conversation

```python
# First query
r1 = chat.ask("What animals are in this image?")
print(r1["answer"])  # "I can see two cats."

# Follow-up query (uses conversation context)
r2 = chat.ask("What are they doing?")
print(r2["answer"])  # "They are sleeping on the couch."

# Switch to pointing task
r3 = chat.point_at("the cats")
print(r3["answer"])  # "[(395, 186), (175, 170)]"
```

#### Conversation Persistence

```python
# Save conversation
chat.save_conversation("conversations/session_001.json")

# Load in new session
new_chat = MultiTurnInference(model, repo_dir)
new_chat.load_conversation("conversations/session_001.json")

# Continue conversation
r4 = new_chat.ask("What color are the cats?")
```

#### API Reference

**MultiTurnInference Class**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `set_image` | `path: str` | `None` | Set current image |
| `ask` | `prompt: str, task: str, enable_thinking: bool` | `dict` | Query with context |
| `ground` | `description: str` | `dict` | Get bounding box |
| `get_affordance` | `action: str` | `dict` | Get affordance map |
| `get_trajectory` | `action: str` | `dict` | Get trajectory |
| `point_at` | `description: str` | `dict` | Get point coordinates |
| `reset` | - | `None` | Clear memory |
| `save_conversation` | `path: str` | `None` | Save to JSON |
| `load_conversation` | `path: str` | `None` | Load from JSON |
| `show_history` | - | `None` | Print history |

**Response Dictionary**

```python
{
    "answer": str,           # Model response
    "turn_number": int,      # Current turn index
    "context_used": bool,    # Whether context was injected
    "thinking": Optional[str] # Chain-of-thought (if enabled)
}
```

---

## Local Model Weights

For offline operation, download and place model weights in the `weights/` directory:

### Directory Structure

```
weights/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

### Downloading Weights

Using Hugging Face CLI:

```bash
huggingface-cli download BAAI/RoboBrain2.0-3B --local-dir weights/
```

Using Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/RoboBrain2.0-3B",
    local_dir="weights/",
    token="hf_your_token"
)
```

The scripts automatically detect and use local weights when available.

---

## Project Structure

```
Robobrain-2.0/
├── Notebooks/
│   ├── robobrain2-quick-test.ipynb    # Quick test notebook
│   └── multi_turn_conversation.ipynb   # Multi-turn demo notebook
├── RoboBrain2.0_lib/                   # Cloned RoboBrain repository
│   └── assets/demo/                    # Demo images
├── conversations/                       # Saved conversations
├── docs/
│   └── images/                         # Documentation images
├── results/                            # Output visualizations
├── scripts/
│   ├── conversation_memory.py          # Multi-turn memory system
│   ├── interactive_chat.py             # CLI chat interface
│   ├── run_affordance.py               # Affordance task script
│   ├── run_general_qa.py               # QA task script
│   ├── run_pointing.py                 # Pointing task script
│   ├── run_trajectory.py               # Trajectory task script
│   ├── run_visual_grounding.py         # Grounding task script
│   ├── setup_conda_env.sh              # Environment setup script
│   ├── test_multi_turn.py              # Multi-turn test suite
│   └── utils.py                        # Shared utilities
├── weights/                            # Local model weights (optional)
├── .env.example                        # Environment template
├── .gitignore
├── environment.yml                     # Conda environment spec
├── README.md
└── requirements.txt                    # Python dependencies
```

---

## Troubleshooting

### Common Errors and Solutions

#### Error: CUDA Out of Memory

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. Close other GPU-intensive applications
2. Reduce batch size (if applicable)
3. Enable model offloading:
   ```python
   model.to("cpu")
   torch.cuda.empty_cache()
   ```
4. Use a GPU with more VRAM (16GB+ recommended)

#### Error: Module Not Found

**Symptom:**
```
ModuleNotFoundError: No module named 'qwen_vl_utils'
```

**Solution:**
```bash
pip install qwen-vl-utils>=0.0.8
```

**Symptom:**
```
ModuleNotFoundError: No module named 'decorator'
```

**Solution:**
```bash
pip install decorator pygments
```

#### Error: Hugging Face Authentication Failed

**Symptom:**
```
huggingface_hub.utils._errors.GatedRepoError: 403 Client Error
```

**Solutions:**

1. Verify token is set correctly in `.env`
2. Accept the model license at [BAAI/RoboBrain2.0-3B](https://huggingface.co/BAAI/RoboBrain2.0-3B)
3. Regenerate token if expired
4. Use local weights instead (see [Local Model Weights](#local-model-weights))

#### Error: Kernel Failed to Start (Jupyter)

**Symptom:**
```
The kernel failed to start due to the missing module 'X'
```

**Solution:**
```bash
conda activate robobrain2-env
pip install pygments decorator ipykernel
python -m ipykernel install --user --name robobrain2-env
```

Then select the `robobrain2-env` kernel in Jupyter.

#### Error: Git Clone Failed

**Symptom:**
```
fatal: destination path 'RoboBrain2.0_lib' already exists
```

**Solution:**
```bash
rm -rf RoboBrain2.0_lib
# Run script again
```

#### Error: Image Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/image.jpg'
```

**Solution:**

Verify the image path exists and is accessible:
```python
import os
print(os.path.exists("path/to/image.jpg"))
```

Use absolute paths when possible:
```python
from pathlib import Path
image_path = Path(__file__).parent / "assets" / "demo" / "image.jpg"
```

#### Error: Torch Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**

Reinstall PyTorch with the correct CUDA version:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Frequently Asked Questions

### General

**Q: What GPU is required to run RoboBrain 2.0?**

A: A GPU with at least 8GB VRAM is required. For optimal performance, 16GB+ is recommended. The model has been tested on NVIDIA RTX 3080, RTX 4090, A100, and V100 GPUs.

**Q: Can I run RoboBrain 2.0 on CPU only?**

A: While technically possible, CPU inference is extremely slow and not recommended for practical use. Expect inference times of several minutes per query.

**Q: What image formats are supported?**

A: JPEG, PNG, BMP, and WebP formats are supported. Both local file paths and HTTP/HTTPS URLs are accepted.

**Q: How do I update to the latest version?**

A: 
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Multi-Turn Conversation

**Q: How many turns can the conversation memory hold?**

A: By default, the last 10 turns are retained. This can be configured:
```python
chat = MultiTurnInference(model, repo_dir)
chat.memory.max_turns = 20  # Increase to 20 turns
```

**Q: Does multi-turn conversation affect inference speed?**

A: Minimally. The context prompt adds approximately 5-10% overhead to inference time. The benefit of contextual understanding typically outweighs this cost.

**Q: Can I disable conversation context for specific queries?**

A: Yes. Either disable globally or per-query:
```python
# Disable globally
chat.use_context = False

# Or use single-turn inference directly
from scripts.utils import get_model
model, repo_dir = get_model()
# Use model directly without MultiTurnInference wrapper
```

**Q: Are conversations persisted across sessions?**

A: Not automatically. Use `save_conversation()` before exiting and `load_conversation()` when resuming.

### Model and Weights

**Q: How large are the model weights?**

A: Approximately 6GB for the 3B parameter model.

**Q: Can I use quantized models?**

A: The current implementation uses full precision. Quantization support (INT8, INT4) is planned for future releases.

**Q: Where can I find model documentation?**

A: See the official repository at [FlagOpen/RoboBrain2.0](https://github.com/FlagOpen/RoboBrain2.0).

### Integration

**Q: Can I use RoboBrain 2.0 in a web application?**

A: Yes. Wrap the API in a REST endpoint using Flask or FastAPI:
```python
from flask import Flask, request, jsonify
from scripts.utils import get_model
from scripts.conversation_memory import MultiTurnInference

app = Flask(__name__)
model, repo_dir = get_model()
chat = MultiTurnInference(model, repo_dir)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    response = chat.ask(data["prompt"])
    return jsonify(response)
```

**Q: Is there a Docker container available?**

A: Not currently. A Dockerfile is planned for future releases.

---

## Testing

Run the test suite to verify functionality:

```bash
# Multi-turn memory tests
python scripts/test_multi_turn.py

# Quick verification
python -c "from scripts.utils import get_model; print('Import successful')"
```

### Test Coverage

| Test | Description | Command |
|------|-------------|---------|
| Unit Tests | Memory system | `python scripts/test_multi_turn.py` |
| Integration | Full pipeline | `python scripts/run_general_qa.py` |
| Import | Dependencies | `python -c "import scripts.utils"` |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings for public functions
- Include type hints where applicable
- Write tests for new functionality

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The RoboBrain 2.0 model weights are subject to the [BAAI license](https://huggingface.co/BAAI/RoboBrain2.0-3B).

---

## Acknowledgments

- [BAAI](https://www.baai.ac.cn/) for the RoboBrain 2.0 model
- [FlagOpen](https://github.com/FlagOpen) for the original implementation
- [Hugging Face](https://huggingface.co/) for model hosting

---

<div align="center">

**[Back to Top](#robobrain-20)**

</div>
