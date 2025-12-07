# RoboBrain 2.0

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/react-18.2-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Vision-Language Model for Robotic Understanding and Interaction**

*Advanced multi-modal AI system with interactive web interface, automatic task detection, and optimized memory management*

[Installation](#installation) | [Quick Start](#quick-start) | [Web Interface](#web-interface) | [Troubleshooting](#troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Backend Setup](#2-backend-setup)
  - [3. Frontend Setup](#3-frontend-setup)
  - [4. Environment Configuration](#4-environment-configuration)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Web Interface](#web-interface)
  - [Auto Mode (AI Task Detection)](#auto-mode-ai-task-detection)
  - [CLI Scripts](#cli-scripts)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

---

## 🌟 Overview

RoboBrain 2.0 is a state-of-the-art vision-language model designed for robotic perception and interaction tasks. This implementation provides:

- **Interactive Web Interface**: Modern React-based chat UI with dark mode support
- **Automatic Task Detection**: AI-powered task classification using Groq's Llama 3.3
- **Multi-Turn Conversations**: Persistent conversation history with image context
- **Optimized Memory Management**: 8-bit quantization for efficient GPU usage
- **Local & Cloud Support**: Run offline with downloaded weights or use Hugging Face

The system combines Qwen2.5-VL (7B/32B) for vision-language understanding with a Flask backend and React frontend for seamless interaction.

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **General QA** | Answer questions about images with natural language |
| **Object Grounding** | Detect and localize objects with bounding boxes |
| **Affordance Prediction** | Identify interaction points for robotic manipulation |
| **Trajectory Generation** | Plan motion paths for task completion |
| **Pointing Tasks** | Localize specific points of interest in images |

### Technical Features

- 🤖 **Auto Mode**: Automatically detects task type from natural language prompts
- 💬 **Multi-Turn Memory**: Maintains context across conversation
- 🎨 **Visual Output**: Generates annotated images for spatial tasks
- 🧠 **Thinking Mode**: Optional chain-of-thought reasoning display
- 🔒 **Session Management**: Independent conversation sessions
- 📦 **8-bit Quantization**: Reduces memory from ~6GB to ~3GB
- 🌙 **Dark Mode UI**: Easy on the eyes for extended use

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│  • Modern chat interface with dark mode                 │
│  • Image upload and preview                             │
│  • Task selection (Auto/Manual modes)                   │
│  • Real-time response streaming                         │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────┐
│                  Backend (Flask)                         │
│  • Session management                                    │
│  • Auto task detection (Groq Llama 3.3)                 │
│  • Inference orchestration                              │
│  • Memory optimization & cleanup                        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│            RoboBrain2.0 Inference Engine                 │
│  • Qwen2.5-VL model (7B/32B)                            │
│  • 8-bit quantization with bitsandbytes                 │
│  • Multi-turn conversation memory                       │
│  • Visual annotation & plotting                         │
└─────────────────────────────────────────────────────────┘
```

---

## 💻 Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 6 GB (RTX 2060) | 8+ GB (RTX 3070+) |
| **System RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB free | 50 GB free |
| **CUDA** | 11.8+ | 12.1+ |

### Software

- **Python** 3.10 or higher
- **Node.js** 16.x or higher
- **npm** 8.x or higher
- **CUDA** 11.8+ (for GPU acceleration)
- **Git**

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YasiruDEX/Robobrain-2.0.git
cd Robobrain-2.0
```

### 2. Backend Setup

#### Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate robobrain2-env

# Install additional dependencies
pip install -r requirements.txt
```

#### Using pip + venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Hugging Face token (optional, for cloud model access)
HF_TOKEN=hf_your_token_here

# Groq API key (required for Auto Mode)
GROQ_API_KEY=your_groq_api_key_here
```

**Get API Keys:**
- **Hugging Face**: https://huggingface.co/settings/tokens
- **Groq**: https://console.groq.com/keys

---

## 🎯 Usage

### Starting the Application

#### 1. Start Backend

```bash
# Using the convenience script
./run_backend.sh

# Or manually
conda activate robobrain2-env
python backend.py
```

The backend will start on `http://localhost:5001`

#### 2. Start Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The web interface will open at `http://localhost:5173`

### Web Interface

1. **Create Session**: Click "New Chat" to start a conversation
2. **Upload Image** (optional): Click the image icon to upload
3. **Select Mode**:
   - **Auto Mode**: AI automatically detects the task type
   - **Manual Mode**: Choose specific task (General/Grounding/Affordance/Trajectory/Pointing)
4. **Send Message**: Type your question and press Enter

#### Auto Mode (AI Task Detection)

When Auto Mode is enabled, the system uses Groq's Llama 3.3 to automatically classify your prompt:

- "Where is the apple?" → **Grounding**
- "How can I grab this?" → **Affordance**
- "Plan a path to reach the cup" → **Trajectory**
- "Point to all the chairs" → **Pointing**
- "What color is the table?" → **General QA**

The detected task is displayed in the response with a ✨ sparkle icon.

### CLI Scripts

For command-line usage or testing:

```bash
# General question answering
python scripts/general.py --image path/to/image.jpg --prompt "What is in this image?"

# Object grounding
python scripts/grounding.py --image path/to/image.jpg --object "red apple"

# Affordance prediction
python scripts/affordance.py --image path/to/image.jpg --task "pick up the cup"

# Trajectory generation
python scripts/trajectory.py --image path/to/image.jpg --task "move to the door"

# Multi-turn conversation
python scripts/multi_turn.py
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5001/api
```

### Endpoints

#### Create Session
```http
POST /session
```

**Response:**
```json
{
  "session_id": "uuid",
  "sessionId": "uuid"
}
```

#### Send Message
```http
POST /chat
```

**Request Body:**
```json
{
  "session_id": "uuid",
  "message": "What is in this image?",
  "image": "filename.jpg",
  "task": "auto",
  "enable_thinking": true
}
```

**Response:**
```json
{
  "answer": "The image shows...",
  "thinking": "[[coordinates]]",
  "output_image": "/result/annotated.jpg",
  "task": "grounding",
  "task_source": "auto"
}
```

#### Upload Image
```http
POST /upload
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "path": "/absolute/path/to/image.jpg",
  "filename": "uuid_image.jpg",
  "url": "/uploads/uuid_image.jpg"
}
```

#### Delete Session
```http
DELETE /session/<session_id>
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_sessions": 2
}
```

---

## 📁 Project Structure

```
Robobrain-2.0/
├── backend.py                 # Flask API server
├── RoboBrain2.0_lib/          # Core inference library
│   ├── inference.py           # Model loading & inference
│   └── multi_turn.py          # Conversation memory
├── scripts/                   # CLI task scripts
│   ├── general.py
│   ├── grounding.py
│   ├── affordance.py
│   ├── trajectory.py
│   ├── multi_turn.py
│   └── utils.py               # Model utilities
├── frontend/                  # React web interface
│   ├── src/
│   │   ├── components/        # UI components
│   │   │   ├── ChatContainer.jsx
│   │   │   ├── Message.jsx
│   │   │   └── Sidebar.jsx
│   │   ├── api.js             # Backend API client
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── uploads/                   # Uploaded images
├── result/                    # Generated output images
├── conversations/             # Saved conversation JSON
├── weights/                   # Local model weights (optional)
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment spec
├── .env                       # API keys (not in git)
├── .env.example               # Template for .env
└── README.md
```

---

## 🔧 Troubleshooting

### GPU Out of Memory

**Symptoms:** `CUDA out of memory` error during inference

**Solutions:**
1. The system automatically uses 8-bit quantization and reserves ~800MB headroom
2. If still failing, reduce image resolution before uploading
3. Close other GPU applications (browsers, games, etc.)
4. Restart the backend to clear GPU cache:
   ```bash
   pkill -9 python
   python backend.py
   ```

### Model Not Loading

**Symptoms:** `Model loaded: False` on startup

**Solutions:**
1. Check if weights exist in `weights/` directory
2. Verify Hugging Face token in `.env` if using cloud weights
3. Ensure sufficient disk space (20GB+)
4. Check CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Frontend Not Connecting

**Symptoms:** "Failed to fetch" or connection errors in browser

**Solutions:**
1. Verify backend is running on port 5001:
   ```bash
   curl http://localhost:5001/api/health
   ```
2. Check if port 5001 is blocked by firewall
3. Ensure CORS is enabled (already configured in `backend.py`)

### Auto Mode Not Working

**Symptoms:** Task detection fails or returns "general" for all prompts

**Solutions:**
1. Verify `GROQ_API_KEY` is set in `.env`
2. Check Groq API quota: https://console.groq.com/
3. Test Groq connection:
   ```bash
   python -c "from groq import Groq; import os; from dotenv import load_dotenv; load_dotenv(); client = Groq(api_key=os.getenv('GROQ_API_KEY')); print('Connected')"
   ```

### Port Already in Use

**Symptoms:** `Address already in use` when starting backend

**Solutions:**
```bash
# Find and kill process using port 5001
lsof -ti:5001 | xargs kill -9

# Or use the provided script
./kill_backend.sh
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

**Developed by:** Yasiru Jayasooriya ([@YasiruDEX](https://github.com/YasiruDEX))

**Built with:**
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL) - Vision-Language Model
- [Groq](https://groq.com/) - Fast LLM Inference for Auto Mode
- [React](https://reactjs.org/) - Frontend Framework
- [Flask](https://flask.palletsprojects.com/) - Backend API
- [Transformers](https://huggingface.co/docs/transformers) - Model Library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 8-bit Quantization

**Special Thanks:**
- BAAI Team for RoboBrain model architecture
- Hugging Face for model hosting and tools
- The open-source AI community

---

<div align="center">

**⭐ Star this repo if you find it useful!**

[Report Bug](https://github.com/YasiruDEX/Robobrain-2.0/issues) · [Request Feature](https://github.com/YasiruDEX/Robobrain-2.0/issues)

</div>
