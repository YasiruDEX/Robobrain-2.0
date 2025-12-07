# RoboBrain 2.0 - System Overview

## Overview
A Vision-Language AI system for robotic perception tasks using Qwen2.5-VL-7B-Instruct model with Auto Mode task classification powered by Groq's Llama-3.3-70B.

## Features

### 🤖 Auto Mode with AI Task Detection
- Automatic task classification using Groq API (Llama-3.3-70B)
- UI feedback showing detected task with sparkle icon
- Supports 5 task types: General Chat, Grounding, Affordance, Trajectory, Pointing

### 🎯 Vision-Language Tasks
1. **General Chat**: Q&A about images
2. **Grounding**: Object detection with bounding boxes
3. **Affordance**: Predict actionable areas for robot manipulation
4. **Trajectory**: Plan robot arm movement paths
5. **Pointing**: Identify specific points on objects

### 🧠 Memory Management
- 8-bit quantization for GPU memory efficiency
- 4.8GB GPU memory limit with expandable segments
- Aggressive garbage collection after inference
- CPU offload for model components
- Automatic image resizing to 2000px max dimension

### 🎨 Frontend Features
- **Client-side Annotation**: Canvas API fallback when backend parsing fails
- **Dark Mode**: Persistent theme switching
- **Responsive Design**: Tailwind CSS with smooth gradients
- **Image Upload**: Automatic resize before processing
- **Auto-save**: Messages persist to localStorage on change
- **Session Restoration**: Continues previous conversation on reload

### 💾 Chat History System
- **Local Storage**: Saves conversations with images (base64 encoded)
- **Session Management**: Unique IDs for each conversation
- **History Viewer**: Modal UI to browse past conversations
- **Metadata Tracking**: Task type, timestamps, image presence
- **Delete & Load**: Manage saved conversations
- **New Chat**: Saves current session before starting fresh

### 🔧 Technical Stack
- **Backend**: Flask API (Python)
- **Model**: Qwen2.5-VL-7B-Instruct (8-bit quantization)
- **AI Classification**: Groq Llama-3.3-70B
- **Frontend**: React + Vite + Tailwind CSS
- **Storage**: Browser localStorage
- **Annotation**: HTML5 Canvas API

## Architecture

### Backend (Port 5001)
```
backend.py
├── Flask REST API
├── Session management
├── Groq integration (Auto Mode)
└── Model inference coordination

RoboBrain2.0_lib/inference.py
├── Qwen2.5-VL model loading
├── 8-bit quantization setup
├── Coordinate extraction (regex)
├── Image annotation (PIL)
└── Memory management
```

### Frontend (Vite Dev Server)
```
frontend/src/
├── App.jsx (Main state & session management)
├── components/
│   ├── ChatContainer.jsx (Message display & input)
│   ├── Header.jsx (Status & controls)
│   ├── Sidebar.jsx (Task selection & image upload)
│   └── HistorySidebar.jsx (Chat history viewer)
├── utils/
│   ├── chatHistory.js (localStorage management)
│   └── imageAnnotation.js (Canvas drawing & parsing)
└── api.js (Backend communication)
```

## Data Flow

### Message Sending
1. User sends message + optional image
2. Frontend resizes image if > 2000px
3. POST to `/api/chat` with sessionId, task, message, image
4. Backend classifies task (if Auto Mode) via Groq
5. Backend runs inference with Qwen2.5-VL
6. Backend extracts coordinates & annotates image
7. Response includes message, image URL, task_source
8. Frontend displays result
9. If backend annotation fails, frontend draws using Canvas
10. Auto-saves to localStorage

### Session Management
1. On load: Check localStorage for previous session
2. If exists: Restore sessionId + messages
3. If not: Create new session via `/api/session`
4. On message change: Auto-save to localStorage
5. On New Chat: Save current session, create new one
6. On History button: Show HistorySidebar modal
7. On Load Session: Restore selected conversation

## API Endpoints

### POST /api/session
Create new chat session
- **Returns**: `{ session_id: string }`

### POST /api/chat
Send message for processing
- **Body**: `{ message, image?, task }`
- **Returns**: `{ reply, output_image?, task_source? }`

### GET /api/health
Check backend status
- **Returns**: `{ status: "healthy", model_loaded: bool }`

### GET /api/tasks
Get available task types
- **Returns**: `{ tasks: Array<Task> }`

### GET /api/history/:sessionId
Get conversation history
- **Returns**: `Array<Message>`

## Configuration

### Environment Variables
```bash
# Frontend (.env)
VITE_API_URL=http://localhost:5001

# Backend
GROQ_API_KEY=<your-groq-api-key>
```

### Model Settings
- **Device**: CUDA (GPU)
- **Precision**: 8-bit quantization (bitsandbytes)
- **Memory Limit**: 4.8GB per GPU
- **CPU Offload**: Enabled for larger inputs
- **Max Tokens**: 1024 (output)

### Image Processing
- **Max Dimension**: 2000px (auto-resize)
- **Format**: JPEG/PNG
- **Storage**: Base64 in localStorage
- **Annotation**: Canvas API with 2px red strokes

## Memory Optimizations

1. **Model Loading**
   - 8-bit quantization reduces memory by ~50%
   - CPU offload for attention layers
   - Lazy loading of model components

2. **Inference**
   - Gradient disabled (`torch.no_grad()`)
   - Empty CUDA cache after generation
   - Aggressive garbage collection
   - Thread locks prevent concurrent loads

3. **Image Processing**
   - Client-side resize before upload
   - Efficient PIL operations
   - Minimal tensor conversions

## Chat History Storage

### localStorage Schema
```javascript
{
  "robobrain_chat_history": {
    "<sessionId>": {
      "sessionId": "uuid",
      "messages": [
        {
          "role": "user|assistant",
          "content": "text",
          "image": "base64string",
          "timestamp": "ISO8601",
          "metadata": {...}
        }
      ],
      "metadata": {
        "task": "auto|general|grounding|...",
        "hasImage": true|false,
        "createdAt": "ISO8601",
        "updatedAt": "ISO8601"
      }
    }
  },
  "robobrain_current_session": "<currentSessionId>"
}
```

## UI Components

### Header
- Connection status (Online/Offline)
- Session ID display
- Dark mode toggle
- History button
- Sidebar toggle

### Sidebar
- Task selection dropdown
- Image upload
- New Chat button
- Current image preview
- Task descriptions

### Chat Container
- Message list (scrollable)
- User/Assistant bubbles
- Image display with annotations
- Task detection badge
- Input field
- Send button

### History Sidebar (Modal)
- List of all sessions
- Preview text & metadata
- Timestamp display
- Load session button
- Delete session button
- Close button

## Development

### Start Backend
```bash
cd /home/yasiru/Documents/github/Robobrain-2.0
python backend.py
```

### Start Frontend
```bash
cd frontend
npm run dev
```

### Environment Setup
1. Install Python dependencies
2. Set up Groq API key
3. Install npm packages
4. Configure VITE_API_URL

## Known Behaviors

### Output Parsing
- Backend uses regex to extract coordinates
- Supports: `[x1, y1, x2, y2]` format for boxes
- Supports: `(x, y)` format for points
- Falls back to frontend Canvas annotation if parsing fails

### Memory Limits
- ~4.8GB GPU memory allocated
- OOM errors trigger cleanup
- Large images auto-resize to prevent crashes
- Expandable segments enabled for PyTorch

### Session Persistence
- Sessions stored in browser localStorage only
- No server-side storage (stateless backend)
- History survives page refresh
- Clearing browser data removes history

## Credits
**Developed by**: Yasiru (200760N) & Janidu (200277V)
**Course**: EN4554 - Deep Learning for Vision
**Institution**: University of Moratuwa, Sri Lanka

## Future Enhancements
- [ ] Server-side chat history storage
- [ ] Export conversations to JSON/PDF
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Batch image processing
- [ ] Fine-tuning on custom datasets
- [ ] Real-time robot control integration
