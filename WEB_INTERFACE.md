# RoboBrain 2.0 Web Interface

A professional web interface for interacting with the RoboBrain 2.0 multimodal AI model. This application provides a React frontend and Flask backend for image-based conversations.

## 🏗️ Architecture

```
├── backend/           # Flask REST API server
│   ├── app.py         # Main server application
│   ├── requirements.txt
│   ├── uploads/       # Uploaded images
│   └── outputs/       # Generated output images
│
└── frontend/          # React + Vite application
    ├── src/
    │   ├── App.jsx
    │   ├── api.js     # API client
    │   └── components/
    │       ├── ChatContainer.jsx
    │       ├── Header.jsx
    │       ├── Message.jsx
    │       └── Sidebar.jsx
    ├── package.json
    └── vite.config.js
```

## ✨ Features

- **Multi-turn Conversations**: Maintains conversation context across queries
- **Image Upload**: Drag-and-drop or click to upload images
- **Multiple Task Types**:
  - General Q&A
  - Visual Grounding
  - Affordance Detection  
  - Trajectory Planning
  - Pointing
- **Thinking Mode**: View the model's reasoning process
- **Real-time Updates**: Live chat interface with typing indicators
- **Session Management**: Create, reset, and manage chat sessions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server (mock mode for testing)
python app.py --port 5001

# Or with real model (requires GPU)
python app.py --port 5001 --no-mock
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access the Application

Open your browser and navigate to: **http://localhost:3000**

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/session` | POST | Create new session |
| `/api/session/<id>` | DELETE | Delete session |
| `/api/session/<id>/reset` | POST | Reset session memory |
| `/api/upload` | POST | Upload image file |
| `/api/images/<filename>` | GET | Retrieve uploaded image |
| `/api/chat` | POST | Send chat message |
| `/api/history/<session_id>` | GET | Get conversation history |
| `/api/tasks` | GET | List available tasks |

### Chat Request Format

```json
{
  "session_id": "uuid",
  "message": "What objects do you see?",
  "image": "filename.jpg",
  "task": "general",
  "enable_thinking": true
}
```

### Chat Response Format

```json
{
  "answer": "I can see...",
  "thinking": "Analyzing the image...",
  "turn_number": 1,
  "context_used": false,
  "task": "general"
}
```

## 🧪 Testing

### Test Backend API

```bash
# Health check
curl http://localhost:5001/api/health

# Create session
curl -X POST http://localhost:5001/api/session

# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:5001/api/upload

# Send message
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "message": "Describe this image", "image": "filename.jpg"}'
```

## 🔧 Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_MODEL` | `true` | Use mock model for testing |
| `HF_TOKEN` | - | HuggingFace token for model access |

### Frontend Proxy

The Vite dev server proxies `/api` requests to the backend. Configure in `vite.config.js`:

```javascript
proxy: {
  '/api': {
    target: 'http://localhost:5001',
    changeOrigin: true,
  }
}
```

## 📦 Production Deployment

### Backend

```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### Frontend

```bash
# Build for production
npm run build

# Serve with nginx or any static server
# The built files will be in the 'dist' directory
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is part of RoboBrain 2.0. See the main repository for license information.
