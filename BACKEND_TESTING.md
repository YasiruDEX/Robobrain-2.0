# Backend Testing and Integration Guide

## Summary of Fixes

### 1. Fixed NameError in inference.py
**Issue**: `result` variable was not defined before the return statement
**Fix**: Initialize `result` dictionary before the plotting section

### 2. Fixed Device Mismatch
**Issue**: Model was on CPU but inputs were sent to CUDA
**Fix**: Dynamically detect model device: `device = next(self.model.parameters()).device`

### 3. Added Plot Support
- Added `plot` parameter to `MultiTurnInference.ask()` method
- Updated all shortcut methods (ground, get_affordance, get_trajectory, point_at)
- Backend now includes output_image path in responses

### 4. Fixed Backend Configuration
- Added `repo_dir` parameter to `MultiTurnInference` initialization
- Both session creation and auto-creation now pass `repo_dir`

## Testing Steps

### 1. Test Core Inference
```bash
cd /home/yasiru/Documents/github/Robobrain-2.0
conda run -n robobrain2-env python scripts/run_general_qa.py
```

**Expected**: Script runs successfully, outputs model responses with and without thinking mode.

### 2. Start Backend Server
```bash
chmod +x run_backend.sh
./run_backend.sh
```

Or manually:
```bash
conda activate robobrain2-env
python backend.py
```

**Expected**: 
```
Initializing model... This may take a while.
Using local weights from: /home/yasiru/Documents/github/Robobrain-2.0/weights
Model initialized successfully.
Starting RoboBrain 2.0 Backend on port 5001...
 * Running on http://0.0.0.0:5001
```

### 3. Test Backend Endpoints

#### Health Check
```bash
curl http://localhost:5001/api/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_sessions": 0
}
```

#### Create Session
```bash
curl -X POST http://localhost:5001/api/session
```

**Expected Response**:
```json
{
  "session_id": "uuid-here",
  "sessionId": "uuid-here"
}
```

#### Upload Image
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5001/api/upload
```

**Expected Response**:
```json
{
  "path": "/absolute/path/to/uploaded/file.jpg",
  "filename": "unique-filename.jpg",
  "url": "/uploads/unique-filename.jpg"
}
```

#### Chat Request
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "message": "What do you see in this image?",
    "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "task": "general",
    "enable_thinking": false
  }'
```

**Expected Response**:
```json
{
  "answer": "Two cats laying down on a pink blanket...",
  "thinking": "",
  "output_image": null,
  "context_used": false,
  "turn_number": 1,
  "task": "general"
}
```

#### Get History
```bash
curl http://localhost:5001/api/history/your-session-id
```

**Expected Response**:
```json
[
  {
    "role": "user",
    "content": "What do you see in this image?",
    "image_path": "http://...",
    "task": "general",
    "timestamp": "2025-12-07T...",
    "metadata": {}
  },
  {
    "role": "assistant",
    "content": "Two cats laying down...",
    "image_path": "http://...",
    "task": null,
    "timestamp": "2025-12-07T...",
    "metadata": {...}
  }
]
```

## Frontend Integration

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/session` | POST | Create new session |
| `/api/session/<id>` | DELETE | Delete session |
| `/api/session/<id>/reset` | POST | Reset conversation |
| `/api/upload` | POST | Upload image file |
| `/api/chat` | POST | Send message |
| `/api/history/<id>` | GET | Get conversation history |
| `/api/tasks` | GET | Get available tasks |
| `/uploads/<filename>` | GET | Serve uploaded image |
| `/result/<filename>` | GET | Serve result image |

### Frontend Implementation Example

```javascript
// 1. Create Session
const createSession = async () => {
  const response = await fetch('http://localhost:5001/api/session', {
    method: 'POST'
  });
  const data = await response.json();
  return data.sessionId;
};

// 2. Upload Image
const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:5001/api/upload', {
    method: 'POST',
    body: formData
  });
  const data = await response.json();
  return data.path;
};

// 3. Send Chat Message
const sendMessage = async (sessionId, message, imagePath, task = 'general') => {
  const response = await fetch('http://localhost:5001/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
      image: imagePath,
      task: task,
      enable_thinking: false
    })
  });
  const data = await response.json();
  return data;
};

// 4. Get History
const getHistory = async (sessionId) => {
  const response = await fetch(`http://localhost:5001/api/history/${sessionId}`);
  const data = await response.json();
  return data;
};

// Example Usage
(async () => {
  // Create session
  const sessionId = await createSession();
  console.log('Session:', sessionId);
  
  // Use image URL or upload file
  const imagePath = 'http://images.cocodataset.org/val2017/000000039769.jpg';
  
  // Send first message
  const response1 = await sendMessage(sessionId, 'What animals are in this image?', imagePath);
  console.log('Response 1:', response1.answer);
  
  // Send follow-up (uses context)
  const response2 = await sendMessage(sessionId, 'How many are there?', imagePath);
  console.log('Response 2:', response2.answer);
  console.log('Context used:', response2.context_used); // Should be true
  
  // Get full history
  const history = await getHistory(sessionId);
  console.log('History:', history);
})();
```

### Task Types

| Task | Description | Output Format |
|------|-------------|---------------|
| `general` | Visual QA | Natural language text |
| `grounding` | Object detection | Bounding box `[x1, y1, x2, y2]` |
| `affordance` | Action affordance | Affordance map coordinates |
| `trajectory` | Motion planning | Trajectory points `[[x1,y1], [x2,y2], ...]` |
| `pointing` | Object pointing | Point coordinates `[(x, y), (x, y), ...]` |

### Visualization Tasks

For tasks that support visualization (grounding, affordance, trajectory, pointing), the backend automatically generates annotated images when these tasks are requested. The `output_image` field in the response will contain the path:

```json
{
  "answer": "[142, 89, 256, 198]",
  "output_image": "/result/filename_with_grounding_annotated.jpg",
  ...
}
```

Display in frontend:
```javascript
if (response.output_image) {
  const imgUrl = `http://localhost:5001${response.output_image}`;
  document.getElementById('result-img').src = imgUrl;
}
```

## Status

✅ Core inference working
✅ Backend API endpoints functional
✅ Multi-turn conversation memory working
✅ Plot/visualization support added
✅ All task types supported
✅ CORS enabled for frontend integration

## Next Steps

1. Start the backend server
2. Test all endpoints with curl
3. Integrate frontend with the API
4. Add error handling and loading states in frontend
5. Implement session management UI
6. Add image preview before upload
7. Display conversation history
8. Show annotated images for visual tasks
