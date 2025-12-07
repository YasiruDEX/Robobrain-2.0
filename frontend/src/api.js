import axios from 'axios';

// Use environment variable for backend URL, fallback to relative path for local dev
const BACKEND_URL = import.meta.env.VITE_API_URL || '';
const API_BASE = BACKEND_URL ? `${BACKEND_URL}/api` : '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minutes for long inference
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Session management
export const createSession = async () => {
  const response = await api.post('/session');
  return response.data;
};

export const deleteSession = async (sessionId) => {
  const response = await api.delete(`/session/${sessionId}`);
  return response.data;
};

export const resetSession = async (sessionId) => {
  const response = await api.post(`/session/${sessionId}/reset`);
  return response.data;
};

// Image upload
export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Chat
export const sendMessage = async (sessionId, message, options = {}) => {
  const response = await api.post('/chat', {
    session_id: sessionId,
    message,
    image: options.image,
    task: options.task || 'general',
    enable_thinking: options.enableThinking !== false,
  });
  return response.data;
};

// History
export const getHistory = async (sessionId) => {
  const response = await api.get(`/history/${sessionId}`);
  return response.data;
};

// Available tasks
export const getTasks = async () => {
  const response = await api.get('/tasks');
  return response.data;
};

export default api;
