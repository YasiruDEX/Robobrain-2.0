import {
  X,
  Plus,
  Image as ImageIcon,
  Brain,
  Target,
  Hand,
  Route,
  MousePointer,
  MessageSquare,
  Upload,
  Trash2,
  Sparkles,
  Clock,
  Zap
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { getAllSessions, deleteSessionHistory } from '../utils/chatHistory';

const taskIcons = {
  auto: Sparkles,
  general: MessageSquare,
  grounding: Target,
  affordance: Hand,
  trajectory: Route,
  pointing: MousePointer,
};

function Sidebar({
  isOpen,
  onToggle,
  currentTask,
  onTaskChange,
  availableTasks,
  onNewChat,
  currentImage,
  onImageUpload,
  onLoadSession,
  currentSessionId,
}) {
  const [sessions, setSessions] = useState([]);
  const [showTasks, setShowTasks] = useState(false);

  useEffect(() => {
    loadSessions();
  }, [currentSessionId]);

  const loadSessions = () => {
    const allSessions = getAllSessions();
    setSessions(allSessions.slice(0, 10)); // Show last 10 conversations
  };

  const handleLoadSession = (sessionId, messages, metadata) => {
    onLoadSession(sessionId, messages, metadata);
  };

  const handleDeleteSession = (sessionId, e) => {
    e.stopPropagation();
    if (window.confirm('Delete this conversation?')) {
      deleteSessionHistory(sessionId);
      loadSessions();
      if (sessionId === currentSessionId) {
        onNewChat();
      }
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffHours < 24) return `${diffHours}h`;
    if (diffDays < 7) return `${diffDays}d`;
    return date.toLocaleDateString();
  };

  const getSessionTitle = (session) => {
    // Use conversation name if available, otherwise use preview
    if (session.metadata?.conversationName) {
      return session.metadata.conversationName;
    }

    const firstUserMsg = session.messages?.find(m => m.role === 'user');
    if (firstUserMsg?.content) {
      const text = firstUserMsg.content;
      return text.length > 30 ? text.substring(0, 30) + '...' : text;
    }
    return 'New Conversation';
  };
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onImageUpload({
          file,
          preview: reader.result,
          name: file.name,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onImageUpload({
          file,
          preview: reader.result,
          name: file.name,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  if (!isOpen) return null;

  return (
    <aside className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full shadow-xl transition-colors duration-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-bold text-gray-800 dark:text-white">RoboBrain 2.0</h2>
          <button
            onClick={onToggle}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
          >
            <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg transition-all duration-200 shadow-md hover:shadow-lg font-medium"
        >
          <Plus className="w-5 h-5" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Chat History Section */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Recent Chats
            </h3>
            {sessions.length > 0 && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {sessions.length}
              </span>
            )}
          </div>

          {sessions.length === 0 ? (
            <div className="text-center py-8 text-gray-400 dark:text-gray-500">
              <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs mt-1">Start a new chat to begin</p>
            </div>
          ) : (
            <div className="space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.sessionId}
                  onClick={() => handleLoadSession(session.sessionId, session.messages, session.metadata)}
                  className={`group relative p-3 rounded-lg cursor-pointer transition-all duration-200 border ${session.sessionId === currentSessionId
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-500 dark:border-blue-600 shadow-sm'
                      : 'bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-sm'
                    }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium truncate ${session.sessionId === currentSessionId
                          ? 'text-blue-900 dark:text-blue-100'
                          : 'text-gray-900 dark:text-gray-100'
                        }`}>
                        {getSessionTitle(session)}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`text-xs ${session.sessionId === currentSessionId
                            ? 'text-blue-600 dark:text-blue-400'
                            : 'text-gray-500 dark:text-gray-400'
                          }`}>
                          {formatTimestamp(session.metadata?.updatedAt)}
                        </span>
                        {session.metadata?.hasImage && (
                          <ImageIcon className="w-3 h-3 text-gray-400" />
                        )}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteSession(session.sessionId, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-red-500 transition-opacity"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Collapsible Task & Image Section */}
      <div className="border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setShowTasks(!showTasks)}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        >
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <Target className="w-4 h-4" />
            Task & Image Settings
          </span>
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${showTasks ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showTasks && (
          <div className="p-4 space-y-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
            {/* Image Upload */}
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
                Image Upload
              </label>
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-3 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
              >
                {currentImage ? (
                  <div className="relative">
                    <img
                      src={currentImage.preview}
                      alt="Current"
                      className="w-full h-32 object-cover rounded-lg"
                    />
                    <button
                      onClick={() => onImageUpload(null)}
                      className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ) : (
                  <label className="cursor-pointer">
                    <Upload className="w-6 h-6 mx-auto text-gray-400 dark:text-gray-500 mb-1" />
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Drop or click to upload
                    </p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                )}
              </div>
            </div>

            {/* Task Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
                Task Mode
              </label>
              <div className="space-y-1.5">
                {availableTasks.map((task) => {
                  const Icon = taskIcons[task.id] || MessageSquare;
                  return (
                    <button
                      key={task.id}
                      onClick={() => onTaskChange(task.id)}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-all duration-200 ${currentTask === task.id
                          ? 'bg-blue-600 text-white shadow-md'
                          : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800'
                        }`}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="text-xs font-medium">{task.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-700 text-center bg-gray-50 dark:bg-gray-800/50">
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Yasiru Basnayake & Jayamadu Gammune
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;
