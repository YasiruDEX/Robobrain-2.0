import { useState, useEffect } from 'react';
import ChatContainer from './components/ChatContainer';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { createSession, checkHealth, getTasks } from './api';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentTask, setCurrentTask] = useState('general');
  const [enableThinking] = useState(false); // 3B model doesn't support thinking mode
  const [availableTasks, setAvailableTasks] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);

  // Initialize session and check health
  useEffect(() => {
    const init = async () => {
      try {
        // Check backend health
        const health = await checkHealth();
        setIsConnected(health.status === 'healthy');
        
        // Get available tasks
        const tasksData = await getTasks();
        setAvailableTasks(tasksData.tasks || []);
        
        // Create session
        const session = await createSession();
        setSessionId(session.session_id);
      } catch (error) {
        console.error('Failed to initialize:', error);
        setIsConnected(false);
      }
    };
    
    init();
  }, []);

  const handleNewChat = async () => {
    try {
      const session = await createSession();
      setSessionId(session.session_id);
      setMessages([]);
      setCurrentImage(null);
    } catch (error) {
      console.error('Failed to create new session:', error);
    }
  };

  const handleImageUpload = (imageData) => {
    setCurrentImage(imageData);
  };

  const addMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  const updateLastMessage = (updates) => {
    setMessages((prev) => {
      const newMessages = [...prev];
      if (newMessages.length > 0) {
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          ...updates,
        };
      }
      return newMessages;
    });
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        currentTask={currentTask}
        onTaskChange={setCurrentTask}
        availableTasks={availableTasks}
        onNewChat={handleNewChat}
        currentImage={currentImage}
        onImageUpload={handleImageUpload}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header
          isConnected={isConnected}
          sessionId={sessionId}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
        />
        
        <ChatContainer
          sessionId={sessionId}
          currentImage={currentImage}
          currentTask={currentTask}
          enableThinking={enableThinking}
          messages={messages}
          addMessage={addMessage}
          updateLastMessage={updateLastMessage}
          onImageUpload={handleImageUpload}
        />
      </div>
    </div>
  );
}

export default App;
