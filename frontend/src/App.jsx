import { useState, useEffect } from 'react';
import ChatContainer from './components/ChatContainer';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { createSession, checkHealth, getTasks } from './api';
import { saveChatHistory } from './utils/chatHistory';

// Check if screen is mobile size
const isMobileScreen = () => window.innerWidth < 768;

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentTask, setCurrentTask] = useState('auto');
  const [enableThinking] = useState(false); // 3B model doesn't support thinking mode
  const [availableTasks, setAvailableTasks] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(!isMobileScreen()); // Closed by default on mobile
  const [isMobile, setIsMobile] = useState(isMobileScreen());
  const [messages, setMessages] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [complexMode, setComplexMode] = useState(false);

  // Handle window resize for responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const mobile = isMobileScreen();
      setIsMobile(mobile);
      // Auto-close sidebar when switching to mobile
      if (mobile && sidebarOpen) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarOpen]);

  // Handle dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

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

        // Always create a new session on app start
        const session = await createSession();
        setSessionId(session.session_id);
        console.log('Created new session:', session.session_id);
      } catch (error) {
        console.error('Failed to initialize:', error);
        setIsConnected(false);
      }
    };

    init();
  }, []);

  // Auto-save chat history whenever messages change
  useEffect(() => {
    if (sessionId && messages.length > 0) {
      saveChatHistory(sessionId, messages, {
        task: currentTask,
        hasImage: !!currentImage,
      });
    }
  }, [messages, sessionId, currentTask, currentImage]);

  const handleNewChat = async () => {
    try {
      // Save current session before creating new one
      if (sessionId && messages.length > 0) {
        saveChatHistory(sessionId, messages, {
          task: currentTask,
          hasImage: !!currentImage,
        });
      }

      // Create new session
      const session = await createSession();
      setSessionId(session.session_id);
      setMessages([]);
      setCurrentImage(null);
      setCurrentTask('auto');
      console.log(`Created new session: ${session.session_id}`);
    } catch (error) {
      console.error('Failed to create new session:', error);
    }
  };

  const handleImageUpload = (imageData) => {
    setCurrentImage(imageData);
  };

  const handleLoadSession = async (loadedSessionId, loadedMessages, metadata = {}) => {
    if (!loadedSessionId || !loadedMessages) {
      // Create new session if loading failed
      await handleNewChat();
      return;
    }

    // Load the selected session
    setSessionId(loadedSessionId);
    setMessages(loadedMessages);

    // Restore metadata
    if (metadata.task) {
      setCurrentTask(metadata.task);
    }

    // Extract image from last message if exists
    const lastMessageWithImage = [...loadedMessages].reverse().find(m => m.image);
    if (lastMessageWithImage) {
      setCurrentImage(lastMessageWithImage.image);
    } else {
      setCurrentImage(null);
    }

    setShowHistory(false);
    console.log(`Loaded session: ${loadedSessionId} with ${loadedMessages.length} messages`);
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
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200 overflow-hidden">
      {/* Mobile Overlay Backdrop */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

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
        onLoadSession={handleLoadSession}
        currentSessionId={sessionId}
        complexMode={complexMode}
        onComplexModeChange={setComplexMode}
        isMobile={isMobile}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header
          isConnected={isConnected}
          sessionId={sessionId}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
          darkMode={darkMode}
          onToggleDarkMode={() => setDarkMode(!darkMode)}
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
          complexMode={complexMode}
          isMobile={isMobile}
        />
      </div>
    </div>
  );
}

export default App;
