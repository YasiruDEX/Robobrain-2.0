import { generateConversationTitle } from '../api';

// Chat history storage keys
const CHAT_HISTORY_KEY = 'robobrain_chat_history';
const CURRENT_SESSION_KEY = 'robobrain_current_session';

/**
 * Save chat history to localStorage
 */
export const saveChatHistory = (sessionId, messages, metadata = {}) => {
  try {
    const history = getChatHistory();
    const timestamp = new Date().toISOString();
    
    history[sessionId] = {
      sessionId,
      messages,
      metadata: {
        ...metadata,
        updatedAt: timestamp,
        createdAt: history[sessionId]?.metadata?.createdAt || timestamp,
      },
    };
    
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
    localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
    
    console.log(`💾 Saved chat history: ${sessionId} (${messages.length} messages)`);
  } catch (error) {
    console.error('Failed to save chat history:', error);
  }
};

/**
 * Get all chat history
 */
export const getChatHistory = () => {
  try {
    const stored = localStorage.getItem(CHAT_HISTORY_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.error('Failed to load chat history:', error);
    return {};
  }
};

/**
 * Get chat history for specific session
 */
export const getSessionHistory = (sessionId) => {
  const history = getChatHistory();
  const session = history[sessionId] || null;
  
  if (session) {
    console.log(`📖 Retrieved session: ${sessionId} (${session.messages?.length || 0} messages)`);
  }
  
  return session;
};

/**
 * Delete chat history for specific session
 */
export const deleteSessionHistory = (sessionId) => {
  try {
    const history = getChatHistory();
    delete history[sessionId];
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
    
    // Clear current session if it's the one being deleted
    const currentSessionId = getCurrentSessionId();
    if (currentSessionId === sessionId) {
      localStorage.removeItem(CURRENT_SESSION_KEY);
    }
    
    console.log(`🗑️ Deleted session: ${sessionId}`);
  } catch (error) {
    console.error('Failed to delete chat history:', error);
  }
};

/**
 * Get current session ID
 */
export const getCurrentSessionId = () => {
  return localStorage.getItem(CURRENT_SESSION_KEY);
};

/**
 * Clear all chat history
 */
export const clearAllHistory = () => {
  try {
    localStorage.removeItem(CHAT_HISTORY_KEY);
    localStorage.removeItem(CURRENT_SESSION_KEY);
  } catch (error) {
    console.error('Failed to clear chat history:', error);
  }
};

/**
 * Get list of all sessions sorted by most recent
 */
export const getAllSessions = () => {
  const history = getChatHistory();
  return Object.values(history)
    .sort((a, b) => new Date(b.metadata.updatedAt) - new Date(a.metadata.updatedAt));
};

/**
 * Generate and save conversation name for a session
 */
export const generateAndSaveConversationName = async (sessionId, firstMessage) => {
  try {
    // Get the session
    const session = getSessionHistory(sessionId);
    if (!session || session.metadata?.conversationName) {
      return; // Already has a name or session doesn't exist
    }
    
    // Generate title using Groq
    const response = await generateConversationTitle(firstMessage);
    const title = response.title;
    
    if (title) {
      // Update session metadata with the generated name
      const history = getChatHistory();
      if (history[sessionId]) {
        history[sessionId].metadata = {
          ...history[sessionId].metadata,
          conversationName: title,
        };
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
        console.log(`💬 Generated conversation name: "${title}"`);
      }
    }
  } catch (error) {
    console.error('Failed to generate conversation name:', error);
  }
};
