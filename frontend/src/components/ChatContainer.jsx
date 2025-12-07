import { useState, useRef, useEffect } from 'react';
import { Send, Image as ImageIcon, Loader2, Bot } from 'lucide-react';
import { sendMessage, uploadImage } from '../api';
import { annotateImage, parseCoordinates, resizeImageIfNeeded } from '../utils/imageAnnotation';
import { generateAndSaveConversationName } from '../utils/chatHistory';
import Message from './Message';

function ChatContainer({
  sessionId,
  currentImage,
  currentTask,
  enableThinking,
  messages,
  addMessage,
  updateLastMessage,
  onImageUpload,
}) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!input.trim() || !sessionId || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      image: currentImage?.preview,
      task: currentTask,
      timestamp: new Date().toISOString(),
    };

    addMessage(userMessage);
    setInput('');
    setIsLoading(true);

    // Add placeholder for assistant response
    const assistantPlaceholder = {
      id: Date.now() + 1,
      role: 'assistant',
      content: '',
      thinking: '',
      isLoading: true,
      timestamp: new Date().toISOString(),
    };
    addMessage(assistantPlaceholder);

    try {
      // Upload image if new
      let imageRef = null;
      if (currentImage?.file) {
        const uploadResult = await uploadImage(currentImage.file);
        imageRef = uploadResult.filename;
      }

      // Send message
      const response = await sendMessage(sessionId, userMessage.content, {
        image: imageRef,
        task: currentTask,
        enableThinking,
      });

      // If no output image but task requires visualization, annotate on frontend
      let finalOutputImage = response.output_image;
      
      if (!finalOutputImage && response.task !== 'general' && currentImage?.preview) {
        console.log('No backend image, attempting frontend annotation...');
        const textToUse = response.answer || response.thinking || '';
        const annotations = parseCoordinates(textToUse, response.task);
        
        if (annotations.points || annotations.boxes || annotations.trajectories) {
          try {
            finalOutputImage = await annotateImage(currentImage.preview, annotations);
            console.log('✓ Frontend annotation successful');
          } catch (err) {
            console.error('Frontend annotation failed:', err);
            finalOutputImage = currentImage.preview; // Fallback to original
          }
        } else {
          console.log('No coordinates found to annotate');
          finalOutputImage = currentImage.preview;
        }
      }

      // Update assistant message with response
      updateLastMessage({
        content: response.answer,
        thinking: response.thinking,
        outputImage: finalOutputImage,
        task: response.task,
        taskSource: response.task_source,
        isLoading: false,
      });

      // Generate conversation name for first user message
      const userMessages = messages.filter(m => m.role === 'user');
      if (userMessages.length === 0) {
        // This is the first user message, generate a title
        generateAndSaveConversationName(sessionId, userMessage.content);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      updateLastMessage({
        content: `Error: ${error.response?.data?.error || error.message || 'Failed to get response'}`,
        isLoading: false,
        isError: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const originalImage = reader.result;
        
        try {
          // Resize image if needed (max dimension 2000px)
          const resizedResult = await resizeImageIfNeeded(originalImage, 2000);
          
          if (resizedResult.scaleFactor < 1) {
            console.log(`Image resized: ${resizedResult.originalWidth}x${resizedResult.originalHeight} → ${resizedResult.newWidth}x${resizedResult.newHeight}`);
          }
          
          // Convert resized data URL back to File object for upload
          const blob = await fetch(resizedResult.dataUrl).then(r => r.blob());
          const resizedFile = new File([blob], file.name, { type: 'image/jpeg' });
          
          onImageUpload({
            file: resizedFile,
            preview: resizedResult.dataUrl,
            name: file.name,
            scaleFactor: resizedResult.scaleFactor,
            originalDimensions: {
              width: resizedResult.originalWidth,
              height: resizedResult.originalHeight
            },
            newDimensions: {
              width: resizedResult.newWidth,
              height: resizedResult.newHeight
            }
          });
        } catch (error) {
          console.error('Failed to resize image:', error);
          // Fallback to original if resize fails
          onImageUpload({
            file,
            preview: originalImage,
            name: file.name,
          });
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/30 rounded-2xl flex items-center justify-center mb-6 border border-blue-200 dark:border-blue-800">
              <Bot className="w-10 h-10 text-blue-600 dark:text-blue-400" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
              Welcome to RoboBrain 2.0
            </h2>
            <p className="text-gray-600 dark:text-gray-400 max-w-md mb-8 leading-relaxed">
              Upload an image and start asking questions. I can help with general Q&A, visual grounding, affordance detection, trajectory planning, and pointing tasks.
            </p>

            {!currentImage && (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors shadow-sm hover:shadow-md"
              >
                <ImageIcon className="w-5 h-5" />
                <span>Upload an Image to Start</span>
              </button>
            )}
          </div>
        ) : (
          messages.map((message) => (
            <Message key={message.id} message={message} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Current Image Preview (if set) */}
      {currentImage && messages.length === 0 && (
        <div className="px-4 pb-2">
          <div className="bg-white rounded-lg p-3 border border-gray-200 flex items-center gap-3">
            <img
              src={currentImage.preview}
              alt="Current"
              className="w-16 h-16 object-cover rounded-lg"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-700 truncate">
                {currentImage.name}
              </p>
              <p className="text-xs text-gray-500">Ready to analyze</p>
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 transition-colors duration-200">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3">
            {/* Image Upload Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex-shrink-0 p-3 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              title="Upload image"
            >
              <ImageIcon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder={currentImage ? "Ask about the image..." : "Upload an image first..."}
                disabled={!currentImage || isLoading}
                rows={1}
                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                style={{ maxHeight: '150px' }}
              />
            </div>

            {/* Send Button */}
            <button
              type="submit"
              disabled={!input.trim() || !currentImage || isLoading}
              className="flex-shrink-0 p-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Task Indicator */}
          <div className="mt-2 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <span>Current task:</span>
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 rounded-full font-medium">
              {currentTask}
            </span>
            {enableThinking && (
              <span className="px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300 rounded-full font-medium">
                thinking enabled
              </span>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

export default ChatContainer;
