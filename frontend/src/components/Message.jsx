import { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, AlertCircle, Loader2, Image as ImageIcon, ZoomIn } from 'lucide-react';

function Message({ message }) {
  const [showThinking, setShowThinking] = useState(false);
  const [showFullImage, setShowFullImage] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 message-enter ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center ${isUser
          ? 'bg-blue-600 text-white'
          : message.isError
            ? 'bg-red-500 text-white'
            : 'bg-gradient-to-br from-purple-500 to-blue-600 text-white'
          }`}
      >
        {isUser ? (
          <User className="w-5 h-5" />
        ) : message.isError ? (
          <AlertCircle className="w-5 h-5" />
        ) : (
          <Bot className="w-5 h-5" />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-2xl ${isUser ? 'text-right' : ''}`}>
        {/* User Image Preview */}
        {isUser && message.image && (
          <div className={`mb-2 ${isUser ? 'flex justify-end' : ''}`}>
            <img
              src={message.image}
              alt="Uploaded"
              className="max-w-xs max-h-48 rounded-lg border border-gray-200 object-cover"
            />
          </div>
        )}

        {/* Message Bubble */}
        <div
          className={`inline-block px-4 py-3 rounded-2xl ${isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : message.isError
              ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-md'
              : 'bg-white text-gray-800 border border-gray-200 rounded-bl-md shadow-sm'
            }`}
        >
          {message.isLoading ? (
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="loading-dots">Thinking</span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap">
              {message.content || (message.thinking ? (
                <span className="text-gray-500 italic">
                  (See reasoning below)
                </span>
              ) : (
                <span className="text-gray-400 italic">
                  (No response content)
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Thinking Section (for assistant) */}
        {!isUser && message.thinking && !message.isLoading && (
          <div className="mt-2">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 transition-colors"
            >
              {showThinking ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  <span>Hide coordinates</span>
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  <span>Show coordinates</span>
                </>
              )}
            </button>

            {(showThinking || !message.content) && (
              <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 font-mono">
                <p className="text-xs font-medium text-blue-600 mb-1">📍 Raw Coordinates:</p>
                <div className="whitespace-pre-wrap text-xs">{message.thinking}</div>
              </div>
            )}
          </div>
        )}

        {/* Output Image (for visual tasks) */}
        {!isUser && message.outputImage && !message.isLoading && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <ImageIcon className="w-4 h-4 text-green-600" />
              <p className="text-xs font-medium text-green-700">Visual Output:</p>
            </div>
            <div className="relative group">
              <img
                src={message.outputImage}
                alt="Model output visualization"
                className="max-w-md rounded-lg border-2 border-green-200 shadow-md cursor-pointer hover:border-green-400 transition-colors"
                onClick={() => setShowFullImage(true)}
              />
              <button
                onClick={() => setShowFullImage(true)}
                className="absolute top-2 right-2 p-1.5 bg-black/50 text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </div>

            {/* Full-screen Image Modal */}
            {showFullImage && (
              <div
                className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
                onClick={() => setShowFullImage(false)}
              >
                <div className="relative max-w-4xl max-h-full">
                  <img
                    src={message.outputImage}
                    alt="Model output visualization (full size)"
                    className="max-w-full max-h-[90vh] rounded-lg"
                  />
                  <button
                    onClick={() => setShowFullImage(false)}
                    className="absolute top-2 right-2 p-2 bg-white/90 text-gray-800 rounded-full hover:bg-white transition-colors"
                  >
                    ✕
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Task Badge (for user messages) */}
        {isUser && message.task && (
          <div className="mt-1">
            <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
              {message.task}
            </span>
          </div>
        )}

        {/* Timestamp */}
        <div className={`mt-1 text-xs text-gray-400 ${isUser ? 'text-right' : ''}`}>
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}

export default Message;
