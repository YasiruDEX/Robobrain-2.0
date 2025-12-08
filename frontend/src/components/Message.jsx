import { useState, useEffect } from 'react';
import { User, Bot, ChevronDown, ChevronUp, AlertCircle, Loader2, Image as ImageIcon, ZoomIn, MapPin, Sparkles, Zap, GitBranch, CheckCircle2, XCircle, Circle } from 'lucide-react';

// Typing animation component
const TypingText = ({ text, speed = 20 }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    }
  }, [currentIndex, text, speed]);

  return <span>{displayedText}</span>;
};

// Pipeline Steps Display Component
const PipelineSteps = ({ steps, isLoading }) => {
  const [expanded, setExpanded] = useState(true);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-3 mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 transition-colors mb-2"
      >
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        <GitBranch className="w-3.5 h-3.5" />
        <span className="font-medium">Pipeline Steps ({steps.length})</span>
      </button>

      {expanded && (
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200 dark:border-purple-800">
          <div className="space-y-2">
            {steps.map((step, idx) => (
              <div key={idx} className="flex items-start gap-2 text-xs">
                <div className="flex-shrink-0 mt-0.5">
                  {step.status === 'running' ? (
                    <Loader2 className="w-4 h-4 text-purple-500 animate-spin" />
                  ) : step.status === 'completed' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : step.status === 'failed' ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : (
                    <Circle className="w-4 h-4 text-gray-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`font-medium ${step.status === 'running' ? 'text-purple-700 dark:text-purple-300' :
                        step.status === 'completed' ? 'text-green-700 dark:text-green-400' :
                          step.status === 'failed' ? 'text-red-600 dark:text-red-400' :
                            'text-gray-600 dark:text-gray-400'
                      }`}>
                      Step {idx + 1}: {step.task}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${step.task === 'grounding' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' :
                        step.task === 'affordance' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300' :
                          step.task === 'trajectory' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' :
                            step.task === 'pointing' ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300' :
                              'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                      }`}>
                      {step.task}
                    </span>
                  </div>
                  <p className="text-gray-600 dark:text-gray-400 mt-0.5">
                    {step.description || step.prompt}
                  </p>
                  {step.result && step.result.answer && (
                    <div className="mt-1 p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-gray-700 dark:text-gray-300 text-xs">
                        {step.result.answer.length > 150
                          ? step.result.answer.substring(0, 150) + '...'
                          : step.result.answer}
                      </p>
                    </div>
                  )}
                  {step.result && step.result.error && (
                    <p className="text-red-500 text-xs mt-1">
                      Error: {step.result.error}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

function Message({ message }) {
  const [showThinking, setShowThinking] = useState(false);
  const [showFullImage, setShowFullImage] = useState(false);
  const [showTyping, setShowTyping] = useState(!message.isLoading && message.role === 'assistant' && message.content);
  const [imageRevealed, setImageRevealed] = useState(false);
  const isUser = message.role === 'user';
  const isPipeline = message.isPipeline;

  // Trigger image reveal animation when image is loaded
  useEffect(() => {
    if (message.outputImage && !message.isLoading) {
      const timer = setTimeout(() => setImageRevealed(true), 100);
      return () => clearTimeout(timer);
    }
  }, [message.outputImage, message.isLoading]);

  return (
    <div className={`flex gap-3 message-enter ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-sm ${isUser
          ? 'bg-blue-600 text-white'
          : message.isError
            ? 'bg-red-600 text-white'
            : isPipeline
              ? 'bg-purple-100 text-purple-600 dark:bg-purple-900/50 dark:text-purple-400 border border-purple-200 dark:border-purple-800'
              : 'bg-blue-100 text-blue-600 dark:bg-blue-900/50 dark:text-blue-400 border border-blue-200 dark:border-blue-800'
          }`}
      >
        {isUser ? (
          <User className="w-5 h-5" />
        ) : message.isError ? (
          <AlertCircle className="w-5 h-5" />
        ) : isPipeline ? (
          <GitBranch className="w-5 h-5" />
        ) : (
          <Bot className="w-5 h-5" />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-2xl ${isUser ? 'text-right' : ''}`}>
        {/* Task Badge (for assistant) */}
        {!isUser && message.task && (
          <div className={`flex items-center gap-1.5 mb-1.5 text-xs ml-1 ${isPipeline
              ? 'text-purple-600 dark:text-purple-400'
              : message.taskSource === 'auto'
                ? 'text-green-600 dark:text-green-400'
                : 'text-blue-500 dark:text-blue-400'
            }`}>
            {isPipeline ? (
              <GitBranch className="w-3.5 h-3.5" />
            ) : message.taskSource === 'auto' ? (
              <Zap className="w-3.5 h-3.5" />
            ) : (
              <Sparkles className="w-3 h-3" />
            )}
            <span>
              {isPipeline ? 'Complex Pipeline' : message.taskSource === 'auto' ? 'Auto-detected: ' : 'Task: '}
              <span className={`font-medium capitalize ${isPipeline
                  ? 'text-purple-700 dark:text-purple-300'
                  : message.taskSource === 'auto'
                    ? 'text-green-700 dark:text-green-300'
                    : 'text-blue-700 dark:text-blue-300'
                }`}>
                {isPipeline ? `(${message.pipelineSteps?.length || 0} steps)` : message.task}
              </span>
            </span>
          </div>
        )}

        {/* User Image Preview */}
        {isUser && message.image && (
          <div className={`mb-2 ${isUser ? 'flex justify-end' : ''}`}>
            <img
              src={message.image}
              alt="Uploaded"
              className="max-w-xs max-h-48 rounded-lg border border-gray-200 dark:border-gray-700 object-cover"
            />
          </div>
        )}

        {/* Message Bubble */}
        <div
          className={`inline-block px-5 py-3.5 rounded-2xl shadow-sm ${isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : message.isError
              ? 'bg-red-50 text-red-800 border-2 border-red-200 rounded-bl-md dark:bg-red-900/20 dark:text-red-300 dark:border-red-900'
              : isPipeline
                ? 'bg-purple-50 text-gray-800 border border-purple-200 rounded-bl-md dark:bg-purple-900/20 dark:text-gray-200 dark:border-purple-800'
                : 'bg-white text-gray-800 border border-gray-200 rounded-bl-md dark:bg-gray-800 dark:text-gray-200 dark:border-gray-700'
            }`}
        >
          {message.isLoading ? (
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="loading-dots">
                {isPipeline ? 'Running pipeline' : 'Generating response'}
              </span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap">
              {message.content ? (
                showTyping && !isUser && !isPipeline ? (
                  <TypingText text={message.content} speed={15} />
                ) : (
                  message.content
                )
              ) : (
                message.thinking ? (
                  <span className="text-gray-500 dark:text-gray-400 italic">
                    (See coordinates below)
                  </span>
                ) : (
                  <span className="text-gray-400 dark:text-gray-500 italic">
                    (No response content)
                  </span>
                )
              )}
            </div>
          )}
        </div>

        {/* Pipeline Steps (for pipeline messages) */}
        {!isUser && isPipeline && message.pipelineSteps && (
          <PipelineSteps steps={message.pipelineSteps} isLoading={message.isLoading} />
        )}

        {/* Thinking Section (for assistant) */}
        {!isUser && message.thinking && !message.isLoading && (
          <div className="mt-2">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
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
              <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 font-mono dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-200">
                <p className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1 flex items-center gap-1">
                  <MapPin className="w-3 h-3" />
                  Raw Coordinates:
                </p>
                <div className="whitespace-pre-wrap text-xs">{message.thinking}</div>
              </div>
            )}
          </div>
        )}

        {/* Output Image (for visual tasks) */}
        {!isUser && message.outputImage && !message.isLoading && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <ImageIcon className={`w-4 h-4 ${isPipeline ? 'text-purple-600' : message.taskSource === 'auto' ? 'text-green-600' : 'text-blue-600'
                }`} />
              <p className={`text-xs font-medium ${isPipeline
                  ? 'text-purple-700 dark:text-purple-400'
                  : message.taskSource === 'auto'
                    ? 'text-green-700 dark:text-green-400'
                    : 'text-blue-700 dark:text-blue-400'
                }`}>
                {isPipeline ? 'Combined Visual Output:' : 'Visual Output:'}
              </p>
            </div>
            <div className="relative group">
              <img
                src={message.outputImage.startsWith('http') ? message.outputImage : (message.outputImage.startsWith('data:') ? message.outputImage : `${import.meta.env.VITE_API_URL || ''}${message.outputImage}`)}
                alt="Model output visualization"
                className={`max-w-md rounded-lg border-2 shadow-md cursor-pointer transition-all ${imageRevealed ? 'image-reveal' : 'opacity-0'
                  } ${isPipeline
                    ? 'border-purple-200 hover:border-purple-400 dark:border-purple-700 dark:hover:border-purple-500'
                    : message.taskSource === 'auto'
                      ? 'border-green-200 hover:border-green-400 dark:border-green-700 dark:hover:border-green-500'
                      : 'border-blue-200 hover:border-blue-400 dark:border-blue-700 dark:hover:border-blue-500'
                  }`}
                onClick={() => setShowFullImage(true)}
                onError={(e) => { e.target.onerror = null; e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" fill="%23999"%3EImage unavailable%3C/text%3E%3C/svg%3E'; }}
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
                    src={message.outputImage.startsWith('http') ? message.outputImage : (message.outputImage.startsWith('data:') ? message.outputImage : `${import.meta.env.VITE_API_URL || ''}${message.outputImage}`)}
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
            <span className={`text-xs px-2 py-0.5 rounded-full ${message.task === 'pipeline'
                ? 'bg-purple-100 text-purple-700'
                : 'bg-blue-100 text-blue-700'
              }`}>
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
