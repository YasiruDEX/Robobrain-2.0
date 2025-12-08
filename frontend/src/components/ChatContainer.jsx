import { useState, useRef, useEffect } from 'react';
import { Send, Image as ImageIcon, Loader2, Bot, GitBranch, CheckCircle2, Circle, XCircle } from 'lucide-react';
import { sendMessage, uploadImage } from '../api';
import { annotateImage, parseCoordinates, resizeImageIfNeeded } from '../utils/imageAnnotation';
import { generateAndSaveConversationName } from '../utils/chatHistory';
import { decomposeQuery, executePipeline } from '../utils/complexPipeline';
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
  complexMode,
}) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, pipelineProgress]);

  // Handle complex pipeline execution
  const handleComplexSubmit = async (userMessage, imageRef) => {
    try {
      // Step 1: Decompose the query into pipeline steps
      updateLastMessage({
        content: 'Analyzing query and creating execution pipeline...',
        isPipeline: true,
        pipelineSteps: [],
        isLoading: true,
      });

      const decompositionResult = await decomposeQuery(userMessage.content);
      const { pipeline, fallback, error } = decompositionResult;

      if (!pipeline || pipeline.length === 0) {
        throw new Error('Failed to decompose query');
      }

      // Show warning if using fallback
      if (fallback) {
        console.warn('Using fallback pipeline due to:', error);
        updateLastMessage({
          content: `⚠️ Pipeline decomposition failed: ${error || 'Unknown error'}\n\nFalling back to single general task...`,
          isPipeline: true,
          pipelineSteps: [],
          isLoading: true,
        });
      }

      console.log('Pipeline created:', pipeline, fallback ? '(fallback)' : '');

      // Initialize pipeline progress
      const initialSteps = pipeline.map((step, idx) => ({
        ...step,
        status: idx === 0 ? 'running' : 'pending',
        result: null,
      }));

      setPipelineProgress(initialSteps);
      updateLastMessage({
        content: fallback
          ? `⚠️ Using fallback (${error || 'decomposition failed'})\nExecuting ${pipeline.length}-step pipeline...`
          : `Executing ${pipeline.length}-step pipeline...`,
        isPipeline: true,
        pipelineSteps: initialSteps,
        pipelineFallback: fallback,
        pipelineError: error,
        isLoading: true,
      });

      // Step 2: Execute pipeline steps sequentially
      const results = await executePipeline(
        sessionId,
        pipeline,
        imageRef,
        currentImage?.preview,
        // onStepStart callback
        (stepIdx, step) => {
          setPipelineProgress(prev => {
            const updated = [...prev];
            updated[stepIdx] = { ...updated[stepIdx], status: 'running' };
            return updated;
          });
          updateLastMessage({
            content: `Executing step ${stepIdx + 1}/${pipeline.length}: ${step.description}`,
            isPipeline: true,
            pipelineSteps: pipelineProgress,
            currentStep: stepIdx,
            isLoading: true,
          });
        },
        // onStepComplete callback
        (stepIdx, result) => {
          setPipelineProgress(prev => {
            const updated = [...prev];
            updated[stepIdx] = {
              ...updated[stepIdx],
              status: result.success ? 'completed' : 'failed',
              result,
            };
            // Mark next step as running if exists
            if (stepIdx + 1 < updated.length) {
              updated[stepIdx + 1] = { ...updated[stepIdx + 1], status: 'running' };
            }
            return updated;
          });
        }
      );

      // Step 3: Build final response with combined annotations
      const completedSteps = results.results.map((r, idx) => ({
        ...pipeline[idx],
        status: r.success ? 'completed' : 'failed',
        result: r,
      }));

      updateLastMessage({
        content: results.summary,
        thinking: results.results.map(r => r.thinking).filter(Boolean).join('\n\n'),
        outputImage: results.finalOutputImage,
        isPipeline: true,
        pipelineSteps: completedSteps,
        task: 'pipeline',
        taskSource: 'complex',
        isLoading: false,
      });

      setPipelineProgress(null);

    } catch (error) {
      console.error('Pipeline execution failed:', error);
      updateLastMessage({
        content: `Pipeline Error: ${error.message}`,
        isPipeline: true,
        isLoading: false,
        isError: true,
      });
      setPipelineProgress(null);
    }
  };

  // Handle simple (non-pipeline) submission
  const handleSimpleSubmit = async (userMessage, imageRef) => {
    try {
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
            finalOutputImage = currentImage.preview;
          }
        } else {
          console.log('No coordinates found to annotate');
          finalOutputImage = currentImage.preview;
        }
      }

      updateLastMessage({
        content: response.answer,
        thinking: response.thinking,
        outputImage: finalOutputImage,
        task: response.task,
        taskSource: response.task_source,
        isLoading: false,
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      updateLastMessage({
        content: `Error: ${error.response?.data?.error || error.message || 'Failed to get response'}`,
        isLoading: false,
        isError: true,
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!input.trim() || !sessionId || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      image: currentImage?.preview,
      task: complexMode ? 'pipeline' : currentTask,
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
      isPipeline: complexMode,
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

      // Execute based on mode
      if (complexMode) {
        await handleComplexSubmit(userMessage, imageRef);
      } else {
        await handleSimpleSubmit(userMessage, imageRef);
      }

      // Generate conversation name for first user message
      const userMessages = messages.filter(m => m.role === 'user');
      if (userMessages.length === 0) {
        generateAndSaveConversationName(sessionId, userMessage.content);
      }
    } catch (error) {
      console.error('Submit error:', error);
      updateLastMessage({
        content: `Error: ${error.message || 'Failed to get response'}`,
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

  // Render pipeline progress indicator
  const renderPipelineProgress = () => {
    if (!pipelineProgress || pipelineProgress.length === 0) return null;

    return (
      <div className="px-4 pb-2">
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200 dark:border-purple-800">
          <div className="flex items-center gap-2 mb-2">
            <GitBranch className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
              Pipeline Execution
            </span>
          </div>
          <div className="space-y-1.5">
            {pipelineProgress.map((step, idx) => (
              <div key={idx} className="flex items-center gap-2 text-xs">
                {step.status === 'running' ? (
                  <Loader2 className="w-3.5 h-3.5 text-purple-500 animate-spin" />
                ) : step.status === 'completed' ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                ) : step.status === 'failed' ? (
                  <XCircle className="w-3.5 h-3.5 text-red-500" />
                ) : (
                  <Circle className="w-3.5 h-3.5 text-gray-400" />
                )}
                <span className={`${step.status === 'running' ? 'text-purple-700 dark:text-purple-300 font-medium' :
                  step.status === 'completed' ? 'text-green-700 dark:text-green-400' :
                    step.status === 'failed' ? 'text-red-600 dark:text-red-400' :
                      'text-gray-500 dark:text-gray-400'
                  }`}>
                  Step {idx + 1}: {step.description || step.task}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
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

            {complexMode && (
              <div className="mb-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800 max-w-md">
                <div className="flex items-center gap-2 text-purple-700 dark:text-purple-300 mb-2">
                  <GitBranch className="w-5 h-5" />
                  <span className="font-semibold">Pipeline Mode Active</span>
                </div>
                <p className="text-sm text-purple-600 dark:text-purple-400">
                  Complex queries will be automatically decomposed into sequential sub-tasks for better results.
                </p>
              </div>
            )}

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

      {/* Pipeline Progress Indicator */}
      {pipelineProgress && renderPipelineProgress()}

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
                placeholder={currentImage ? (complexMode ? "Describe what you want to do (e.g., 'pick up the red cup')..." : "Ask about the image...") : "Upload an image first..."}
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
              className={`flex-shrink-0 p-3 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors ${complexMode
                ? 'bg-purple-600 hover:bg-purple-700'
                : 'bg-blue-600 hover:bg-blue-700'
                }`}
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
            <span>Mode:</span>
            {complexMode ? (
              <span className="px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300 rounded-full font-medium flex items-center gap-1">
                <GitBranch className="w-3 h-3" />
                Pipeline
              </span>
            ) : (
              <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 rounded-full font-medium">
                {currentTask}
              </span>
            )}
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
