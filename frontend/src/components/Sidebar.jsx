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
  Trash2
} from 'lucide-react';

const taskIcons = {
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
}) {
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
      <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between bg-white dark:bg-gray-800">
        <h2 className="text-lg font-bold text-gray-800 dark:text-white">Control Panel</h2>
        <button
          onClick={onToggle}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
        >
          <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-all duration-200 shadow-md hover:shadow-lg font-medium"
        >
          <Plus className="w-5 h-5" />
          <span>New Chat</span>
        </button>

        {/* Image Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <ImageIcon className="w-4 h-4 inline mr-1.5" />
            Current Image
          </label>

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
          >
            {currentImage ? (
              <div className="relative">
                <img
                  src={currentImage.preview}
                  alt="Current"
                  className="w-full h-40 object-cover rounded-lg"
                />
                <button
                  onClick={() => onImageUpload(null)}
                  className="absolute top-2 right-2 p-1.5 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
                <p className="mt-2 text-xs text-gray-500 dark:text-gray-400 truncate">
                  {currentImage.name}
                </p>
              </div>
            ) : (
              <label className="cursor-pointer">
                <Upload className="w-8 h-8 mx-auto text-gray-400 dark:text-gray-500 mb-2" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Drop image here or click to upload
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
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <Target className="w-4 h-4 inline mr-1.5" />
            Task Type
          </label>

          <div className="space-y-2">
            {availableTasks.map((task) => {
              const Icon = taskIcons[task.id] || MessageSquare;
              return (
                <button
                  key={task.id}
                  onClick={() => onTaskChange(task.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-200 shadow-sm hover:shadow-md ${currentTask === task.id
                      ? 'bg-blue-600 text-white border-2 border-blue-600 shadow-lg'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800'
                    }`}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <div className="min-w-0">
                    <p className="font-medium text-sm">{task.name}</p>
                    <p className={`text-xs truncate ${currentTask === task.id ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                      }`}>
                      {task.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Model Info */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-800 rounded-xl shadow-sm">
          <p className="text-sm font-bold text-blue-900 dark:text-blue-100 mb-1.5 flex items-center gap-2">
            <Brain className="w-5 h-5" />
            RoboBrain2.0-3B Model
          </p>
          <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
            Specialized in visual understanding and robotic task planning with real-time inference.
          </p>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 text-center">
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Created by Yasiru Basnayake and Jayamadu Gammune
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;
