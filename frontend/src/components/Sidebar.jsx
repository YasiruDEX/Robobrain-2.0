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
    <aside className="w-72 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h2 className="font-semibold text-gray-700">Settings</h2>
        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          <span>New Chat</span>
        </button>

        {/* Image Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <ImageIcon className="w-4 h-4 inline mr-1.5" />
            Current Image
          </label>

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-blue-400 transition-colors"
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
                <p className="mt-2 text-xs text-gray-500 truncate">
                  {currentImage.name}
                </p>
              </div>
            ) : (
              <label className="cursor-pointer">
                <Upload className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
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
          <label className="block text-sm font-medium text-gray-700 mb-2">
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
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${currentTask === task.id
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'hover:bg-gray-50 text-gray-700 border border-transparent'
                    }`}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <div className="min-w-0">
                    <p className="font-medium text-sm">{task.name}</p>
                    <p className="text-xs text-gray-500 truncate">
                      {task.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Model Info */}
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-xs font-medium text-blue-900 mb-1">💡 Using RoboBrain2.0-3B</p>
          <p className="text-xs text-blue-700">
            This model specializes in visual understanding and robotic task planning.
          </p>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 text-center">
        <p className="text-xs text-gray-400">
          RoboBrain 2.0 © 2024
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;
