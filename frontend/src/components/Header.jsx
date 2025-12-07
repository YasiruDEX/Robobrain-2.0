import { Menu, Wifi, WifiOff } from 'lucide-react';

function Header({ isConnected, sessionId, onToggleSidebar, sidebarOpen }) {
  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
        )}
        
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
            <span className="text-white text-lg">🤖</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-800">RoboBrain 2.0</h1>
            <p className="text-xs text-gray-500">Interactive Chat</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Connection Status */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
          isConnected 
            ? 'bg-green-50 text-green-700' 
            : 'bg-red-50 text-red-700'
        }`}>
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4" />
              <span>Disconnected</span>
            </>
          )}
        </div>

        {/* Session ID */}
        {sessionId && (
          <div className="hidden md:block text-xs text-gray-400 font-mono">
            Session: {sessionId.slice(0, 8)}...
          </div>
        )}
      </div>
    </header>
  );
}

export default Header;
