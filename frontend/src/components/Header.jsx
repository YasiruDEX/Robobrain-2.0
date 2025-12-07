import { Menu, Wifi, WifiOff } from 'lucide-react';

function Header({ isConnected, sessionId, onToggleSidebar, sidebarOpen }) {
  return (
    <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-purple-700 shadow-lg px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-lg hover:bg-white/10 transition-all duration-200"
          >
            <Menu className="w-5 h-5 text-white" />
          </button>
        )}

        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg border border-white/30">
            <span className="text-2xl">🤖</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">RoboBrain 2.0</h1>
            <p className="text-xs text-blue-100">Vision-Language AI Assistant</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Connection Status */}
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium shadow-md ${isConnected
            ? 'bg-green-500 text-white'
            : 'bg-red-500 text-white'
          }`}>
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4" />
              <span>Online</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4" />
              <span>Offline</span>
            </>
          )}
        </div>

        {/* Session ID */}
        {sessionId && (
          <div className="hidden md:block text-xs text-blue-100 font-mono bg-white/10 px-3 py-1.5 rounded-lg">
            ID: {sessionId.slice(0, 8)}
          </div>
        )}
      </div>
    </header>
  );
}

export default Header;
