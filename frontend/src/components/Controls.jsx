import React from 'react';

function Controls({ isRecording, onStart, onStop, mode, onModeChange, uploadedVideo,  onClearUpload }) {
  const modes = [
    { value: 'user-mode-1', label: 'User Mode 1' },
    { value: 'user-mode-2', label: 'User Mode 2' },
    { value: 'developer-mode-1', label: 'Developer Mode 1' },
    { value: 'developer-mode-2', label: 'Developer Mode 2' }
  ];

  return (
    <div className="flex justify-between items-center gap-3 sm:gap-4">
      {/* Mode Selector Dropdown */}
      <div className="flex items-center gap-2">
        <label htmlFor="mode-select" className="text-xs sm:text-sm text-gray-400 font-medium">
          Mode:
        </label>
        <select
          id="mode-select"
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
          className="px-3 py-1.5 sm:px-4 sm:py-2 bg-gray-700/80 border border-gray-600 rounded-md sm:rounded-lg text-xs sm:text-sm text-gray-200 focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 outline-none transition duration-200 hover:bg-gray-700"
        >
          {modes.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-3 sm:gap-4">
        {mode === 'user-mode-2' && uploadedVideo && (
          <button
            onClick={onClearUpload}
            className="flex items-center gap-1.5 sm:gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-gray-700/80 text-gray-300 font-medium rounded-md sm:rounded-lg border border-gray-600 hover:bg-gray-700 hover:text-white transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-60 text-xs sm:text-sm"
          >
            Clear Video
          </button>
        )}
        
        {mode !== 'user-mode-2' && (
          !isRecording ? (
            <button
              onClick={onStart}
              className="flex items-center gap-1.5 sm:gap-2 px-4 py-2 sm:px-5 sm:py-2.5 md:px-6 md:py-3 bg-gradient-to-br from-teal-500 to-teal-600 text-white font-semibold rounded-md sm:rounded-lg shadow-md hover:from-teal-600 hover:to-teal-700 transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-400 focus:ring-opacity-60 text-xs sm:text-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="w-4 h-4 sm:w-[18px] sm:h-[18px]" viewBox="0 0 16 16">
                <path fillRule="evenodd" d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2z"/>
              </svg>
              Start Video
            </button>
          ) : (
            <button
              onClick={onStop}
              className="flex items-center gap-1.5 sm:gap-2 px-4 py-2 sm:px-5 sm:py-2.5 md:px-6 md:py-3 bg-gradient-to-br from-red-500 to-red-600 text-white font-semibold rounded-md sm:rounded-lg shadow-md hover:from-red-600 hover:to-red-700 transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-opacity-60 text-xs sm:text-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="w-4 h-4 sm:w-[18px] sm:h-[18px]" viewBox="0 0 16 16">
                <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0M6.5 5A1.5 1.5 0 0 0 5 6.5v3A1.5 1.5 0 0 0 6.5 11h3A1.5 1.5 0 0 0 11 9.5v-3A1.5 1.5 0 0 0 9.5 5z"/>
              </svg>
              Stop Video
            </button>
          )
        )}
      </div>
    </div>
  );
}

export default Controls;