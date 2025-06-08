import React from 'react';

function WindowSizeSelector({ windowSize, onWindowSizeChange }) {
  const windowSizes = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5];

  return (
    <div className="mb-4 p-3 bg-gray-900/50 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-cyan-400">
          Data Window Size
        </h4>
        <span className="text-sm font-mono text-gray-300">
          {windowSize}s chunks
        </span>
      </div>
      
      <div className="space-y-2">
        <input
          type="range"
          min="0.5"
          max="5"
          step="0.5"
          value={windowSize}
          onChange={(e) => onWindowSizeChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #14b8a6 0%, #14b8a6 ${((windowSize - 0.5) / 4.5) * 100}%, #374151 ${((windowSize - 0.5) / 4.5) * 100}%, #374151 100%)`
          }}
        />
        
        <div className="flex justify-between text-xs text-gray-500">
          {windowSizes.map((size) => (
            <span key={size} className="font-mono">
              {size}s
            </span>
          ))}
        </div>
      </div>
      
      <div className="mt-3 text-xs text-gray-400">
        <p>MediaPipe data will be sent to backend in {windowSize} second chunks.</p>
        <p className="mt-1">Smaller windows = more real-time, larger windows = better context.</p>
      </div>
    </div>
  );
}

export default WindowSizeSelector;