import React from 'react';

function DeveloperAnalytics({ 
  fps = 0,
  landmarkCount = 0,
  latency = 0,
  confidence = 0,
  isRecording = false,
  windowSize,
  bufferSize = 0,
  frameCount = 0,
  backendStatus = 'Disconnected',
  rawPrediction = '',
  rawConfidence = 0
}) {
  const getStatusColor = (status) => {
    if (status.includes('Connected')) return 'text-green-400';
    if (status.includes('Error')) return 'text-red-400';
    return 'text-yellow-400';
  };

  return (
    <div className="p-3 sm:p-4 bg-gray-900/50 border-b border-gray-700">
      <h3 className="text-sm sm:text-base font-semibold text-cyan-400 mb-3">
        Developer Analytics
      </h3>
      
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">FPS:</span>
          <span className="ml-1 text-gray-300 font-mono">{fps}</span>
        </div>
        
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">Keypoints:</span>
          <span className="ml-1 text-gray-300 font-mono">{landmarkCount}</span>
        </div>
        
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">Latency:</span>
          <span className="ml-1 text-gray-300 font-mono">{latency}ms</span>
        </div>
        
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">Model Conf:</span>
          <span className="ml-1 text-gray-300 font-mono">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">Buffer:</span>
          <span className="ml-1 text-gray-300 font-mono">{bufferSize}/60</span>
        </div>
        
        <div className="bg-gray-800/50 p-2 rounded">
          <span className="text-gray-500">Frames:</span>
          <span className="ml-1 text-gray-300 font-mono">{frameCount}</span>
        </div>
        
        {windowSize !== undefined && (
          <div className="bg-gray-800/50 p-2 rounded col-span-2">
            <span className="text-gray-500">Window Size:</span>
            <span className="ml-1 text-gray-300 font-mono">{windowSize}s</span>
          </div>
        )}
      </div>
      
      <div className="mt-3 space-y-1 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-gray-500">Backend:</span>
          <span className={`${getStatusColor(backendStatus)} font-medium`}>
            {backendStatus}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-gray-500">MediaPipe:</span>
          <span className={`${isRecording ? 'text-green-400' : 'text-gray-500'} font-medium`}>
            {isRecording ? 'Processing' : 'Idle'}
          </span>
        </div>
        
        {rawPrediction && (
          <div className="flex items-start gap-2 mt-2 pt-2 border-t border-gray-700/50">
            <span className="text-gray-500">Raw:</span>
            <div className="flex-1">
              <span className="text-gray-300 font-medium">{rawPrediction}</span>
              <span className="ml-2 text-gray-500">({rawConfidence.toFixed(1)}%)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default DeveloperAnalytics;