// src/components/DeveloperAnalytics.jsx
import React from 'react';

function DeveloperAnalytics({ isConnected, isRecording, accuracy, bufferSize }) {
  const statusColor = isConnected ? 'text-green-400' : 'text-yellow-400';
  const statusText = isConnected ? 'Connected' : 'Disconnected';
  const recordingColor = isRecording ? 'text-green-400' : 'text-gray-500';

  return (
    <div className="p-3 sm:p-4 border-b border-gray-700/60">
      <p className="text-sm sm:text-base md:text-md font-semibold text-cyan-400 mb-3">
        Developer Analytics
      </p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:text-sm text-gray-300">
        
        <div className="font-mono">Recording:</div>
        <div className={`font-semibold ${recordingColor}`}>{isRecording ? 'Active' : 'Stopped'}</div>

        <div className="font-mono">Backend:</div>
        <div className={`font-semibold ${statusColor}`}>{statusText}</div>

        <div className="font-mono">Model Conf:</div>
        <div>{accuracy.toFixed(1)}%</div>
        
        <div className="font-mono">Buffer:</div>
        <div>{bufferSize} / 60</div>
        
      </div>
    </div>
  );
}

export default DeveloperAnalytics;