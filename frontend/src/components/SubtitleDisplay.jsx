// src/components/SubtitleDisplay.jsx

import React from 'react';

function SubtitleDisplay({ latestPrediction, isRecording, accuracy, showAccuracy = false }) {
  const getAccuracyColor = (acc) => {
    if (acc >= 90) return 'text-green-400';
    if (acc >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const displayText = latestPrediction ? latestPrediction.prediction : (isRecording ? "Listening..." : "Start video for captions.");

  return (
    <div className="flex flex-col p-3 sm:p-4 border-b border-gray-700/60">
      {/* Header and Accuracy Bar */}
      <div className="mb-3">
        <p className="text-sm sm:text-base font-semibold text-cyan-400 border-b border-gray-700 pb-2">
          Live Caption
        </p>
        {showAccuracy && isRecording && (
          <div className="mt-2.5 flex items-center gap-2">
            <span className="text-xs text-gray-400">Accuracy:</span>
            <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 transition-all duration-300"
                style={{ width: `${accuracy}%` }}
              />
            </div>
            <span className={`text-xs font-semibold ${getAccuracyColor(accuracy)}`}>
              {accuracy.toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      {/* Highlighted Box for the Latest Subtitle */}
      <div className="w-full p-3 sm:p-4 bg-gray-900/50 border border-cyan-500/30 rounded-lg min-h-[60px] flex items-center justify-center animate-fade-in">
        <p className="text-center font-mono font-bold text-lg text-cyan-300 break-words">
          {displayText}
        </p>
      </div>
    </div>
  );
}

export default SubtitleDisplay;