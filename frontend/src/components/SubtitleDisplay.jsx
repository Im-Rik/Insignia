import React from 'react';

function SubtitleDisplay({ subtitles, isRecording, accuracy, showAccuracy = false }) {
  const getAccuracyColor = (acc) => {
    if (acc >= 90) return 'text-green-400';
    if (acc >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex flex-col h-full p-3 sm:p-4">
      <div className="mb-2 sm:mb-3">
        <p className="text-sm sm:text-base md:text-md font-semibold text-cyan-400 border-b border-gray-700 pb-2 sm:pb-2.5">
          Live Captions
        </p>
        
        {showAccuracy && isRecording && (
          <div className="mt-2 flex items-center gap-2">
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

      <pre 
        className="flex-grow whitespace-pre-wrap text-xs sm:text-sm text-gray-300 overflow-y-auto font-mono 
                   scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500 
                   scrollbar-track-gray-700/50 scrollbar-thumb-rounded-full scrollbar-track-rounded-full pr-1.5 sm:pr-2 leading-relaxed sm:leading-normal"
      >
        {subtitles || (isRecording ? "Listening..." : "Start video for captions.")}
      </pre>
      
      <div className="mt-auto pt-2 sm:pt-3 border-t border-gray-700">
        <input 
          type="text" 
          placeholder="" 
          disabled 
          className="w-full p-2 sm:p-2.5 bg-gray-700/70 border border-gray-600 rounded-md sm:rounded-lg text-xs sm:text-sm placeholder-gray-500 focus:ring-1 sm:focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 outline-none" 
        />
      </div>
    </div>
  );
}

export default SubtitleDisplay;