import React from 'react';

function SubtitleDisplay({ subtitles, isRecording }) {
  return (
    // h-full allows it to take the height defined by its parent in App.js
    <div className="flex flex-col h-full p-3 sm:p-4"> 
      <p className="text-sm sm:text-base md:text-md font-semibold mb-2 sm:mb-3 text-cyan-400 border-b border-gray-700 pb-2 sm:pb-2.5">
        Live Captions
      </p>
      <pre 
        className="flex-grow whitespace-pre-wrap text-xs sm:text-sm text-gray-300 overflow-y-auto font-mono 
                   scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500 
                   scrollbar-track-gray-700/50 scrollbar-thumb-rounded-full scrollbar-track-rounded-full pr-1.5 sm:pr-2 leading-relaxed sm:leading-normal"
      >
        {subtitles || (isRecording ? "Listening..." : "Start video for captions.")} {/* Shorter text for mobile */}
      </pre>
      <div className="mt-auto pt-2 sm:pt-3 border-t border-gray-700">
        <input 
          type="text" 
          placeholder="Send message (UI only)" 
          disabled 
          className="w-full p-2 sm:p-2.5 bg-gray-700/70 border border-gray-600 rounded-md sm:rounded-lg text-xs sm:text-sm placeholder-gray-500 focus:ring-1 sm:focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 outline-none" 
        />
      </div>
    </div>
  );
}

export default SubtitleDisplay;