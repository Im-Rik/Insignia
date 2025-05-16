import React from 'react';

function SubtitleDisplay({ subtitles, isRecording }) {
  return (
    // The parent in App.js already provides overall rounding for this section.
    // We add padding here for the internal content of the subtitle box.
    <div className="flex flex-col h-full p-4 sm:p-5"> 
      <p className="text-md sm:text-lg font-semibold mb-3 sm:mb-4 text-cyan-400 border-b border-gray-700 pb-2.5 sm:pb-3"> {/* Larger text, more padding */}
        Live Captions
      </p>
      <pre 
        className="flex-grow whitespace-pre-wrap text-sm sm:text-base text-gray-300 overflow-y-auto font-mono 
                   scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500 
                   scrollbar-track-gray-700/50 scrollbar-thumb-rounded-full scrollbar-track-rounded-full pr-2 sm:pr-3 leading-relaxed" // leading-relaxed for better readability
      >
        {subtitles || (isRecording ? "Listening for audio..." : "Start video to view live captions.")}
      </pre>
      {/* Example of a "Send a message" input, styled similarly */}
      <div className="mt-auto pt-3 sm:pt-4 border-t border-gray-700">
        <input 
          type="text" 
          placeholder="Send a message (UI only)" 
          disabled 
          className="w-full p-2.5 sm:p-3 bg-gray-700/70 border border-gray-600 rounded-lg text-sm placeholder-gray-500 focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 outline-none" 
        />
      </div>
    </div>
  );
}

export default SubtitleDisplay;