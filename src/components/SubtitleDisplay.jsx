import React from 'react';

function SubtitleDisplay({ subtitles, isRecording }) {
  return (
    <div className="h-36 p-5 text-left">
      <p className="text-sm font-semibold mb-3 text-cyan-300 uppercase tracking-wide">
        Live Subtitles
      </p>
      <pre className="whitespace-pre-wrap text-sm text-gray-300 h-full overflow-y-auto font-mono">
        {subtitles || (isRecording ? "Waiting for captions..." : "Start recording to see captions.")}
      </pre>
      {/*
        Note: Custom scrollbar styles (like scrollbar-thin)
        are not standard Tailwind. You might need to add them
        to your tailwind.config.js or main CSS file.
      */}
    </div>
  );
}

export default SubtitleDisplay;