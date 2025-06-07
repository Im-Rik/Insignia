import React, { useState } from 'react';
import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';

import useCamera from './hooks/useCamera';
import useSubtitles from './hooks/useSubtitles';

function App() {
  const { videoRef, startVideo, stopVideo, isRecording } = useCamera();
  const mockSubtitles = [
    "Hello there!",
    "Welcome to this live caption demo.",
    "This interface is now responsive.",
  ];
  const subtitles = useSubtitles(isRecording, mockSubtitles);

  return (
    <div className="flex flex-col items-center h-screen bg-gray-900 text-gray-100 font-sans">
      <div className="flex flex-col w-full max-w-screen-2xl h-full p-2 sm:p-3 md:p-4 lg:p-6">

        {/* Video + Subtitles Section */}
        <div className="flex flex-col md:flex-row flex-1 overflow-hidden rounded-lg md:rounded-xl shadow-xl bg-gray-800/70 backdrop-blur-md border border-gray-700/50">
          
          <div className="flex-grow flex items-center justify-center p-4 bg-black/50 relative md:rounded-l-xl">
            <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
          </div>

          <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
            <SubtitleDisplay subtitles={subtitles} isRecording={isRecording} />
          </div>
        </div>

        {/* Controls */}
        <div className="mt-4 p-3 bg-gray-800/70 backdrop-blur-md rounded-lg shadow-lg border border-gray-700/50">
          <Controls isRecording={isRecording} onStart={startVideo} onStop={stopVideo} />
        </div>
      </div>
    </div>
  );
}

export default App;
