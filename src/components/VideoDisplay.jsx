import React from 'react';

function VideoDisplay({ videoRef, isRecording }) {
  return (
    <div className="relative w-full h-full bg-black rounded-md sm:rounded-lg md:rounded-xl overflow-hidden shadow-inner"> 
      <video
        ref={videoRef}
        className="block w-full h-full object-contain"
        playsInline
        muted
      />
      {!isRecording && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-75 text-gray-400 text-xs sm:text-sm space-y-2 sm:space-y-3 p-3 sm:p-4 text-center">
          {/* Responsive SVG size */}
          <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="currentColor" className="sm:w-48 sm:h-48 md:w-56 md:h-56 bi bi-camera-video-off-fill text-gray-500 mb-1 sm:mb-2" viewBox="0 0 16 16">
            <path fillRule="evenodd" d="M10.961 12.365a1.99 1.99 0 0 0 .522-1.103l3.11 1.382A1 1 0 0 0 16 11.731V4.269a1 1 0 0 0-1.406-.913l-3.111 1.382A2 2 0 0 0 9.5 3H4.272l6.69 9.365zm-10.114-9 A1.99 1.99 0 0 0 0 5v6a2 2 0 0 0 2 2h5.728L.847 3.366zm9.746-1.539 1.187 1.187L1 13.793l-1.187-1.187z"/>
          </svg>
          <span className="font-medium text-sm sm:text-base md:text-lg text-gray-300">Camera is Off</span>
          <span className="text-gray-500">Click "Start Video" to show your feed</span>
        </div>
      )}
    </div>
  );
}

export default VideoDisplay;