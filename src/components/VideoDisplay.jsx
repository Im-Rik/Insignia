import React from 'react';

function VideoDisplay({ videoRef, isRecording }) {
  return (
    <div className="relative w-full aspect-video bg-black border-b border-gray-700">
      {/* Video element */}
      <video
        ref={videoRef}
        className="block w-full h-full object-cover rounded-t-lg"
        playsInline // Important for iOS Safari
        muted // Ensure it's muted as we don't request audio
      />
      {/* Placeholder text when not recording */}
      {!isRecording && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-60 text-gray-400 text-base space-y-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="currentColor" className="bi bi-camera-video-off-fill text-gray-500" viewBox="0 0 16 16">
            <path fillRule="evenodd" d="M10.961 12.365a1.99 1.99 0 0 0 .522-1.103l3.11 1.382A1 1 0 0 0 16 11.731V4.269a1 1 0 0 0-1.406-.913l-3.111 1.382A2 2 0 0 0 9.5 3H4.272l6.69 9.365zm-10.114-9 A1.99 1.99 0 0 0 0 5v6a2 2 0 0 0 2 2h5.728L.847 3.366zm9.746-1.539 1.187 1.187L1 13.793l-1.187-1.187z"/>
          </svg>
          <span>Camera feed will appear here</span>
        </div>
      )}
    </div>
  );
}

export default VideoDisplay;