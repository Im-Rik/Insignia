import React from 'react';

function Controls({ isRecording, onStart, onStop }) {
  return (
    <div className="flex gap-4">
      {!isRecording ? (
        <button
          onClick={onStart}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-br from-teal-500 to-teal-600 text-white font-semibold rounded-lg shadow-md hover:from-teal-600 hover:to-teal-700 transition duration-200 ease-in-out transform hover:-translate-y-1 active:translate-y-0 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-opacity-50"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-camera-video-fill" viewBox="0 0 16 16">
             <path fillRule="evenodd" d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2z"/>
           </svg>
          Start Recording
        </button>
      ) : (
        <button
          onClick={onStop}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-br from-red-500 to-red-600 text-white font-semibold rounded-lg shadow-md hover:from-red-600 hover:to-red-700 transition duration-200 ease-in-out transform hover:-translate-y-1 active:translate-y-0 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-stop-circle-fill" viewBox="0 0 16 16">
             <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0M6.5 5A1.5 1.5 0 0 0 5 6.5v3A1.5 1.5 0 0 0 6.5 11h3A1.5 1.5 0 0 0 11 9.5v-3A1.5 1.5 0 0 0 9.5 5z"/>
           </svg>
          Stop Recording
        </button>
      )}
    </div>
  );
}

export default Controls;