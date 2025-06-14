import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import SubtitleDisplay from './SubtitleDisplay';

const WEBSOCKET_URL = 'http://localhost:5000';

// A new sub-component for a cleaner, non-blocking loading UI
const ProcessingStatus = ({ status, progress }) => (
  <div className="flex flex-col items-center justify-center h-full p-4 text-center">
    {/* Simple Spinner */}
    <svg className="animate-spin h-10 w-10 text-cyan-400 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    <p className="text-lg font-semibold text-white mb-2">Processing Video...</p>
    <p className="text-sm text-gray-400 mb-4">{status}</p>
    {/* Progress Bar */}
    <div className="w-full bg-gray-700 rounded-full h-2.5">
      <div className="bg-cyan-500 h-2.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
    </div>
  </div>
);

function UserMode2Processor({ videoFile }) {
  const [statusMessage, setStatusMessage] = useState('Awaiting video...');
  const [progress, setProgress] =useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [videoLoadError, setVideoLoadError] = useState(false);

  useEffect(() => {
    if (!videoFile) return;

    // --- 1. SETUP FOR NEW FILE ---
    setIsProcessing(true);
    setVideoLoadError(false);
    setPrediction(null);
    setProbabilities(null);
    setProgress(0);
    setStatusMessage('Preparing upload...');
    
    const localUrl = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(localUrl);

    // --- 2. CONNECT AND UPLOAD ---
    const socket = io(WEBSOCKET_URL);

    socket.on('connect', () => {
      setStatusMessage('Uploading to server...');
      socket.emit('predict_video', videoFile);
    });

    socket.on('status_update', (data) => {
      setStatusMessage(data.message);
      if (data.progress) {
        setProgress(data.progress);
      }
    });

    socket.on('prediction_result', (data) => {
      setPrediction(data.prediction);
      setProbabilities(data.probabilities);
      setStatusMessage('Prediction complete!');
      setIsProcessing(false); // Processing is done!
      setProgress(100);
    });

    // REMOVED 'video_ready' listener as it's no longer needed

    socket.on('prediction_error', (data) => {
      setStatusMessage(`Error: ${data.error}`);
      setIsProcessing(false); // Processing is done!
    });
    
    socket.on('disconnect', () => {
      if (isProcessing) {
        setStatusMessage('Lost connection to server.');
        setIsProcessing(false);
      }
    });

    // --- 3. CLEANUP ---
    return () => {
      URL.revokeObjectURL(localUrl);
      socket.disconnect();
    };
  }, [videoFile]);

  const topPrediction = prediction ? { prediction } : null;

  return (
    <>
      {/* --- LEFT PANEL: The Video Player --- */}
      <div className="flex-grow flex flex-col items-center justify-center p-4 bg-black/50 relative md:rounded-l-xl">
        <div className="w-full h-full flex items-center justify-center">
          {videoPreviewUrl && (
            <video 
              key={videoFile.name} // Use a stable key
              src={videoPreviewUrl} 
              className="block w-full max-w-full max-h-full object-contain rounded-xl" 
              controls 
              autoPlay
              muted
              onError={() => setVideoLoadError(true)} 
            />
          )}
          {videoLoadError && (
             <div className="text-red-400 text-center">
                <p>Could not load video preview.</p>
                <p className="text-sm text-gray-400">The file's codec may not be supported by this browser.</p>
             </div>
          )}
        </div>
      </div>
      
      {/* --- RIGHT PANEL: Status and Results --- */}
      <div className="w-full md:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col">
        {isProcessing ? (
          // Show the non-blocking loader while processing
          <ProcessingStatus status={statusMessage} progress={progress} />
        ) : (
          // Show the results after processing is complete
          <>
            <div className="p-4 border-b border-gray-700">
                <h4 className="font-semibold text-white">Result</h4>
                <p className="text-sm text-gray-400">{statusMessage}</p>
            </div>
            <div>
              <SubtitleDisplay
                latestPrediction={topPrediction}
                accuracy={probabilities && prediction ? parseFloat(probabilities[prediction]) * 100 : 0}
                showAccuracy={!!prediction}
                isRecording={false}
              />
            </div>
            {prediction && (
                <div className="flex-grow p-4 border-t border-gray-700 min-h-0">
                  <h4 className="font-semibold text-white mb-2">Confidence Scores</h4>
                  <div className="h-full overflow-y-auto text-sm pr-2">
                      {Object.entries(probabilities || {})
                          .sort(([, a], [, b]) => parseFloat(b) - parseFloat(a))
                          .map(([label, value]) => (
                              <div key={label} className="flex justify-between items-center mb-1">
                                  <span className="text-gray-300">{label}</span>
                                  <span className="text-gray-400 font-mono">{`${(parseFloat(value) * 100).toFixed(2)}%`}</span>
                              </div>
                          ))
                      }
                  </div>
                </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

export default UserMode2Processor;