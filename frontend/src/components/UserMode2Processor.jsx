import React, { useState, useEffect, useRef } from 'react'; // ✅ 1. Import useRef
import { io } from 'socket.io-client';
import SubtitleDisplay from './SubtitleDisplay';

const WEBSOCKET_URL = 'http://localhost:5000';

// A new sub-component for a cleaner, non-blocking loading UI
const ProcessingStatus = ({ status, progress }) => (
  <div className="flex flex-col items-center justify-center h-full p-4 text-center">
    {/* ... (no changes in this component) ... */}
    <svg className="animate-spin h-10 w-10 text-cyan-400 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    <p className="text-lg font-semibold text-white mb-2">Processing Video...</p>
    <p className="text-sm text-gray-400 mb-4">{status}</p>
    <div className="w-full bg-gray-700 rounded-full h-2.5">
      <div className="bg-cyan-500 h-2.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
    </div>
  </div>
);

function UserMode2Processor({ videoFile }) {
  const [statusMessage, setStatusMessage] = useState('Awaiting video...');
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [videoLoadError, setVideoLoadError] = useState(false);

  // ✅ 2. Create a ref to track the processing state without causing re-renders
  const isProcessingRef = useRef(isProcessing);
  // ✅ 3. Keep the ref's value in sync with the state on every render
  isProcessingRef.current = isProcessing;

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
      setIsProcessing(false);
      setProgress(100);
    });

    socket.on('prediction_error', (data) => {
      setStatusMessage(`Error: ${data.error}`);
      setIsProcessing(false);
    });
    
    socket.on('disconnect', () => {
      // ✅ 4. Use the ref's current value here. This reads the latest value without being a dependency.
      if (isProcessingRef.current) {
        setStatusMessage('Lost connection to server.');
        setIsProcessing(false);
      }
    });

    // --- 3. CLEANUP ---
    return () => {
      URL.revokeObjectURL(localUrl);
      socket.disconnect();
    };
     // ✅ 5. The dependency array ONLY contains videoFile. The effect will only re-run for a new file.
  }, [videoFile]);

  const topPrediction = prediction ? { prediction } : null;

  return (
    // ... (no changes in the JSX return statement) ...
    <div className="flex flex-col md:flex-row h-full w-full overflow-hidden rounded-xl border border-gray-700/50 bg-gray-800/70">
      <div className="flex-grow flex flex-col items-center justify-center p-4 bg-black/50 relative md:rounded-l-xl">
        <div className="w-full h-full flex items-center justify-center">
          {videoPreviewUrl && !videoLoadError && (
            <video 
              key={videoFile.name}
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
              <p className="text-lg font-semibold mb-2">Could not load video preview</p>
              <p className="text-sm text-gray-400">The file's codec may not be supported by this browser.</p>
            </div>
          )}
          {!videoPreviewUrl && !videoLoadError && (
            <div className="text-gray-400 text-center">
              <p className="text-lg">No video loaded</p>
            </div>
          )}
        </div>
      </div>
      <div className="w-full md:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col md:rounded-r-xl">
        {isProcessing ? (
          <ProcessingStatus status={statusMessage} progress={progress} />
        ) : (
          <div className="flex flex-col h-full">
            <div className="p-4 border-b border-gray-700">
              <h4 className="font-semibold text-white">Result</h4>
              <p className="text-sm text-gray-400">{statusMessage}</p>
            </div>
            <div className="flex-shrink-0">
              <SubtitleDisplay
                latestPrediction={topPrediction}
                accuracy={probabilities && prediction ? parseFloat(probabilities[prediction]) * 100 : 0}
                showAccuracy={!!prediction}
                isRecording={false}
              />
            </div>
            {prediction && probabilities && (
              <div className="flex-grow p-4 border-t border-gray-700 min-h-0 overflow-hidden">
                <h4 className="font-semibold text-white mb-3">Confidence Scores</h4>
                <div className="h-full overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
                  {Object.entries(probabilities)
                    .sort(([, a], [, b]) => parseFloat(b) - parseFloat(a))
                    .map(([label, value]) => {
                      const percentage = (parseFloat(value) * 100).toFixed(2);
                      const isTopPrediction = label === prediction;
                      
                      return (
                        <div 
                          key={label} 
                          className={`flex justify-between items-center mb-2 p-2 rounded transition-colors ${
                            isTopPrediction ? 'bg-cyan-900/30 border border-cyan-700/50' : 'hover:bg-gray-700/30'
                          }`}
                        >
                          <span className={isTopPrediction ? 'text-cyan-300 font-medium' : 'text-gray-300'}>
                            {label}
                          </span>
                          <span className={`font-mono ${isTopPrediction ? 'text-cyan-400 font-bold' : 'text-gray-400'}`}>
                            {percentage}%
                          </span>
                        </div>
                      );
                    })
                  }
                </div>
              </div>
            )}
            {!prediction && !isProcessing && (
              <div className="flex-grow flex items-center justify-center p-4 text-gray-400">
                <p className="text-center">Upload and process a video to see results here.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default UserMode2Processor;