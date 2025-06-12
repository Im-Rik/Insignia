import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import SubtitleDisplay from './SubtitleDisplay';
import Loader from './Loader';

const SERVER_URL = 'http://localhost:5000';
const WEBSOCKET_URL = SERVER_URL;

function UserMode2Processor({ videoFile }) {
  // --- SIMPLIFIED STATE ---
  const [statusMessage, setStatusMessage] = useState('Awaiting video...');
  const [progress, setProgress] = useState(0); // For the progress bar
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [playableVideoUrl, setPlayableVideoUrl] = useState(null);
  const [videoLoadError, setVideoLoadError] = useState(false);

  useEffect(() => {
    if (!videoFile) return;

    // --- Reset states for a new file ---
    setIsProcessing(true);
    setStatusMessage('Connecting to server...');
    setProgress(0);
    setPlayableVideoUrl(null);
    setVideoLoadError(false);
    setPrediction(null);
    setProbabilities(null);
    
    const socket = io(WEBSOCKET_URL);

    socket.on('connect', () => {
      setStatusMessage('Connection established. Uploading video...');
      socket.emit('predict_video', videoFile);
    });

    // --- NEW: Listen for detailed status updates ---
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
      // Keep isProcessing true until the video preview is also ready
    });

    socket.on('video_ready', (data) => {
      setPlayableVideoUrl(SERVER_URL + data.url);
      setIsProcessing(false); // All tasks are done, stop processing indicator
      setProgress(100);
    });

    socket.on('prediction_error', (data) => {
      setStatusMessage(`Error: ${data.error}`);
      setIsProcessing(false);
    });
    
    socket.on('disconnect', () => {
      setStatusMessage('Disconnected.');
      setIsProcessing(false);
    });

    return () => {
      socket.disconnect();
    };
  }, [videoFile]);

  // --- Rendering logic uses the new simplified state ---
  const renderVideoPlayer = () => {
    // Show spinner if we don't have a URL yet AND we are processing
    if (!playableVideoUrl && isProcessing) {
      return  <Loader message={statusMessage} progress={progress} />;
    }
    // Show the video once the URL is available
    if (playableVideoUrl) {
      return (
        <video 
          key={playableVideoUrl}
          src={playableVideoUrl} 
          className="block w-full h-full object-contain rounded-xl" 
          controls 
          autoPlay
          onError={() => setVideoLoadError(true)} 
        />
      );
    }
    // Handle error case
    if (videoLoadError) {
        return <div className="text-red-400">Could not load video preview.</div>;
    }
    return null;
  };

  const topPrediction = prediction ? { prediction } : null;

  return (
    <>
      <div className="flex-grow flex flex-col items-center justify-center p-4 bg-black/50 relative md:rounded-l-xl">
        <div className="w-full h-[calc(100%-80px)] flex items-center justify-center">
            {renderVideoPlayer()}
        </div>
        <div className="mt-4 text-center">
          <p className="text-lg font-semibold text-white">{videoFile?.name}</p>
        </div>
      </div>
      
      <div className="w-full md:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h4 className="font-semibold text-white">Status</h4>
          {/* Display the real-time status message from the backend */}
          <p className="text-sm text-gray-400">{statusMessage}</p>
        </div>
        {/* FIX: Removed flex-grow from the container below */}
        <div>
          <SubtitleDisplay
            latestPrediction={topPrediction}
            accuracy={probabilities && prediction ? parseFloat(probabilities[prediction]) * 100 : 0}
            showAccuracy={!!prediction}
            isRecording={isProcessing}
          />
        </div>
        {prediction && (
            <div className="p-4 border-t border-gray-700">
              <h4 className="font-semibold text-white mb-2">Confidence Scores</h4>
              <div className="max-h-60 overflow-y-auto text-sm">
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
      </div>
    </>
  );
}

export default UserMode2Processor;