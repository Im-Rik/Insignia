// App.jsx - Final Robust Version
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';
import VideoUploader from './components/VideoUploader';
import UserMode2Processor from './components/UserMode2Processor';
import DeveloperAnalytics from './components/DeveloperAnalytics';
import MediaPipeOverlay from './components/MediaPipeOverlay';
import WindowSizeSelector from './components/WindowSizeSelector';
import useVideoStream from './hooks/useVideoStream';
import useMediaPipe from './hooks/useMediaPipe';
import { socket } from './socket';
import { extractKeypoints } from './utils/keypointExtractor';
import { mockSubtitles } from './utils/subtitles';

function App() {
  const { videoRef, isRecording, startVideo, stopVideo } = useVideoStream();
  const [subtitles, setSubtitles] = useState('');
  const [currentSubtitleIndex, setCurrentSubtitleIndex] = useState(0);
  const [mode, setMode] = useState('user-mode-1');
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [accuracy, setAccuracy] = useState(0);
  const [windowSize, setWindowSize] = useState(2);
  const [results, setResults] = useState(null);
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [devStats, setDevStats] = useState({ buffer: 0 });

  const modeRef = useRef(mode);
  modeRef.current = mode;

  const showMediaPipe = useMemo(() => mode === 'developer-mode-1', [mode]);

  // --- EFFECT 1: Manages the WebSocket connection. Runs only ONCE. ---
  useEffect(() => {
    console.log("Setting up socket listeners...");

    // This handles the race condition where socket connects before listener is attached.
    if (socket.connected) {
      console.log("Socket was already connected on component mount.");
      setIsConnected(true);
    }

    function onConnect() {
      console.log("✅✅✅ 'connect' event received from socket!");
      setIsConnected(true);
    }

    function onDisconnect() {
      console.log("❌❌❌ 'disconnect' event received from socket!");
      setIsConnected(false);
    }

    function onPrediction(data) {
      if (modeRef.current === 'developer-mode-1') {
        setSubtitles(prev => `${prev}\n> ${data.prediction} (${parseFloat(data.confidence).toFixed(2)})`);
        setAccuracy(parseFloat(data.confidence) * 100);
      }
    }

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('prediction_result', onPrediction);

    socket.connect(); // Harmless if already connected

    return () => {
      console.log("Cleaning up socket listeners.");
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('prediction_result', onPrediction);
      socket.disconnect();
    };
  }, []); // Empty dependency array is correct and intentional.

  // --- EFFECT 2: Manages UI logic that depends on the mode ---
  useEffect(() => {
    let mockSubtitleInterval;
    if (mode !== 'developer-mode-1') {
      if (isRecording || (uploadedVideo && mode === 'user-mode-2')) {
        mockSubtitleInterval = setInterval(() => {
          setSubtitles(prev => {
            const newSubtitle = mockSubtitles[currentSubtitleIndex % mockSubtitles.length];
            return `${prev}\n${newSubtitle}`.slice(-1000);
          });
          setCurrentSubtitleIndex(prev => prev + 1);
        }, 2200);
      }
    } else {
        setSubtitles('');
    }

    return () => {
      clearInterval(mockSubtitleInterval);
    };
  }, [mode, isRecording, uploadedVideo, currentSubtitleIndex]);


  const handleVideoUpload = useCallback((file) => setUploadedVideo(file), []);

const onMediaPipeResults = useCallback((results) => {
    if (showMediaPipe) setResults(results);

    if (mode === 'developer-mode-1' && isRecording && isConnected) {
      const keypoints = extractKeypoints(results);
      
      // Data validation check
      const hasInvalidData = keypoints.some(p => !isFinite(p));
      if (hasInvalidData) {
        console.error("❌ Invalid data detected in keypoints array (NaN or Infinity). Aborting send.");
        return;
      }
      
      // Update UI state
      setDevStats(prev => ({ ...prev, buffer: (prev.buffer >= 59) ? 60 : prev.buffer + 1 }));
      
      // --- ADDING A LOG BEFORE WE EMIT ---
      console.log("Sending keypoints to backend. Length:", keypoints.length);
      
      socket.emit('live_keypoints', keypoints);

    } else if (mode === 'developer-mode-1' && isRecording && !isConnected) {
        setDevStats(prev => ({...prev, buffer: 0}));
    }
}, [showMediaPipe, mode, isRecording, isConnected]);

  useMediaPipe(videoRef, isRecording && showMediaPipe, onMediaPipeResults);

  // --- Render logic ---
  const renderContent = useMemo(() => {
    switch (mode) {
      case 'user-mode-1':
        return (
            <>
              <div className="flex-grow flex items-center justify-center p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
                <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
              </div>
              <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
                <div className="h-64 sm:h-72 md:h-full">
                  <SubtitleDisplay subtitles={subtitles} isRecording={isRecording} />
                </div>
              </div>
            </>
          );
      case 'user-mode-2':
        return (
          uploadedVideo ? (
            <UserMode2Processor
              videoFile={uploadedVideo}
              onReset={() => setUploadedVideo(null)}
            />
          ) : (
            <div className="flex-grow flex items-center justify-center p-4">
              <VideoUploader onFileSelect={handleVideoUpload} />
            </div>
          )
        );
      case 'developer-mode-1':
        return (
          <>
            <div className="flex-grow flex items-center justify-center p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
              <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
              {showMediaPipe && results && <MediaPipeOverlay results={results} videoRef={videoRef} />}
            </div>
            <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
              <DeveloperAnalytics
                accuracy={accuracy}
                isRecording={isRecording}
                isConnected={isConnected}
                bufferSize={devStats.buffer}
              />
              <div className="h-64 sm:h-72 md:h-full">
                <SubtitleDisplay
                  subtitles={subtitles}
                  isRecording={isRecording}
                  accuracy={accuracy}
                  showAccuracy={true}
                />
              </div>
            </div>
          </>
        );
      case 'developer-mode-2':
        return (
          <>
            <div className="flex-grow flex flex-col p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
              <WindowSizeSelector 
                windowSize={windowSize} 
                onWindowSizeChange={setWindowSize} 
              />
              <div className="flex-grow flex items-center justify-center">
                <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
              </div>
            </div>
            <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
              <DeveloperAnalytics 
                accuracy={accuracy} 
                isRecording={isRecording}
                windowSize={windowSize}
                isConnected={isConnected} // Pass prop here too
                bufferSize={devStats.buffer}
              />
              <div className="h-64 sm:h-72 md:h-full">
                <SubtitleDisplay 
                  subtitles={subtitles} 
                  isRecording={isRecording} 
                  accuracy={accuracy}
                  showAccuracy={true}
                />
              </div>
            </div>
          </>
        );
      default:
        return null;
    }
  }, [mode, videoRef, isRecording, subtitles, uploadedVideo, accuracy, windowSize, showMediaPipe, results, handleVideoUpload, isConnected, devStats]);


  return (
    <div className="flex flex-col items-center h-screen bg-gray-900 text-gray-100 font-sans">
      <div className="flex flex-col w-full max-w-screen-2xl h-full p-2 sm:p-3 md:p-4 lg:p-6">
        <div className="flex flex-col md:flex-row flex-1 overflow-hidden rounded-lg md:rounded-xl shadow-xl md:shadow-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700/50">
          {renderContent}
        </div>
        <div className="mt-2 sm:mt-3 md:mt-4 p-2.5 sm:p-3 md:p-3.5 bg-gray-800/70 backdrop-blur-md rounded-lg md:rounded-xl shadow-lg border border-gray-700/50">
          <Controls 
            isRecording={isRecording} 
            onStart={startVideo} 
            onStop={stopVideo}
            mode={mode}
            onModeChange={setMode}
            uploadedVideo={uploadedVideo}
            onClearUpload={() => setUploadedVideo(null)}
          />
        </div>
      </div>
    </div>
  );
}

export default App;