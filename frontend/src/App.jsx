// App.jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
import useWebSocket from './hooks/useWebSocket';

import { mockSubtitles } from './utils/subtitles';
import { extractAllKeypoints } from './utils/keypointUtils';

const WEBSOCKET_URL = 'ws://localhost:8000';
const KEYPOINT_SEND_INTERVAL = 100; // Send keypoints every 100ms instead of every frame

function App() {
  const { videoRef, isRecording, startVideo, stopVideo } = useVideoStream();
  const [subtitles, setSubtitles] = useState('');
  const [currentSubtitleIndex, setCurrentSubtitleIndex] = useState(0);
  const [mode, setMode] = useState('user-mode-1');
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [accuracy, setAccuracy] = useState(0);
  const [windowSize, setWindowSize] = useState(2); // seconds
  const [results, setResults] = useState(null);

  
  // Track last keypoint send time
  const lastKeypointSendTime = React.useRef(0);
  const keypointBuffer = React.useRef([]);

  // Memoize showMediaPipe to prevent unnecessary recalculations
  const showMediaPipe = useMemo(() => mode === 'developer-mode-1', [mode]);

  // Subtitle update effect
  useEffect(() => {
    if (!isRecording && !(uploadedVideo && mode === 'user-mode-2')) {
      return;
    }

    const intervalId = setInterval(() => {
      setSubtitles(prev => {
        const newSubtitle = mockSubtitles[currentSubtitleIndex % mockSubtitles.length];
        const lines = `${prev}\n${newSubtitle}`.split('\n');
        return lines.slice(-10).join('\n');
      });
      
      setCurrentSubtitleIndex(prev => prev + 1);
      
      if (mode.includes('developer') || mode === 'user-mode-2') {
        setAccuracy(Math.random() * 30 + 70); // 70-100%
      }
    }, 2200);

    return () => clearInterval(intervalId);
  }, [isRecording, currentSubtitleIndex, uploadedVideo, mode]);

  const handleVideoUpload = useCallback((file) => {
    setUploadedVideo(file);
  }, []);

  // Throttled keypoint sending
  const handleSendToBackend = useCallback((keypointsArray) => {
    const now = Date.now();
    
    // Buffer keypoints
    keypointBuffer.current.push(keypointsArray);
    
    // Only send if enough time has passed
    if (now - lastKeypointSendTime.current >= KEYPOINT_SEND_INTERVAL) {
      // Send the most recent keypoints
      const keypointsToSend = keypointBuffer.current[keypointBuffer.current.length - 1];
      console.log("Sending keypoints to backend:", keypointsToSend.length, "values");
      
      // Clear buffer and update time
      keypointBuffer.current = [];
      lastKeypointSendTime.current = now;
      
      // Here you would actually send to WebSocket
      // ws.send(JSON.stringify({ keypoints: keypointsToSend }));
    }
  }, []);

  // Optimized MediaPipe results handler
  const onMediaPipeResults = useCallback((results) => {
    // Only update results if in developer mode
    if (showMediaPipe) {
      setResults(results);
    }
    
    // Extract and send keypoints
    const keypointsArray = extractAllKeypoints(results);
    handleSendToBackend(keypointsArray);
  }, [handleSendToBackend, showMediaPipe]);
  
  // Use MediaPipe with optimizations
  useMediaPipe(videoRef, isRecording && showMediaPipe, onMediaPipeResults);

  // Memoize content rendering to prevent unnecessary recalculations
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
          // If a video is uploaded, show the processor. Otherwise, show the uploader.
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
              <DeveloperAnalytics accuracy={accuracy} isRecording={isRecording} />
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
  }, [mode, videoRef, isRecording, subtitles, uploadedVideo, accuracy, windowSize, showMediaPipe, results, handleVideoUpload]);

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