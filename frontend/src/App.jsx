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
import HistoryList from './components/HistoryList';
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
  const [devStats, setDevStats] = useState({ buffer: 0, latency: 0, pps: 0, videoResolution: '', packetsSent: 0 });
  const [fps, setFps] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState([]);

  const modeRef = useRef(mode);
  modeRef.current = mode;

  const showMediaPipe = useMemo(() => mode === 'developer-mode-1', [mode]);
  const lastKeypointTime = useRef(0);

  const frameCountRef = useRef(0);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000); // Runs every 1 second

    // Cleanup the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []);

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
      if (modeRef.current === 'developer-mode-1' || modeRef.current === 'user-mode-1') {

        // --- LATENCY CALCULATION ---
        let roundTripTime = 0;
        if (lastKeypointTime.current > 0) {
          roundTripTime = performance.now() - lastKeypointTime.current;
          // lastKeypointTime.current = 0; // Reset for the next measurement
        }
        
        // Create a new prediction object, matching the format for HistoryList
        const newPrediction = {
          prediction: data.prediction,
          confidence: parseFloat(data.confidence),
          timestamp: new Date().getTime(),
          time: new Date().toLocaleTimeString()
        };

        // Add the new object to our history array
        setPredictionHistory(prevHistory => { 
          
          const updatedHistory = [newPrediction, ...prevHistory].slice(0, 50);

          // --- PPS CALCULATION ---
           let currentPPS = 0;
          if (updatedHistory.length > 10) {
            const timeSpan = (updatedHistory[0].timestamp - updatedHistory[9].timestamp) / 1000; // time for last 10 preds in seconds
            if (timeSpan > 0) { // Avoid division by zero
               currentPPS = 10 / timeSpan;
            }
            
          }

          setDevStats(prev => ({ ...prev, latency: roundTripTime, pps: currentPPS }));
          
          return updatedHistory;


        });

        // Continue to update the accuracy from the latest prediction
        setAccuracy(newPrediction.confidence * 100);
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


const handleVideoUpload = useCallback((file) => setUploadedVideo(file), []);

const onMediaPipeResults = useCallback((results) => {
    frameCountRef.current++;
  
   setResults(results);

    // FIX: Combined all state updates into a single, efficient setDevStats call.
    setDevStats(prev => {
        let newResolution = prev.videoResolution;
        if (isRecording && videoRef.current && !newResolution) {
            const w = videoRef.current.videoWidth;
            const h = videoRef.current.videoHeight;
            if (w > 0 && h > 0) {
                newResolution = `${w}x${h}`;
            }
        }

        const isLiveMode = mode === 'developer-mode-1' || mode === 'user-mode-1';
        
        if (isLiveMode && isRecording && isConnected) {
            const keypoints = extractKeypoints(results);
            const hasInvalidData = keypoints.some(p => !isFinite(p));
            if (hasInvalidData) {
                console.error("❌ Invalid data detected. Aborting send.");
                return { ...prev, videoResolution: newResolution }; // Return updated resolution but skip buffer update
            }
            
            lastKeypointTime.current = performance.now();
            socket.emit('live_keypoints', keypoints);

            return {
                ...prev,
                videoResolution: newResolution,
                buffer: (prev.buffer >= 59) ? 60 : prev.buffer + 1,
                packetsSent: prev.packetsSent + 1
            };
        } else if (mode === 'developer-mode-1' && isRecording && !isConnected) {
             return {...prev, buffer: 0};
        }

        return { ...prev, videoResolution: newResolution };
    });

  }, [ mode, isRecording, isConnected]);

   const enableMediaPipe = useMemo(() => isRecording && (mode === 'developer-mode-1' || mode === 'user-mode-1'), [isRecording, mode]);
  useMediaPipe(videoRef, enableMediaPipe, onMediaPipeResults);

  // --- Render logic ---
  const renderContent = useMemo(() => {
    const latestPrediction = predictionHistory.length > 0 ? predictionHistory[0] : null;
    switch (mode) {
      case 'user-mode-1':
        return (
          <>
            <div className="flex-grow flex items-center justify-center p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
              <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
              {/* Note: MediaPipeOverlay is NOT rendered here */}
            </div>
            <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
              {/* Note: DeveloperAnalytics is NOT rendered here */}
              <div className="h-64 sm:h-72 md:h-full">
                <SubtitleDisplay
                  latestPrediction={latestPrediction}
                  isRecording={isRecording}
                  accuracy={accuracy}
                  showAccuracy={true}
                />
              </div>
              <div className="flex-grow min-h-0">
                <HistoryList history={predictionHistory} />
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
              {showMediaPipe && results && <MediaPipeOverlay results={results} videoRef={videoRef} isProcessing={isRecording}/>}
            </div>
            <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">

              {/* 1. Analytics Container: Takes top 50% of height and scrolls if content overflows. */}
              <div className="h-1/2 flex-shrink-0 overflow-y-auto border-b border-gray-700/60 scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500">
                <DeveloperAnalytics
                  accuracy={accuracy}
                  isRecording={isRecording}
                  isConnected={isConnected}
                  bufferSize={devStats.buffer}
                  latency={devStats.latency} 
                  pps={devStats.pps} 
                  fps={fps}
                  videoResolution={devStats.videoResolution} 
                />
              </div>

              {/* 2. Subtitle Container: Takes its natural height and will not be squished. */}
              <div className="flex-shrink-0">
                <SubtitleDisplay
                  latestPrediction={latestPrediction}
                  isRecording={isRecording}
                  accuracy={accuracy}
                  showConfidence={false}
                />
              </div>
              
              {/* 3. History Container: Takes all remaining space and allows its list to scroll. */}
              <div className="flex-grow min-h-0">
                <HistoryList history={predictionHistory} />
              </div>
            </div>
          </>
        );
      case 'developer-mode-2':
        return (
          <>
            <div> Yet to code</div>
          </>
        );
      default:
        return null;
    }
  }, [mode, videoRef, isRecording, subtitles, uploadedVideo, accuracy, windowSize, showMediaPipe, results, handleVideoUpload, isConnected, predictionHistory, devStats.buffer, devStats.latency, devStats.pps, fps, devStats.videoResolution]);


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