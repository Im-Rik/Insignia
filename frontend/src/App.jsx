import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// Import all your components
import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';
import VideoUploader from './components/VideoUploader';
import UserMode2Processor from './components/UserMode2Processor';
import Recorder2Editor from './components/Recorder2Editor';
import AutomatedEditor from './components/AutomatedEditor';
import DeveloperAnalytics from './components/DeveloperAnalytics';
import MediaPipeOverlay from './components/MediaPipeOverlay';
import HistoryList from './components/HistoryList';
import SymptomSelector from './components/SymptomSelector';
import Prescription from './components/Prescription';
import Landing from './components/Landing';

// Import hooks and utils
import useVideoStream from './hooks/useVideoStream';
import useMediaPipe from './hooks/useMediaPipe';
import { socket } from './socket';
import { extractKeypoints } from './utils/keypointExtractor';

// Add custom styles for animations
const customStyles = `
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: scale(0.8) translateY(10px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  @keyframes fadeOut {
    from {
      opacity: 1;
      transform: translateY(0);
    }
    to {
      opacity: 0;
      transform: translateY(-10px);
    }
  }

  @keyframes pulse {
    0% {
      transform: scale(1);
      box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7);
    }
    70% {
      transform: scale(1.05);
      box-shadow: 0 0 0 10px rgba(6, 182, 212, 0);
    }
    100% {
      transform: scale(1);
      box-shadow: 0 0 0 0 rgba(6, 182, 212, 0);
    }
  }

  @keyframes ripple {
    0% {
      transform: scale(0);
      opacity: 1;
    }
    100% {
      transform: scale(4);
      opacity: 0;
    }
  }

  .animate-fadeIn {
    animation: fadeIn 0.3s ease-out;
  }

  .animate-pulse-once {
    animation: pulse 0.6s ease-out;
  }

  .animate-fadeOut {
    animation: fadeOut 0.3s ease-out forwards;
  }

  .ripple-effect {
    position: relative;
    overflow: hidden;
  }

  .ripple-effect::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.5);
    transform: translate(-50%, -50%);
    animation: ripple 0.6s ease-out;
  }
`;

function App() {
  const { videoRef, isRecording, startVideo, stopVideo } = useVideoStream();
  const [mode, setMode] = useState('user-mode-1');
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [accuracy, setAccuracy] = useState(0);
  const [results, setResults] = useState(null);
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [devStats, setDevStats] = useState({ buffer: 0, latency: 0, pps: 0, videoResolution: '', packetsSent: 0 });
  const [fps, setFps] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [showLanding, setShowLanding] = useState(true);
  const [isEditorActive, setIsEditorActive] = useState(false);
  const [showSymptomPanel, setShowSymptomPanel] = useState(false); // New state for mobile symptom panel
  const [recentlySelected, setRecentlySelected] = useState(null); // Track recently selected symptom for animation

  const modeRef = useRef(mode);
  const lastKeypointTime = useRef(0);
  const frameCountRef = useRef(0);
  const editorRef = useRef(null);

  // This ensures the ref is always up-to-date for use in callbacks
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    // This will log the state to your browser's developer console every time it changes.
    console.log("APP.JSX STATE UPDATED:", selectedSymptoms.map(s => s.name));
  }, [selectedSymptoms]);

  const showMediaPipe = useMemo(() => mode === 'developer-mode-1', [mode]);

  // Handler to switch from landing page to main app
  const handleTryNow = useCallback(() => {
    setShowLanding(false);
  }, []);

  // Symptom handlers
  const handleSymptomSelect = useCallback((symptomToToggle) => {
  setSelectedSymptoms(prevSymptoms => {
    const isAlreadySelected = prevSymptoms.some(s => s.name === symptomToToggle.name);

    if (isAlreadySelected) {
      // If already selected, filter it out to de-select
      return prevSymptoms.filter(s => s.name !== symptomToToggle.name);
    } else {
      // If not selected, add it to the array
      return [...prevSymptoms, symptomToToggle];
    }
  });

  // This animation/feedback part can remain as is
  setRecentlySelected(symptomToToggle.name);
  setTimeout(() => setRecentlySelected(null), 2000);

  if (window.navigator && window.navigator.vibrate) {
    window.navigator.vibrate(50);
  }
}, []);

  const handleClearSymptoms = useCallback(() => {
    setSelectedSymptoms([]);
  }, []);

  // Editor reset handler
  const triggerEditorReset = () => {
    if (editorRef.current) {
      editorRef.current.reset();
    }
  };

  // FPS counter effect
  useEffect(() => {
    const intervalId = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);
    return () => clearInterval(intervalId);
  }, []);

  // Reset editor state when mode changes
  useEffect(() => {
    if (mode !== 'developer-mode-2' && mode !== 'developer-mode-3') {
      setIsEditorActive(false);
    }
  }, [mode]);

  // Socket connection effect - merged logic from both versions
  useEffect(() => {
    // Don't connect if we are on the landing page
    if (showLanding) return;

    const socketModes = ['user-mode-1', 'developer-mode-1'];

    // Connect only if the current mode requires a socket
    if (socketModes.includes(mode)) {
      console.log(`Setting up socket listeners for mode: '${mode}'`);

      const onConnect = () => {
        console.log("✅ Socket connected");
        setIsConnected(true);
      };

      const onDisconnect = () => {
        console.log("❌ Socket disconnected");
        setIsConnected(false);
      };

      const onPrediction = (data) => {
        if (modeRef.current === 'developer-mode-1' || modeRef.current === 'user-mode-1') {
          let roundTripTime = 0;
          if (lastKeypointTime.current > 0) {
            roundTripTime = performance.now() - lastKeypointTime.current;
          }
          const newPrediction = {
            prediction: data.prediction,
            confidence: parseFloat(data.confidence),
            timestamp: Date.now(),
            time: new Date().toLocaleTimeString()
          };
          setPredictionHistory(prevHistory => {
            const updatedHistory = [newPrediction, ...prevHistory].slice(0, 50);
            let currentPPS = 0;
            if (updatedHistory.length > 10) {
              const timeSpan = (updatedHistory[0].timestamp - updatedHistory[9].timestamp) / 1000;
              if (timeSpan > 0) {
                currentPPS = 10 / timeSpan;
              }
            }
            setDevStats(prev => ({ ...prev, latency: roundTripTime, pps: currentPPS }));
            return updatedHistory;
          });
          setAccuracy(newPrediction.confidence * 100);
        }
      };

      socket.on('connect', onConnect);
      socket.on('disconnect', onDisconnect);
      socket.on('prediction_result', onPrediction);

      if (!socket.connected) {
        socket.connect();
      }

      return () => {
        console.log(`Cleaning up socket listeners for mode: '${mode}'`);
        socket.off('connect', onConnect);
        socket.off('disconnect', onDisconnect);
        socket.off('prediction_result', onPrediction);
      };
    } else {
      // If we are in a mode that does NOT use the socket, ensure it's disconnected
      if (socket.connected) {
        console.log(`Entering non-socket mode ('${mode}'). Disconnecting...`);
        socket.disconnect();
      }
    }
  }, [showLanding, mode]);

  const handleVideoUpload = useCallback((file) => setUploadedVideo(file), []);

  // MediaPipe results handler
  const onMediaPipeResults = useCallback((results) => {
    frameCountRef.current++;
    setResults(results);

    setDevStats(prev => {
      let newResolution = prev.videoResolution;
      if (isRecording && videoRef.current && !newResolution) {
        const w = videoRef.current.videoWidth;
        const h = videoRef.current.videoHeight;
        if (w > 0 && h > 0) {
          newResolution = `${w}x${h}`;
        }
      }

      const isLiveMode = modeRef.current === 'developer-mode-1' || modeRef.current === 'user-mode-1';
      if (isLiveMode && isRecording && isConnected) {
        const keypoints = extractKeypoints(results);
        const hasInvalidData = keypoints.some(p => !isFinite(p));
        if (hasInvalidData) {
          console.error("❌ Invalid data detected. Aborting send.");
          return { ...prev, videoResolution: newResolution };
        }
        lastKeypointTime.current = performance.now();
        socket.emit('live_keypoints', keypoints);
        return {
          ...prev,
          videoResolution: newResolution,
          buffer: Math.min(60, prev.buffer + 1),
          packetsSent: prev.packetsSent + 1
        };
      } else if (modeRef.current === 'developer-mode-1' && isRecording && !isConnected) {
        return { ...prev, buffer: 0, videoResolution: newResolution };
      }
      return { ...prev, videoResolution: newResolution };
    });
  }, [isRecording, isConnected]);

  // Hook to manage MediaPipe lifecycle
  const enableMediaPipe = useMemo(() =>
    isRecording && (mode === 'developer-mode-1' || mode === 'user-mode-1'),
    [isRecording, mode]
  );
  useMediaPipe(videoRef, enableMediaPipe, onMediaPipeResults);

  // Memoized content based on the current mode
 // --- REPLACE your entire renderContent function in App.jsx with this one ---

  const renderContent = useMemo(() => {
    const latestPrediction = predictionHistory.length > 0 ? predictionHistory[0] : null;

    // Consolidate layout for user-mode-1 and developer-mode-1
    if (mode === 'user-mode-1' || mode === 'developer-mode-1') {
      return (
        <div className="flex flex-col lg:flex-row h-full w-full relative gap-4">
          {/* === START: MAIN CONTENT BLOCK (VIDEO + SIDEBAR) === */}
          <div className="flex-1 flex flex-col min-w-0"> {/* Flex-1 and min-w-0 to allow shrinking */}
            {/* Video and Sidebar Container */}
            <div className="flex flex-col lg:flex-row flex-1 overflow-hidden rounded-xl border border-gray-700/50 bg-gray-800/70">
              {/* Video Section */}
              <div className="flex-grow flex items-center justify-center p-2 sm:p-3 md:p-4 bg-black/50 relative lg:rounded-l-xl min-h-[200px] sm:min-h-[300px] md:min-h-[400px]">
                <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
                {showMediaPipe && results && (
                  <MediaPipeOverlay results={results} videoRef={videoRef} isProcessing={isRecording} />
                )}
              </div>

              {/* Sidebar Section (Captions & Message or Developer Analytics) */}
              <div className="w-full lg:w-72 xl:w-80 lg:flex-shrink-0 bg-gray-800/80 lg:border-l border-gray-700/60 flex flex-col lg:rounded-r-xl overflow-hidden">
                {mode === 'user-mode-1' ? (
                  <div className="h-auto lg:h-full flex flex-col">
                    <SubtitleDisplay
                      latestPrediction={latestPrediction}
                      isRecording={isRecording}
                      accuracy={accuracy}
                      showAccuracy={true}
                    />
                    <div className="flex-grow p-3 sm:p-4 border-t border-gray-700/60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 min-h-[120px] sm:min-h-[150px]">
                      <div className="flex justify-between items-center mb-3">
                        <h4 className="text-sm sm:text-base font-semibold text-gray-300 uppercase tracking-wider">
                          Your Message
                        </h4>
                        <button
                          onClick={handleClearSymptoms}
                          className="text-xs sm:text-sm text-cyan-400 hover:text-cyan-200 transition-colors duration-200 disabled:opacity-50 px-2 py-1"
                          disabled={selectedSymptoms.length === 0}
                        >
                          Clear
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-2 p-3 bg-black/20 rounded-lg min-h-[5rem] sm:min-h-[6rem]">
                        {selectedSymptoms.length > 0 ? (
                          selectedSymptoms.map((s, index) => (
                            <div
                              key={index}
                              className={`flex items-center gap-1.5 bg-gray-600/50 px-2.5 py-1.5 rounded-full text-sm sm:text-base transform transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer animate-fadeIn ${
                                recentlySelected === s.name ? 'animate-pulse-once ring-2 ring-cyan-400 ring-offset-2 ring-offset-gray-800' : ''
                              }`}
                              style={{
                                animation: 'fadeIn 0.3s ease-out',
                                animationDelay: `${index * 0.05}s`,
                                animationFillMode: 'both'
                              }}
                            >
                              <span className="text-base sm:text-lg">{s.emoji}</span>
                              <span className="text-gray-200">{s.name}</span>
                            </div>
                          ))
                        ) : (
                          <p className="text-gray-400 text-sm sm:text-base p-3 text-center w-full">
                            Select symptoms to build your message.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="h-auto lg:h-1/2 flex-shrink-0 overflow-y-auto border-b border-gray-700/60 scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500">
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
                    <div className="flex-shrink-0">
                      <SubtitleDisplay
                        latestPrediction={latestPrediction}
                        isRecording={isRecording}
                        accuracy={accuracy}
                        showConfidence={false}
                      />
                    </div>
                    <div className="flex-grow min-h-[150px] sm:min-h-[200px] lg:min-h-0">
                      <HistoryList history={predictionHistory} />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
          {/* === END: MAIN CONTENT BLOCK === */}
          
          {/* === START: DETACHED SYMPTOM SELECTOR COLUMN (Desktop) === */}
          {mode !== 'doctor-mode' && (
            <div className="hidden lg:flex w-40 flex-shrink-0">
              <div className="flex flex-col h-full w-full bg-gray-800/70 rounded-xl border border-gray-700/50 overflow-hidden">
                <SymptomSelector
                  onSymptomSelect={handleSymptomSelect}
                  isMobile={false}
                  selectedSymptoms={selectedSymptoms} // <-- THE FIX IS HERE
                />
              </div>
            </div>
          )}
          {/* === END: DETACHED SYMPTOM SELECTOR COLUMN === */}
        </div>
      );
    }

    switch (mode) {
      case 'user-mode-2':
      case 'developer-mode-2':
      case 'developer-mode-3':
        return (
          <div className="flex flex-col lg:flex-row h-full w-full relative gap-4">
            <div className="flex-1 flex flex-col min-w-0">
              {mode === 'user-mode-2' && (
                uploadedVideo ? (
                  <UserMode2Processor videoFile={uploadedVideo} onReset={() => setUploadedVideo(null)} />
                ) : (
                  <div className="flex-grow flex items-center justify-center p-4 sm:p-6 md:p-8">
                    <VideoUploader onFileSelect={handleVideoUpload} />
                  </div>
                )
              )}
              {mode === 'developer-mode-2' && (
                <Recorder2Editor
                  ref={editorRef}
                  videoRef={videoRef}
                  isRecording={isRecording}
                  onEditorStateChange={setIsEditorActive}
                />
              )}
              {mode === 'developer-mode-3' && (
                <AutomatedEditor
                  ref={editorRef}
                  videoRef={videoRef}
                  isRecording={isRecording}
                  onEditorStateChange={setIsEditorActive}
                  selectedSymptoms={selectedSymptoms}
                />
              )}
            </div>
            {mode !== 'doctor-mode' && (
              <div className="hidden lg:flex w-40 flex-shrink-0">
                <div className="flex flex-col h-full w-full bg-gray-800/70 rounded-xl border border-gray-700/50 overflow-hidden">
                  <SymptomSelector
                    onSymptomSelect={handleSymptomSelect}
                    isMobile={false}
                    selectedSymptoms={selectedSymptoms} // <-- AND THE FIX IS HERE
                  />
                </div>
              </div>
            )}
          </div>
        );

      case 'doctor-mode':
        return (
          <div className="p-4 sm:p-6 md:p-8 h-full overflow-y-auto">
            <Prescription
              selectedSymptoms={selectedSymptoms}
              onBack={() => setMode('user-mode-1')}
            />
          </div>
        );

      default:
        return (
          <div className="flex items-center justify-center h-full text-lg sm:text-xl md:text-2xl text-gray-400 p-4">
            Mode not found.
          </div>
        );
    }
  }, [mode, isRecording, uploadedVideo, accuracy, showMediaPipe, results, isConnected,
    predictionHistory, devStats, fps, selectedSymptoms, handleClearSymptoms,
    handleSymptomSelect, handleVideoUpload, videoRef, isEditorActive, showSymptomPanel, recentlySelected]);
    
  // Show landing page if needed
  if (showLanding) {
    return <Landing onTryNow={handleTryNow} />;
  }

  return (
    <div className="flex flex-col items-center min-h-screen h-screen bg-gray-900 text-gray-100 font-sans overflow-hidden">
      <style dangerouslySetInnerHTML={{ __html: customStyles }} />
      <div className="flex flex-col w-full h-full p-2 sm:p-3 md:p-4 lg:p-5 xl:p-6 max-w-screen-2xl mx-auto">
        {/* Main Content Area */}
        <div className="flex flex-col flex-1 overflow-hidden mb-2 sm:mb-3 md:mb-4">
          {renderContent}
        </div>

        {/* Controls Section */}
        <div className="p-2 sm:p-3 md:p-4 bg-gray-800/70 backdrop-blur-md rounded-lg md:rounded-xl shadow-lg border border-gray-700/50">
          {/* Mobile and Tablet Layout */}
          <div className="lg:hidden">
            <div className="flex items-center justify-between gap-2">
              {/* Back Button - Mobile/Tablet */}
              <button
                onClick={() => setShowLanding(true)}
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-all duration-200 text-xs sm:text-sm font-medium flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span className="hidden sm:inline">Back to Home</span>
                <span className="sm:hidden">Back</span>
              </button>

              {/* Start/Stop Button - Mobile/Tablet (Hidden for doctor mode) */}
              {mode !== 'doctor-mode' && (
                <button
                  onClick={isRecording ? stopVideo : startVideo}
                  className={`flex-1 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
                    isRecording
                      ? 'bg-red-600 hover:bg-red-700 text-white'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {isRecording ? (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                      </svg>
                      Stop
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Start Video
                    </>
                  )}
                </button>
              )}
            </div>

            {/* Mode Controls - Mobile/Tablet (No additional start/stop button) */}
            <div className="mt-2 overflow-x-auto -mx-2 px-2">
              <Controls
                isRecording={isRecording}
                onStart={startVideo}
                onStop={stopVideo}
                mode={mode}
                onModeChange={(newMode) => {
                  setMode(newMode);
                  if (newMode !== 'user-mode-1') {
                    setSelectedSymptoms([]);
                  }
                }}
                uploadedVideo={uploadedVideo}
                onClearUpload={() => setUploadedVideo(null)}
                onResetEditor={triggerEditorReset}
                isEditorActive={isEditorActive}
                hideStartStop={true}  // Hide start/stop button in Controls on mobile/tablet
                hideStartStopForMode={mode === 'doctor-mode'} // Hide for doctor mode
              />
            </div>
          </div>

          {/* Desktop Layout (1024px and up) */}
          <div className="hidden lg:flex items-center gap-3 lg:gap-4">
            {/* Back Button - Desktop */}
            <button
              onClick={() => setShowLanding(true)}
              className="px-4 py-2.5 bg-gray-700 hover:bg-gray-600 rounded-lg transition-all duration-200 text-sm font-medium flex items-center gap-2 hover:scale-105 transform"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Home
            </button>

            {/* Controls Component - Desktop (With start/stop button) */}
            <div className="flex-1">
              <Controls
                isRecording={isRecording}
                onStart={startVideo}
                onStop={stopVideo}
                mode={mode}
                onModeChange={(newMode) => {
                  setMode(newMode);
                  if (newMode !== 'user-mode-1') {
                    setSelectedSymptoms([]);
                  }
                }}
                uploadedVideo={uploadedVideo}
                onClearUpload={() => setUploadedVideo(null)}
                onResetEditor={triggerEditorReset}
                isEditorActive={isEditorActive}
                hideStartStop={false}  // Show start/stop button on desktop
                hideStartStopForMode={false} // Always show on desktop
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;