// App.jsx (Final Corrected Version)
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';
import VideoUploader from './components/VideoUploader';
import UserMode2Processor from './components/UserMode2Processor';
import Recorder2Editor from './components/Recorder2Editor'
import DeveloperAnalytics from './components/DeveloperAnalytics';
import MediaPipeOverlay from './components/MediaPipeOverlay';
import HistoryList from './components/HistoryList';
import useVideoStream from './hooks/useVideoStream';
import useMediaPipe from './hooks/useMediaPipe';
import { socket } from './socket';
import { extractKeypoints } from './utils/keypointExtractor';

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
    const [isEditorActive, setIsEditorActive] = useState(false);

    const modeRef = useRef(mode);
    modeRef.current = mode;

    const showMediaPipe = useMemo(() => mode === 'developer-mode-1', [mode]);
    const lastKeypointTime = useRef(0);

    const frameCountRef = useRef(0);
    const editorRef = useRef(null);

    const triggerEditorReset = () => {
        if (editorRef.current) {
            editorRef.current.reset();
        }
    };

    useEffect(() => {
        const intervalId = setInterval(() => {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
        }, 1000);
        return () => clearInterval(intervalId);
    }, []);

    useEffect(() => {
        if (mode !== 'developer-mode-2') {
            setIsEditorActive(false);
        }
    }, [mode]);

    // --- EFFECT 1: Manages the WebSocket connection. ---
    useEffect(() => {
        // If the socket isn't already connected, connect it.
        if (!socket.connected) {
            socket.connect();
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
                let roundTripTime = 0;
                if (lastKeypointTime.current > 0) {
                    roundTripTime = performance.now() - lastKeypointTime.current;
                }
                const newPrediction = {
                    prediction: data.prediction,
                    confidence: parseFloat(data.confidence),
                    timestamp: new Date().getTime(),
                    time: new Date().toLocaleTimeString()
                };
                setPredictionHistory(prevHistory => {
                    const updatedHistory = [newPrediction, ...prevHistory].slice(0, 50);
                    let currentPPS = 0;
                    if (updatedHistory.length > 10) {
                        const timeSpan = (updatedHistory[0].timestamp - updatedHistory[9].timestamp) / 1000;
                        if (timeSpan > 0) { currentPPS = 10 / timeSpan; }
                    }
                    setDevStats(prev => ({ ...prev, latency: roundTripTime, pps: currentPPS }));
                    return updatedHistory;
                });
                setAccuracy(newPrediction.confidence * 100);
            }
        }

        socket.on('connect', onConnect);
        socket.on('disconnect', onDisconnect);
        socket.on('prediction_result', onPrediction);

        // This cleanup function now ONLY removes the listeners.
        // It does NOT disconnect the socket, allowing the connection to be stable.
        return () => {
            console.log("Cleaning up App.jsx socket listeners.");
            socket.off('connect', onConnect);
            socket.off('disconnect', onDisconnect);
            socket.off('prediction_result', onPrediction);
            // The socket.disconnect(); line has been removed.
        };
    }, []); // Empty dependency array is correct and intentional.


    const handleVideoUpload = useCallback((file) => setUploadedVideo(file), []);

    const onMediaPipeResults = useCallback((results) => {
        frameCountRef.current++;
        setResults(results);
        setDevStats(prev => {
            let newResolution = prev.videoResolution;
            if (isRecording && videoRef.current && !newResolution) {
                const w = videoRef.current.videoWidth;
                const h = videoRef.current.videoHeight;
                if (w > 0 && h > 0) { newResolution = `${w}x${h}`; }
            }
            const isLiveMode = mode === 'developer-mode-1' || mode === 'user-mode-1';
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
                    buffer: (prev.buffer >= 59) ? 60 : prev.buffer + 1,
                    packetsSent: prev.packetsSent + 1
                };
            } else if (mode === 'developer-mode-1' && isRecording && !isConnected) {
                return { ...prev, buffer: 0 };
            }
            return { ...prev, videoResolution: newResolution };
        });
    }, [mode, isRecording, isConnected]);

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
                        </div>
                        <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
                            <div className="h-64 sm:h-72 md:h-full"> <SubtitleDisplay latestPrediction={latestPrediction} isRecording={isRecording} accuracy={accuracy} showAccuracy={true} /> </div>
                            <div className="flex-grow min-h-0"> <HistoryList history={predictionHistory} /> </div>
                        </div>
                    </>
                );
            case 'user-mode-2':
                return (
                    uploadedVideo ? (<UserMode2Processor videoFile={uploadedVideo} onReset={() => setUploadedVideo(null)} />) : (<div className="flex-grow flex items-center justify-center p-4"> <VideoUploader onFileSelect={handleVideoUpload} /> </div>)
                );
            case 'developer-mode-1':
                return (
                    <>
                        <div className="flex-grow flex items-center justify-center p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
                            <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
                            {showMediaPipe && results && <MediaPipeOverlay results={results} videoRef={videoRef} isProcessing={isRecording} />}
                        </div>
                        <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
                            <div className="h-1/2 flex-shrink-0 overflow-y-auto border-b border-gray-700/60 scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500">
                                <DeveloperAnalytics accuracy={accuracy} isRecording={isRecording} isConnected={isConnected} bufferSize={devStats.buffer} latency={devStats.latency} pps={devStats.pps} fps={fps} videoResolution={devStats.videoResolution} />
                            </div>
                            <div className="flex-shrink-0"> <SubtitleDisplay latestPrediction={latestPrediction} isRecording={isRecording} accuracy={accuracy} showConfidence={false} /> </div>
                            <div className="flex-grow min-h-0"> <HistoryList history={predictionHistory} /> </div>
                        </div>
                    </>
                );
            case 'developer-mode-2':
                return (
                    <Recorder2Editor
                        ref={editorRef}
                        videoRef={videoRef}
                        isRecording={isRecording}
                        onEditorStateChange={setIsEditorActive}
                    />
                );
            default:
                return null;
        }
    }, [mode, videoRef, isRecording, uploadedVideo, accuracy, showMediaPipe, results, handleVideoUpload, isConnected, predictionHistory, devStats.buffer, devStats.latency, devStats.pps, fps, devStats.videoResolution, isEditorActive]);


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
                        onResetEditor={triggerEditorReset}
                        isEditorActive={isEditorActive}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;