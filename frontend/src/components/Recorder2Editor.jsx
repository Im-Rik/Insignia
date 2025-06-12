import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import VideoDisplay from './VideoDisplay';

// --- UI Icons (Unchanged) ---
const PlayIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M7 4v16l13-8L7 4z"></path></svg>;
const PauseIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>;
const MarkInIcon = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M6 4h2v12H6V4zm10-2H8v2h8V2zM8 20h8v-2H8v2zm8-6h-2V8h2v6z"></path></svg>;
const MarkOutIcon = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M14 4h-2v12h2V4zM4 2h8v2H4V2zm8 18H4v-2h8v2zM8 14H6V8h2v6z"></path></svg>;
const TrashIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"></path></svg>;
const SendIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>;

// --- formatTime function (Unchanged) ---
const formatTime = (timeInSeconds) => {
    if (!isFinite(timeInSeconds) || timeInSeconds < 0) return '00:00.0';
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(4, '0')}`;
};


const Recorder2Editor = forwardRef(({ videoRef, isRecording, onEditorStateChange }, ref) => {
    const [recordedBlobUrl, setRecordedBlobUrl] = useState(null);
    const mediaRecorderRef = useRef(null);
    const editorVideoRef = useRef(null);
    const timelineContainerRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [inPoint, setInPoint] = useState(null);
    const [segments, setSegments] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMessage, setStatusMessage] = useState('Welcome! Use the controls below to create segments.');
    const wasPlayingBeforeScrub = useRef(false);
    const [loopingSegment, setLoopingSegment] = useState(null);
    const eventSourceRef = useRef(null); // Ref to hold the EventSource instance

    // --- Core Recording and Reset Logic (Largely Unchanged) ---
    const handleResetEditor = () => {
        if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
        if (eventSourceRef.current) {
            console.log("Closing existing EventSource connection.");
            eventSourceRef.current.close();
        }
        setRecordedBlobUrl(null); setIsPlaying(false); setDuration(0); setCurrentTime(0);
        setInPoint(null); setSegments([]); setIsProcessing(false); setLoopingSegment(null);
        onEditorStateChange(false);
        setStatusMessage('Welcome! Use the controls below to create segments.');
    };

    useEffect(() => {
        return () => { // Cleanup on unmount
            if (eventSourceRef.current) {
                console.log("Closing EventSource connection on component unmount.");
                eventSourceRef.current.close();
            }
        };
    }, []);

    useEffect(() => {
        if (isRecording) {
            const stream = videoRef.current?.srcObject;
            if (!stream) return;
            handleResetEditor();
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
            const chunks = [];
            mediaRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data) };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                setRecordedBlobUrl(url);
                onEditorStateChange(true);
            };
            mediaRecorderRef.current.start();
        } else if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, [isRecording, videoRef, onEditorStateChange]);

    useImperativeHandle(ref, () => ({ reset: handleResetEditor }));

    // --- NEW: HTTP/SSE based processing function ---
    const handleProcessSegments = async () => {
        if (segments.length === 0) {
            setStatusMessage("Please create at least one segment first.");
            return;
        }
        if (!recordedBlobUrl) {
            setStatusMessage("Could not find the recorded video.");
            return;
        }

        setIsProcessing(true);
        setLoopingSegment(null);
        setStatusMessage(`🚀 Uploading video and ${segments.length} segment(s)...`);
        setSegments(prev => prev.map(seg => ({ ...seg, prediction: 'pending...', confidence: null })));

        try {
            // 1. Fetch the recorded data as a Blob
            const videoBlob = await fetch(recordedBlobUrl).then(r => r.blob());

            // 2. Prepare FormData
            const formData = new FormData();
            formData.append('video', videoBlob, 'recording.webm');
            formData.append('segments', JSON.stringify(segments.map(({ id, start, end }) => ({ id, start, end }))));

            // 3. POST to the `/classify` endpoint to start the job
            const response = await fetch('http://localhost:5000/classify', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status ${response.status}`);
            }

            const { job_id } = await response.json();
            setStatusMessage(`⏳ Classification started (Job ID: ${job_id}). Waiting for results...`);

            // 4. Open a Server-Sent Events (SSE) connection to stream results
            const es = new EventSource(`http://localhost:5000/stream/${job_id}`);
            eventSourceRef.current = es; // Store the connection in the ref

            es.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error("Received error from server:", data);
                    setStatusMessage(`Error on segment ${data.segmentId}: ${data.error}`);
                } else {
                    console.log("Received prediction:", data);
                    setSegments(prev =>
                        prev.map(seg =>
                            seg.id === data.segmentId
                                ? { ...seg, prediction: data.prediction, confidence: data.confidence }
                                : seg
                        )
                    );
                }
            };

            es.onerror = (err) => {
                console.error("EventSource failed:", err);
                setStatusMessage("Connection to server lost. Please try again.");
                es.close();
                setIsProcessing(false);
            };

            // This custom event listener will detect when the server closes the connection
            es.addEventListener('close', () => {
                console.log('Server closed the connection.');
                setStatusMessage(`✅ All segments processed!`);
                es.close();
                setIsProcessing(false);
                eventSourceRef.current = null; // Clear the ref
            });


        } catch (error) {
            console.error("❌ Error during processing request:", error);
            setStatusMessage(`Error: ${error.message}`);
            setIsProcessing(false);
        }
    };


    // --- UI and Playback Controls (Largely Unchanged) ---
    const togglePlayPause = () => { if (!editorVideoRef.current) return; setLoopingSegment(null); editorVideoRef.current.paused ? editorVideoRef.current.play() : editorVideoRef.current.pause(); };
    const playSegmentInLoop = (segment) => { if (!editorVideoRef.current) return; setLoopingSegment(segment); editorVideoRef.current.currentTime = segment.start; if (editorVideoRef.current.paused) { editorVideoRef.current.play(); } };
    const handleMarkIn = () => { setInPoint(currentTime); setStatusMessage(`In-point set at ${formatTime(currentTime)}.`); };
    const handleMarkOut = () => { if (inPoint === null || currentTime <= inPoint) { setStatusMessage("Error: 'In' point must be before 'Out' point."); return; } const newSegment = { id: Date.now(), start: inPoint, end: currentTime, prediction: null, confidence: null, }; setSegments(prev => [...prev, newSegment].sort((a, b) => a.start - b.start)); setInPoint(null); setStatusMessage(`Segment created!`); };
    const deleteSegment = (id) => setSegments(prev => prev.filter(seg => seg.id !== id));

    const handleScrubStart = (e) => {
        if (!editorVideoRef.current || !timelineContainerRef.current) return;
        e.preventDefault();
        setLoopingSegment(null);
        wasPlayingBeforeScrub.current = !editorVideoRef.current.paused;
        editorVideoRef.current.pause();
        const timelineRect = timelineContainerRef.current.getBoundingClientRect();
        const updateScrubTime = (clientX) => {
            const newTime = Math.max(0, Math.min(duration, ((clientX - timelineRect.left) / timelineRect.width) * duration));
            if (isFinite(newTime)) {
                editorVideoRef.current.currentTime = newTime;
                setCurrentTime(newTime);
            }
        };
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        updateScrubTime(clientX);
        const handleScrubMove = (moveEvent) => {
            const clientX = moveEvent.touches ? moveEvent.touches[0].clientX : moveEvent.clientX;
            updateScrubTime(clientX);
        };
        const handleScrubEnd = () => {
            if (wasPlayingBeforeScrub.current) { editorVideoRef.current.play(); }
            window.removeEventListener('mousemove', handleScrubMove);
            window.removeEventListener('mouseup', handleScrubEnd);
            window.removeEventListener('touchmove', handleScrubMove);
            window.removeEventListener('touchend', handleScrubEnd);
        };
        window.addEventListener('mousemove', handleScrubMove);
        window.addEventListener('mouseup', handleScrubEnd);
        window.addEventListener('touchmove', handleScrubMove);
        window.addEventListener('touchend', handleScrubEnd);
    };

    const handleDurationUpdate = (e) => {
        const newDuration = e.currentTarget.duration;
        if (isFinite(newDuration) && newDuration > 0 && duration !== newDuration) {
            setDuration(newDuration);
        }
    };


    // --- Render Functions (Unchanged, but button state logic is now more robust) ---
    const renderEditorUI = () => (
        <div className="w-full h-full flex flex-col p-4 gap-4 bg-gray-900 text-white font-sans">
            {/* Video Player and Segment List */}
            <div className="flex-grow flex flex-col md:flex-row gap-4 min-h-0">
                 {/* Video Player */}
                <div className="w-full md:w-2/3 flex items-center justify-center bg-black rounded-lg shadow-lg">
                    <video
                        ref={editorVideoRef} src={recordedBlobUrl} className="max-w-full max-h-full"
                        onLoadedMetadata={handleDurationUpdate} onDurationChange={handleDurationUpdate} onCanPlay={handleDurationUpdate}
                        onTimeUpdate={e => {
                            const video = e.currentTarget;
                            if (isFinite(video.currentTime)) { setCurrentTime(video.currentTime); }
                            if (loopingSegment && video.currentTime >= loopingSegment.end) { video.currentTime = loopingSegment.start; }
                        }}
                        onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)}
                    />
                </div>
                {/* Segment List */}
                <div className="w-full md:w-1/3 bg-gray-800 rounded-lg p-3 flex flex-col">
                    <h3 className="text-lg font-semibold border-b border-gray-600 pb-2 mb-3 flex-shrink-0">Segments</h3>
                    <p className="text-sm text-gray-400 mb-3 flex-shrink-0 h-10">{statusMessage}</p>
                    <div className="flex-grow overflow-y-auto pr-2 space-y-2">
                        {segments.length > 0 ? segments.map((seg, i) => (
                            <div key={seg.id} onClick={() => !isProcessing && playSegmentInLoop(seg)} className={`p-3 rounded-md transition-all border-2 ${isProcessing ? 'cursor-default' : 'cursor-pointer'} ${loopingSegment?.id === seg.id ? 'bg-cyan-800/50 border-cyan-400' : 'bg-gray-700/80 border-transparent hover:border-gray-500'}`}>
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="font-bold text-gray-300">Segment {i + 1}</div>
                                        <div className="text-xs text-cyan-400 font-mono">{formatTime(seg.start)} &rarr; {formatTime(seg.end)}</div>
                                    </div>
                                    <button onClick={(e) => { e.stopPropagation(); deleteSegment(seg.id); }} disabled={isProcessing} className="text-gray-500 hover:text-red-500 disabled:opacity-25"><TrashIcon /></button>
                                </div>
                                {seg.prediction && (
                                    <div className="mt-2 pt-2 border-t border-gray-600/50">
                                        {seg.prediction === 'pending...' ? (
                                            <span className="text-sm text-yellow-400">Awaiting result...</span>
                                        ) : (
                                            <>
                                                <span className="font-bold text-xl text-green-400">{seg.prediction}</span>
                                                {seg.confidence && <span className="text-sm text-gray-400 ml-2">({(seg.confidence * 100).toFixed(1)}%)</span>}
                                            </>
                                        )}
                                    </div>
                                )}
                            </div>
                        )) : <div className="text-gray-500 text-center mt-8 px-4">Create a segment to begin.</div>}
                    </div>
                </div>
            </div>

            {/* Timeline and Controls */}
            <div className="flex-shrink-0 p-3 bg-gray-800/90 rounded-lg flex items-center gap-4 flex-wrap">
                <button onClick={togglePlayPause} className="text-white hover:text-cyan-400">{isPlaying ? <PauseIcon /> : <PlayIcon />}</button>
                <div className="font-mono text-sm text-gray-300">{formatTime(currentTime)}</div>
                <div ref={timelineContainerRef} onMouseDown={handleScrubStart} onTouchStart={handleScrubStart} className="flex-grow h-4 flex items-center group relative cursor-pointer pr-2">
                    <div className="w-full h-2 bg-gray-600 rounded-full relative">
                        <div className="absolute h-full" style={{ left: `${duration > 0 ? (currentTime / duration) * 100 : 0}%`, pointerEvents: 'none' }}><div className="w-4 h-4 bg-white rounded-full -translate-x-1/2 -translate-y-1/2 top-1/2 absolute"></div></div>
                        {segments.map(seg => (<div key={`timeline-${seg.id}`} className={`absolute h-full opacity-75 rounded-full ${loopingSegment?.id === seg.id ? 'bg-cyan-400' : 'bg-green-500'}`} style={{ left: `${duration > 0 ? (seg.start / duration) * 100 : 0}%`, width: `${duration > 0 ? ((seg.end - seg.start) / duration) * 100 : 0}%` }}></div>))}
                        {inPoint !== null && <div className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-yellow-400 rounded-full" style={{ left: `${duration > 0 ? (inPoint / duration) * 100 : 0}%` }}></div>}
                    </div>
                </div>
                <div className="font-mono text-sm text-gray-500">{formatTime(duration)}</div>
                <button onClick={handleMarkIn} disabled={inPoint !== null || isProcessing} className="disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1.5 rounded-md transition-colors"><MarkInIcon /> In</button>
                <button onClick={handleMarkOut} disabled={inPoint === null || isProcessing} className="disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1.5 rounded-md transition-colors"><MarkOutIcon /> Out</button>
                <div className="w-px h-6 bg-gray-700"></div>
                <button onClick={handleProcessSegments} disabled={isProcessing || segments.length === 0} className="flex items-center gap-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white font-semibold px-4 py-1.5 rounded-md">
                    <SendIcon />
                    {isProcessing ? 'Processing...' : `Process ${segments.length || ''} Segment(s)`}
                </button>
            </div>
        </div>
    );

    return recordedBlobUrl ? renderEditorUI() : <VideoDisplay videoRef={videoRef} isRecording={isRecording} />;
});

export default Recorder2Editor;