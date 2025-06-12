import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import VideoDisplay from './VideoDisplay';

// --- UI Icons (Unchanged) ---
const PlayIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M7 4v16l13-8L7 4z"></path></svg>;
const PauseIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>;
const TrashIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"></path></svg>;
const SendIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>;
const AutoIcon = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M11.832 2.25A5.25 5.25 0 006.75 7.5v.032l1.18-.393A3.75 3.75 0 0112 11.25v2.538l-2.02-.674a.75.75 0 00-.83.056l-.585.585a.75.75 0 000 1.06l4.242 4.243a.75.75 0 001.06 0l4.243-4.243a.75.75 0 000-1.06l-.585-.585a.75.75 0 00-.83-.056l-2.02.674v-2.538a3.75 3.75 0 014.07-3.718l1.18.393V7.5a5.25 5.25 0 00-5.25-5.25h-.008z" clipRule="evenodd"></path></svg>;
const UploadIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>;

const formatTime = (timeInSeconds) => {
    if (!isFinite(timeInSeconds) || timeInSeconds < 0) return '00:00.0';
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(4, '0')}`;
};

const AutomatedEditor = forwardRef(({ videoRef, isRecording, onEditorStateChange }, ref) => {
    const [videoSrc, setVideoSrc] = useState(null);
    const [originalBlob, setOriginalBlob] = useState(null);
    const mediaRecorderRef = useRef(null);
    const editorVideoRef = useRef(null);
    const timelineContainerRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [segments, setSegments] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMessage, setStatusMessage] = useState('Record or upload a video to begin.');
    const wasPlayingBeforeScrub = useRef(false);
    const [loopingSegment, setLoopingSegment] = useState(null);
    const fileInputRef = useRef(null);
    const eventSourceRef = useRef(null);

    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                console.log("Closing EventSource connection.");
                eventSourceRef.current.close();
            }
        };
    }, []);

    const handleResetEditor = () => {
        if (videoSrc && videoSrc.startsWith('blob:')) URL.revokeObjectURL(videoSrc);
        setVideoSrc(null); setOriginalBlob(null); setIsPlaying(false); setDuration(0); setCurrentTime(0);
        setSegments([]); setIsProcessing(false); setLoopingSegment(null); onEditorStateChange(false);
        setStatusMessage('Record or upload a video to begin.');
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
        }
    };

    useEffect(() => {
        if (isRecording) {
            const stream = videoRef.current?.srcObject; if (!stream) return;
            handleResetEditor(); mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
            const chunks = []; mediaRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data) };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' }); const url = URL.createObjectURL(blob);
                setOriginalBlob(blob); setVideoSrc(url); onEditorStateChange(true);
                setStatusMessage('Recording finished. Click "Auto-Detect" to process.');
            }; mediaRecorderRef.current.start();
        } else if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, [isRecording, videoRef, onEditorStateChange]);

    useImperativeHandle(ref, () => ({ reset: handleResetEditor }));

    // --- MODIFIED FUNCTION ---
    // Now only manages the server communication and updates state when data arrives.
    // It no longer handles the initial UI switch.
    const processVideoOnServer = async (videoBlob) => {
        if (!videoBlob) { setStatusMessage("Error: No video data available."); return; }
        setIsProcessing(true);
        setLoopingSegment(null);
        setSegments([]); // Clear previous segments
        setStatusMessage("🤖 Uploading and analyzing video... This may take a moment.");
        try {
            const formData = new FormData();
            formData.append('video', videoBlob, 'video.mp4');
            const response = await fetch('http://localhost:8000/segment', { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status ${response.status}`);
            }
            const { segments: detectedSegments, processed_video_url } = await response.json();
            
            // The key change: update the video source to the one processed by the server
            const processedVideoUrl = `http://localhost:8000${processed_video_url}`;
            setVideoSrc(processedVideoUrl);
            
            if (detectedSegments.length === 0) {
                setStatusMessage("✅ Analysis complete. No distinct gestures were detected.");
            } else {
                setSegments(detectedSegments.map((seg, i) => ({ id: Date.now() + i, start: seg.start, end: seg.end, prediction: null, confidence: null })));
                setStatusMessage(`✅ Success! Review segments and send for classification.`);
            }
        } catch (error) {
            console.error("❌ Error during segmentation:", error);
            setStatusMessage(`Error: ${error.message}`);
            // Don't set onEditorStateChange(false) here, as we want to stay in the editor
        } finally {
            setIsProcessing(false);
        }
    };
    
    // --- MODIFIED FUNCTION ---
    // This now immediately switches to the editor UI with a local video preview.
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            handleResetEditor(); // Clear any previous state

            // Create a local URL for instant preview
            const localUrl = URL.createObjectURL(file);
            setVideoSrc(localUrl);
            setOriginalBlob(file);
            onEditorStateChange(true); // Switch to editor view in parent if needed

            // Start server processing in the background
            processVideoOnServer(file);
        }
        event.target.value = null; // Reset the file input
    };

    const handleSendSegments = async () => {
        if (segments.length === 0 || !originalBlob) return;
        setIsProcessing(true); setLoopingSegment(null); setStatusMessage(`🚀 Uploading video for classification...`);
        setSegments(prev => prev.map(seg => ({ ...seg, prediction: 'pending...', confidence: null })));
        try {
            const formData = new FormData();
            formData.append('video', originalBlob);
            formData.append('segments', JSON.stringify(segments.map(({ id, start, end }) => ({ id, start, end }))));
            const response = await fetch('http://localhost:5000/classify', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Failed to start classification job.');
            const { job_id } = await response.json();
            setStatusMessage(`⏳ Classification started. Waiting for results...`);
            const es = new EventSource(`http://localhost:5000/stream/${job_id}`);
            eventSourceRef.current = es;
            es.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error("Received error from server:", data);
                } else {
                    setSegments(prev => prev.map(seg => seg.id === data.segmentId ? { ...seg, prediction: data.prediction, confidence: data.confidence } : seg));
                }
            };
            es.onerror = (err) => {
                console.error("EventSource failed:", err);
                setStatusMessage("Connection to server lost.");
                es.close();
                setIsProcessing(false);
            };
        } catch (error) {
            console.error("❌ Error sending segments:", error); setStatusMessage(`Error: ${error.message}`); setIsProcessing(false);
        }
    };

    const togglePlayPause = () => { if (!editorVideoRef.current) return; setLoopingSegment(null); editorVideoRef.current.paused ? editorVideoRef.current.play() : editorVideoRef.current.pause(); };
    const playSegmentInLoop = (segment) => { if (!editorVideoRef.current) return; setLoopingSegment(segment); editorVideoRef.current.currentTime = segment.start; if (editorVideoRef.current.paused) { editorVideoRef.current.play(); } };
    const deleteSegment = (id) => setSegments(prev => prev.filter(seg => seg.id !== id));
    
    const handleScrubStart = (e) => {
        if (!editorVideoRef.current || !timelineContainerRef.current) return;
        e.preventDefault(); setLoopingSegment(null); wasPlayingBeforeScrub.current = !editorVideoRef.current.paused; editorVideoRef.current.pause();
        const timelineRect = timelineContainerRef.current.getBoundingClientRect();
        const updateScrubTime = (clientX) => {
            const newTime = Math.max(0, Math.min(duration, ((clientX - timelineRect.left) / timelineRect.width) * duration));
            if (isFinite(newTime)) { editorVideoRef.current.currentTime = newTime; }
        };
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        updateScrubTime(clientX);
        const handleScrubMove = (moveEvent) => {
            const clientX = moveEvent.touches ? moveEvent.touches[0].clientX : moveEvent.clientX;
            updateScrubTime(clientX);
        };
        const handleScrubEnd = () => {
            if (wasPlayingBeforeScrub.current) { editorVideoRef.current.play(); }
            window.removeEventListener('mousemove', handleScrubMove); window.removeEventListener('mouseup', handleScrubEnd);
            window.removeEventListener('touchmove', handleScrubMove); window.removeEventListener('touchend', handleScrubEnd);
        };
        window.addEventListener('mousemove', handleScrubMove); window.addEventListener('mouseup', handleScrubEnd);
        window.addEventListener('touchmove', handleScrubMove); window.addEventListener('touchend', handleScrubEnd);
    };

    const handleDurationUpdate = (e) => { if(isFinite(e.currentTarget.duration)) setDuration(e.currentTarget.duration); };

    const renderEditorUI = () => (
        <div className="w-full h-full flex flex-col p-4 gap-4 bg-gray-900 text-white font-sans">
            <div className="flex-grow flex flex-col md:flex-row gap-4 min-h-0">
                 <div className="w-full md:w-2/3 flex items-center justify-center bg-black rounded-lg shadow-lg relative">
                    <video 
                        key={videoSrc} ref={editorVideoRef} src={videoSrc} className="max-w-full max-h-full" 
                        onLoadedMetadata={handleDurationUpdate} onDurationChange={handleDurationUpdate}
                        onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} 
                        crossOrigin="anonymous" 
                        onTimeUpdate={e => {
                            const video = e.currentTarget;
                            if (isFinite(video.currentTime)) { setCurrentTime(video.currentTime); }
                            if (loopingSegment && video.currentTime >= loopingSegment.end) {
                                video.currentTime = loopingSegment.start;
                            }
                        }}
                    />
                    {isProcessing && <div className="absolute inset-0 bg-black/70 flex items-center justify-center text-white text-xl font-semibold">Processing...</div>}
                 </div>
                 <div className="w-full md:w-1/3 bg-gray-800 rounded-lg p-3 flex flex-col">
                    <h3 className="text-lg font-semibold border-b border-gray-600 pb-2 mb-3">Detected Segments</h3>
                    <p className="text-sm text-gray-400 mb-3 flex-shrink-0 h-10">{statusMessage}</p>
                    <div className="flex-grow overflow-y-auto pr-2 space-y-2">
                        {segments.length > 0 ? segments.map((seg, i) => (
                            <div key={seg.id} onClick={() => !isProcessing && playSegmentInLoop(seg)} className={`p-3 rounded-md transition-all border-2 ${isProcessing ? 'cursor-default' : 'cursor-pointer'} ${loopingSegment?.id === seg.id ? 'bg-cyan-800/50 border-cyan-400' : 'bg-gray-700/80 border-transparent hover:border-gray-500'}`}>
                                <div className="flex justify-between items-start">
                                    <div><div className="font-bold text-gray-300">Segment {i + 1}</div><div className="text-xs text-cyan-400 font-mono">{formatTime(seg.start)} &rarr; {formatTime(seg.end)}</div></div>
                                    <button onClick={(e) => { e.stopPropagation(); deleteSegment(seg.id); }} disabled={isProcessing} className="text-gray-500 hover:text-red-500 disabled:opacity-25"><TrashIcon/></button>
                                </div>
                                {seg.prediction && (<div className="mt-2 pt-2 border-t border-gray-600/50"><span className="font-bold text-xl text-green-400">{seg.prediction}</span>{seg.confidence && <span className="text-sm text-gray-400 ml-2">({(seg.confidence * 100).toFixed(1)}%)</span>}</div>)}
                            </div>
                        )) : (<div className="text-gray-500 text-center mt-8 px-4">{isProcessing ? 'Analyzing...' : 'No segments found yet. Please use Auto-Detect.'}</div>)}
                    </div>
                 </div>
            </div>
            <div className="flex-shrink-0 p-3 bg-gray-800/90 rounded-lg flex items-center gap-4 flex-wrap">
                <button onClick={togglePlayPause} className="text-white hover:text-cyan-400">{isPlaying ? <PauseIcon/> : <PlayIcon/>}</button>
                <div className="font-mono text-sm text-gray-300">{formatTime(currentTime)}</div>
                <div ref={timelineContainerRef} onMouseDown={handleScrubStart} onTouchStart={handleScrubStart} className="flex-grow h-4 flex items-center group relative cursor-pointer pr-2">
                    <div className="w-full h-2 bg-gray-600 rounded-full relative">
                        {segments.map(seg => (<div key={`timeline-${seg.id}`} className={`absolute h-full opacity-75 rounded-full ${loopingSegment?.id === seg.id ? 'bg-cyan-400' : 'bg-green-500'}`} style={{ left: `${duration > 0 ? (seg.start / duration) * 100 : 0}%`, width: `${duration > 0 ? ((seg.end - seg.start) / duration) * 100 : 0}%` }}></div>))}
                        <div className="absolute h-full" style={{ left: `${duration > 0 ? (currentTime / duration) * 100 : 0}%`, pointerEvents: 'none' }}><div className="w-4 h-4 bg-white rounded-full -translate-x-1/2 -translate-y-1/2 top-1/2 absolute"></div></div>
                    </div>
                </div>
                <div className="font-mono text-sm text-gray-500">{formatTime(duration)}</div>
                <div className="w-px h-6 bg-gray-700"></div>
                <button onClick={() => processVideoOnServer(originalBlob)} disabled={!originalBlob || isProcessing} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-semibold px-4 py-1.5 rounded-md"><AutoIcon /> {isProcessing ? 'Analyzing...' : 'Auto-Detect'}</button>
                <button onClick={handleSendSegments} disabled={isProcessing || segments.length === 0} className="flex items-center gap-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white font-semibold px-4 py-1.5 rounded-md"><SendIcon /> Send to Classifier</button>
            </div>
        </div>
    );
    
    const renderWelcomeScreen = () => (
        <div className="w-full h-full flex flex-col items-center justify-center p-4 gap-8 bg-gray-900 text-white">
            <h2 className="text-2xl font-bold text-gray-300">Automated Segment Editor</h2>
            <div className="text-center">
                <p className="text-gray-400">You are not currently recording.</p>
                <p className="text-gray-400 mt-1">Please start a recording from the main controls to begin.</p>
            </div>
            <div className="flex items-center gap-4"><hr className="w-24 border-gray-700" /><span className="text-gray-500 font-semibold">OR</span><hr className="w-24 border-gray-700" /></div>
            <button onClick={() => fileInputRef.current.click()} className="flex items-center gap-3 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-lg hover:bg-blue-500 transition-colors"><UploadIcon />Upload a Video File</button>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4,video/webm" className="hidden" />
        </div>
    );

    return videoSrc ? renderEditorUI() : renderWelcomeScreen();
});

export default AutomatedEditor;