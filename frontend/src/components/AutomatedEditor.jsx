import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';

// --- UI Icons & Helper Components (No Changes) ---
const PendingIndicator = () => (
    <div className="flex items-center space-x-1">
        <span className="text-sm text-yellow-400">pending</span>
        <div className="flex items-center justify-center space-x-1">
            <span className="h-1.5 w-1.5 bg-yellow-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="h-1.5 w-1.5 bg-yellow-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="h-1.5 w-1.5 bg-yellow-400 rounded-full animate-bounce"></span>
        </div>
    </div>
);

const AnalysisLoader = ({ status }) => (
    <div className="flex flex-col items-center justify-center h-full p-4 text-center text-white">
        <svg className="animate-spin h-8 w-8 text-cyan-400 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="text-lg font-semibold">{status}</p>
    </div>
);

const PlayIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M7 4v16l13-8L7 4z"></path></svg>;
const PauseIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>;
const TrashIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"></path></svg>;
const SendIcon = () => <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>;
const AutoIcon = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M11.832 2.25A5.25 5.25 0 006.75 7.5v.032l1.18-.393A3.75 3.75 0 0112 11.25v2.538l-2.02-.674a.75.75 0 00-.83.056l-.585.585a.75.75 0 000 1.06l4.242 4.243a.75.75 0 001.06 0l4.243-4.243a.75.75 0 000-1.06l-.585-.585a.75.75 0 00-.83-.056l-2.02.674v-2.538a3.75 3.75 0 014.07-3.718l1.18.393V7.5a5.25 5.25 0 00-5.25-5.25h-.008z" clipRule="evenodd"></path></svg>;
const UploadIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>;
const RecordIcon = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15.91 11.672a.375.375 0 0 1 0 .656l-5.603 3.113a.375.375 0 0 1-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112Z" /></svg>;
const SparklesIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10.868 2.884c.321.772.115 1.68-.445 2.239l-1.528 1.528A1.5 1.5 0 016.5 8.118l-1.528-1.528c-.56-.56-1.467-.766-2.239-.445A1.5 1.5 0 012 4.632V3a1 1 0 011-1h1.632c.5 0 .963.19 1.314.555c.445.445.64.99.868 1.5.228.51.99.868 1.5.868s1.272-.358 1.5-.868c.228-.51.423-1.055.868-1.5.351-.364.814-.555 1.314-.555H17a1 1 0 011 1v1.632a1.5 1.5 0 01-.632 1.239z" clipRule="evenodd" /><path fillRule="evenodd" d="M10 12a1 1 0 011 1v5a1 1 0 11-2 0v-5a1 1 0 011-1z" clipRule="evenodd" /></svg>

const formatTime = (timeInSeconds) => {
    if (!isFinite(timeInSeconds) || timeInSeconds < 0) return '00:00.0';
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(4, '0')}`;
};

// --- CHANGE 1: Accept `selectedSymptoms` as a prop ---
const AutomatedEditor = forwardRef(({ videoRef, isRecording, onEditorStateChange, selectedSymptoms }, ref) => {
    const [editorMode, setEditorMode] = useState('welcome');
    const [videoSrc, setVideoSrc] = useState(null);
    const [originalBlob, setOriginalBlob] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [segments, setSegments] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isSegmenting, setIsSegmenting] = useState(false);
    const [statusMessage, setStatusMessage] = useState('Record or upload a video to begin.');
    const [finalSentence, setFinalSentence] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const mediaRecorderRef = useRef(null);
    const editorVideoRef = useRef(null);
    const timelineContainerRef = useRef(null);
    const wasPlayingBeforeScrub = useRef(false);
    const [loopingSegment, setLoopingSegment] = useState(null);
    const fileInputRef = useRef(null);
    const eventSourceRef = useRef(null);

    const hasClassifiedSegments = segments.some(seg => seg.prediction && seg.prediction !== 'pending...');

    const processVideoOnServer = async (videoBlob) => {
        if (!videoBlob) { setStatusMessage("Error: No video data available."); return; }
        setIsProcessing(true); setIsSegmenting(true); setLoopingSegment(null); setSegments([]); setFinalSentence('');
        setStatusMessage("Uploading and analyzing for segments...");
        try {
            const formData = new FormData();
            formData.append('video', videoBlob, videoBlob.name);
            const response = await fetch('http://localhost:8000/segment', { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status ${response.status}`);
            }
            const { segments: detectedSegments } = await response.json();
            if (detectedSegments.length === 0) {
                setStatusMessage("✅ Analysis complete. No distinct gestures were detected.");
            } else {
                setSegments(detectedSegments.map((seg, i) => ({ id: Date.now() + i, start: seg.start, end: seg.end, prediction: null, confidence: null })));
                setStatusMessage(`✅ Success! Review segments and send for classification.`);
            }
        } catch (error) {
            console.error("❌ Error during segmentation:", error);
            setStatusMessage(`Error: ${error.message}`);
        } finally {
            setIsProcessing(false); setIsSegmenting(false);
        }
    };
    
    const handleSendSegments = async () => {
        if (segments.length === 0 || !originalBlob) return;
        setIsProcessing(true); setLoopingSegment(null); setFinalSentence('');
        setStatusMessage(`🚀 Uploading video for classification...`);
        setSegments(prev => prev.map(seg => ({ ...seg, prediction: 'pending...', confidence: null })));
        try {
            const formData = new FormData();
            formData.append('video', originalBlob, originalBlob.name);
            formData.append('segments', JSON.stringify(segments.map(({ id, start, end }) => ({ id, start, end }))));
            const response = await fetch('http://localhost:5020/classify', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Failed to start classification job.');
            const { job_id } = await response.json();
            setStatusMessage(`⏳ Classification started. Waiting for results...`);
            
            const es = new EventSource(`http://localhost:5020/stream/${job_id}`);
            eventSourceRef.current = es;

            es.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.segmentId) {
                    setSegments(prev => prev.map(seg => seg.id === data.segmentId ? { ...seg, prediction: data.prediction, confidence: data.confidence } : seg));
                }
            };

            const endStream = () => {
                setIsProcessing(false);
                setStatusMessage("Classification finished. Ready to generate sentence.");
                es.close();
            };

            es.onerror = endStream;
            es.addEventListener('complete', endStream);

        } catch (error) {
            console.error("❌ Error sending segments:", error);
            setStatusMessage(`Error: ${error.message}`);
            setSegments(prev => prev.map(seg => ({ ...seg, prediction: null, confidence: null })));
            setIsProcessing(false);
        }
    };

    // --- CHANGE 2: Use the `selectedSymptoms` prop to build the context ---
    const handleGenerateSentence = async () => {
        const glosses = segments
            .map(seg => seg.prediction)
            .filter(p => p && p !== 'pending...');

        if (glosses.length === 0) {
            setFinalSentence("Error: No classified words available to generate a sentence.");
            return;
        }

        // Create context sentences from the passed-in prop
        const context_sentences = selectedSymptoms.map(symptom => symptom.name);

        setIsGenerating(true);
        setFinalSentence('');
        
        try {
            const response = await fetch('http://localhost:5020/generate_sentence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ glosses, context_sentences }) // Send both glosses and context
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate sentence.');
            }
            const { generated_sentence } = await response.json();
            setFinalSentence(generated_sentence);
            setStatusMessage("✅ All steps complete!");
        } catch (error) {
            console.error("❌ Error generating sentence:", error);
            setFinalSentence(`Error: ${error.message}`);
            setStatusMessage("Error during final sentence generation.");
        } finally {
            setIsGenerating(false);
        }
    };

    const handleResetEditor = () => {
        if (videoSrc && videoSrc.startsWith('blob:')) URL.revokeObjectURL(videoSrc);
        setVideoSrc(null); setOriginalBlob(null); setIsPlaying(false);
        setDuration(0); setCurrentTime(0); setSegments([]);
        setIsProcessing(false); setIsSegmenting(false); setLoopingSegment(null);
        onEditorStateChange(false); setStatusMessage('Record or upload a video to begin.');
        setEditorMode('welcome');
        if (eventSourceRef.current) eventSourceRef.current.close();
        setFinalSentence('');
        setIsGenerating(false);
    };

    useEffect(() => {
        const stream = videoRef.current?.srcObject;
        if (isRecording && stream) {
            handleResetEditor();
            setEditorMode('recording');
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
            const chunks = [];
            mediaRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data) };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const recordedFile = new File([blob], "recorded_video.webm", { type: "video/webm", lastModified: Date.now() });
                const url = URL.createObjectURL(recordedFile);
                setOriginalBlob(recordedFile); setVideoSrc(url);
                onEditorStateChange(true);
                setStatusMessage('Recording finished. Click "Auto-Detect" to process.');
                setEditorMode('editing');
            };
            mediaRecorderRef.current.start();
        } else if (!isRecording && mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, [isRecording, videoRef, onEditorStateChange]);
    
    useImperativeHandle(ref, () => ({ reset: handleResetEditor }));
    
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            handleResetEditor();
            const localUrl = URL.createObjectURL(file);
            setVideoSrc(localUrl); setOriginalBlob(file);
            onEditorStateChange(true); setEditorMode('editing');
            processVideoOnServer(file);
        }
        event.target.value = null;
    };
    
    const togglePlayPause = () => { if (!editorVideoRef.current) return; setLoopingSegment(null); editorVideoRef.current.paused ? editorVideoRef.current.play() : editorVideoRef.current.pause(); };
    const playSegmentInLoop = (segment) => { if (!editorVideoRef.current) return; setLoopingSegment(segment); editorVideoRef.current.currentTime = segment.start; if (editorVideoRef.current.paused) { editorVideoRef.current.play(); } };
    const deleteSegment = (id) => { setSegments(prev => prev.filter(seg => seg.id !== id)); };
    const handleScrubStart = (e) => { if (!editorVideoRef.current || !timelineContainerRef.current) return; e.preventDefault(); setLoopingSegment(null); wasPlayingBeforeScrub.current = !editorVideoRef.current.paused; editorVideoRef.current.pause(); const timelineRect = timelineContainerRef.current.getBoundingClientRect(); const updateScrubTime = (clientX) => { const newTime = Math.max(0, Math.min(duration, ((clientX - timelineRect.left) / timelineRect.width) * duration)); if (isFinite(newTime)) { editorVideoRef.current.currentTime = newTime; } }; const clientX = e.touches ? e.touches[0].clientX : e.clientX; updateScrubTime(clientX); const handleScrubMove = (moveEvent) => { const clientX = moveEvent.touches ? moveEvent.touches[0].clientX : moveEvent.clientX; updateScrubTime(clientX); }; const handleScrubEnd = () => { if (wasPlayingBeforeScrub.current) { editorVideoRef.current.play(); } window.removeEventListener('mousemove', handleScrubMove); window.removeEventListener('mouseup', handleScrubEnd); window.removeEventListener('touchmove', handleScrubMove); window.removeEventListener('touchend', handleScrubEnd); }; window.addEventListener('mousemove', handleScrubMove); window.addEventListener('mouseup', handleScrubEnd); window.addEventListener('touchmove', handleScrubMove); window.addEventListener('touchend', handleScrubEnd); };
    const handleDurationUpdate = (e) => { if (isFinite(e.currentTarget.duration)) { setDuration(e.currentTarget.duration); } };
    
    const renderEditorUI = () => (
        <div className="w-full h-full flex flex-col p-4 gap-4 bg-gray-900 text-white font-sans">
            <div className="flex-grow flex flex-col md:flex-row gap-4 min-h-0">
                <div className="w-full md:w-2/3 flex items-center justify-center bg-black rounded-lg shadow-lg relative">
                    <video key={videoSrc} ref={editorVideoRef} src={videoSrc} className="max-w-full max-h-full" onLoadedMetadata={handleDurationUpdate} onDurationChange={handleDurationUpdate} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} crossOrigin="anonymous" onTimeUpdate={e => { const video = e.currentTarget; if (isFinite(video.currentTime)) setCurrentTime(video.currentTime); if (loopingSegment && video.currentTime >= loopingSegment.end) video.currentTime = loopingSegment.start; }} />
                </div>
                <div className="w-full md:w-1/3 bg-gray-800 rounded-lg p-3 flex flex-col">
                    <h3 className="text-lg font-semibold border-b border-gray-600 pb-2 mb-3">Analysis Results</h3>
                    <div className="flex-grow min-h-0 overflow-y-auto pr-2">
                        {isSegmenting ? <AnalysisLoader status={statusMessage} /> : (
                            <>
                                <p className="text-sm text-gray-400 mb-3 flex-shrink-0 h-10">{statusMessage}</p>
                                <div className="space-y-2">
                                    {segments.length > 0 ? segments.map((seg, i) => (
                                        <div key={seg.id} onClick={() => playSegmentInLoop(seg)} className="p-3 rounded-md transition-all border-2 cursor-pointer bg-gray-700/80 border-transparent hover:border-gray-500">
                                            <div className="flex justify-between items-start">
                                                <div>
                                                    <div className="font-bold text-gray-300">Segment {i + 1}</div>
                                                    <div className="text-xs text-cyan-400 font-mono">{formatTime(seg.start)} &rarr; {formatTime(seg.end)}</div>
                                                </div>
                                                <button onClick={(e) => { e.stopPropagation(); deleteSegment(seg.id); }} disabled={isProcessing} className="text-gray-500 hover:text-red-500 disabled:opacity-25"><TrashIcon/></button>
                                            </div>
                                            {seg.prediction && (
                                                <div className="mt-2 pt-2 border-t border-gray-600/50">
                                                    {seg.prediction === 'pending...' ? <PendingIndicator /> : (
                                                        <>
                                                            <span className="font-bold text-xl text-green-400">{seg.prediction}</span>
                                                            {seg.confidence && <span className="text-sm text-gray-400 ml-2">({(seg.confidence * 100).toFixed(1)}%)</span>}
                                                        </>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    )) : ( <div className="text-gray-500 text-center mt-8 px-4">No segments found. Use 'Auto-Detect' to find them.</div> )}
                                </div>
                                
                                {segments.length > 0 && !isProcessing && (
                                    <div className="mt-4 p-3 rounded-md bg-gray-700/80 border-2 border-purple-500/50">
                                        <div className="flex justify-between items-center">
                                            <h4 className="font-bold text-gray-300 flex items-center gap-2"><SparklesIcon/> Final Sentence</h4>
                                        </div>
                                        <div className="mt-2 pt-2 border-t border-gray-600/50">
                                            {finalSentence ? (
                                                <p className={`text-base ${finalSentence.startsWith('Error:') ? 'text-red-400' : 'text-gray-200'}`}>{finalSentence}</p>
                                            ) : (
                                                <button 
                                                    onClick={handleGenerateSentence} 
                                                    disabled={isGenerating || !hasClassifiedSegments} 
                                                    className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold px-4 py-2 rounded-md"
                                                >
                                                    {isGenerating ? 'Generating...' : 'Generate Sentence'}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
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
                <button onClick={() => processVideoOnServer(originalBlob)} disabled={!originalBlob || isProcessing || isSegmenting} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-semibold px-4 py-1.5 rounded-md"><AutoIcon /> {isSegmenting ? 'Analyzing...' : 'Auto-Detect'}</button>
                <button onClick={handleSendSegments} disabled={isProcessing || segments.length === 0} className="flex items-center gap-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white font-semibold px-4 py-1.5 rounded-md"><SendIcon /> Send to Classifier</button>
            </div>
        </div>
    );
    
    const renderWelcomeScreen = () => (
        <div className="w-full h-full flex flex-col items-center justify-center p-4 gap-8 bg-gray-900 text-white">
            <h2 className="text-2xl font-bold text-gray-300">Automated Segment Editor</h2>
            <div className="text-center"><p className="text-gray-400 flex items-center gap-2"><RecordIcon/> Use the main controls to start a new recording.</p></div>
            <div className="flex items-center gap-4"><hr className="w-24 border-gray-700" /><span className="text-gray-500 font-semibold">OR</span><hr className="w-24 border-gray-700" /></div>
            <button onClick={() => fileInputRef.current.click()} className="flex items-center gap-3 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-lg hover:bg-blue-500 transition-colors"><UploadIcon /> Upload a Video File</button>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4,video/webm" className="hidden" />
        </div>
    );
    
    const renderContent = () => {
        if (editorMode === 'editing') { return renderEditorUI(); }
        return (
            <div className="w-full h-full relative bg-gray-900">
                <div className={`absolute inset-0 w-full h-full flex items-center justify-center transition-opacity duration-300 ${editorMode === 'recording' ? 'opacity-100 z-10' : 'opacity-0 -z-10'}`}><video ref={videoRef} className="block w-full h-full object-contain -scale-x-100" playsInline muted autoPlay /></div>
                <div className={`absolute inset-0 w-full h-full transition-opacity duration-300 ${editorMode === 'welcome' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>{renderWelcomeScreen()}</div>
            </div>
        );
    };

    return <div className="w-full h-full">{renderContent()}</div>;
});

export default AutomatedEditor;