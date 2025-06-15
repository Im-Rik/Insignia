//components/Controls.jsx

import React, { useState, useEffect, useRef, createRef } from 'react';

// --- Self-Contained SVG Icons ---
const IconUser = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
);
const IconUpload = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
);
const IconDev = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
);
const IconDoctor = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
    </svg>
);
const IconStart = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" /><path d="M14.553 5.106A1 1 0 0116 6v8a1 1 0 01-1.447.894l-3-2A1 1 0 0111 12V8a1 1 0 01.553-.894l3-2z" /></svg>
);
const IconStop = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1zm4 0a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
);
const IconRecordNew = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M15.322 4.688a2.25 2.25 0 013.182 3.182l-8.66 8.66-3.536.252a.75.75 0 01-.815-.815l.252-3.536 8.66-8.66zM13.5 7.5a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5a.75.75 0 01-.75-.75z" clipRule="evenodd" />
    </svg>
);

// --- The Main Controls Component ---
function Controls({
    isRecording,
    onStart,
    onStop,
    mode,
    onModeChange,
    uploadedVideo,
    onClearUpload,
    isEditorActive,
    onResetEditor,
    hideStartStop = false,  // Add the hideStartStop prop with default value
    hideStartStopForMode = false // Add this prop for mode-specific hiding
}) {

    const modes = [
        { value: 'user-mode-1', label: 'Live', icon: <IconUser /> },
        { value: 'user-mode-2', label: 'Upload', icon: <IconUpload /> },
        { value: 'developer-mode-1', label: 'Dev', icon: <IconDev /> },
        { value: 'developer-mode-2', label: 'Dev 2', icon: <IconDev /> },
        { value: 'developer-mode-3', label: 'Dev 3', icon: <IconDev /> },
        { value: 'doctor-mode', label: 'Doctor', icon: <IconDoctor /> },
    ];

    // Create an array of refs, one for each button (now 6 buttons)
    const buttonRefs = useRef(modes.map(() => createRef()));
    const [sliderStyle, setSliderStyle] = useState({});

    useEffect(() => {
        const activeIndex = modes.findIndex((m) => m.value === mode);

        // Get the DOM node from our reliable ref array
        const activeTabNode = buttonRefs.current[activeIndex]?.current;

        if (activeTabNode) {
            setSliderStyle({
                left: `${activeTabNode.offsetLeft}px`,
                width: `${activeTabNode.offsetWidth}px`,
            });
        }
    }, [mode]); // Dependency array is now just 'mode'


    return (
        <div className="flex justify-between items-center gap-4 w-full px-4 py-2">

            {/* --- Animated Mode Selector --- */}
            <div className="relative flex items-center p-1 bg-gray-800/70 border border-gray-700/60 rounded-lg">
                {/* The animated sliding background */}
                <div
                    className={`absolute shadow rounded-md h-[calc(100%-0.5rem)] transition-all duration-300 ease-in-out ${
                        mode === 'doctor-mode'
                            ? 'bg-gradient-to-r from-red-600 to-pink-600'
                            : 'bg-cyan-600'
                    }`}
                    style={sliderStyle}
                />

                {modes.map((m, index) => (
                    <button
                        key={m.value}
                        // Assign the correct ref from the array to each button
                        ref={buttonRefs.current[index]}
                        onClick={() => onModeChange(m.value)}
                        className={`
                            relative z-10 flex items-center justify-center gap-2 px-4 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors duration-300
                            focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400
                            ${mode === m.value
                                ? 'text-white' // Active text is white
                                : m.restricted
                                    ? 'text-red-300 hover:text-red-200' // Restricted button styling
                                    : 'text-gray-400 hover:text-gray-200' // Regular inactive text
                            }
                            ${m.restricted ? 'relative' : ''}
                        `}
                        title={m.restricted ? "🩺 Medical professionals only" : undefined}
                    >
                        {m.restricted && (
                            <span className="absolute -top-1 -right-1 text-xs">🔒</span>
                        )}
                        {m.icon}
                        <span>{m.label}</span>
                    </button>
                ))}
            </div>

            {/* --- Action Buttons --- */}
            <div className="flex-shrink-0 flex items-center gap-3">
    {(() => {
        // Case 1: Handle the "Upload" mode
        if (mode === 'user-mode-2') {
            return uploadedVideo ? (
                <button
                    onClick={onClearUpload}
                    className="px-4 py-2 bg-gray-700 text-gray-300 font-medium rounded-lg border border-gray-600 hover:bg-gray-600 hover:text-white transition duration-200 text-sm"
                >
                    Clear Video
                </button>
            ) : null;
        }

        // Case 2: Handle "Doctor" mode - no recording controls needed
        if (mode === 'doctor-mode') {
            return null; // Doctor mode doesn't need recording controls
        }

        // Case 3: Handle our special editor modes
        if ((mode === 'developer-mode-2' || mode === 'developer-mode-3') && isEditorActive) {
            // When the editor is active, show the "Record New" button
            return (
                <button
                    onClick={onResetEditor}
                    className="flex items-center gap-2 px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg shadow-md hover:bg-cyan-700 transition duration-200"
                >
                    <IconRecordNew />
                    <span>Record New</span>
                </button>
            );
        }

        // Case 4: For all other situations (Live, Dev 1, and the initial Dev 2/3 screens)
        // Check if we should hide the Start/Stop button
        if (hideStartStop || hideStartStopForMode) {
            return null; // Don't show any button if hideStartStop is true
        }

        // show the normal Start/Stop button.
        return !isRecording ? (
            <button
                onClick={onStart}
                className="flex items-center gap-2 px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg shadow-md hover:bg-cyan-700 transition duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-70 text-sm"
            >
                <IconStart />
                <span>Start Video</span>
            </button>
        ) : (
            <button
                onClick={onStop}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 transition duration-200 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-70 text-sm"
            >
                <IconStop />
                <span>Stop Video</span>
            </button>
                );
            })()}
            </div>
        </div>
    );
}

export default Controls;