import React, { useState, useRef, useEffect } from 'react';

// Import the new components
import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';


// Main App Component
function App() {
  // State to hold the subtitles received from the backend
  const [subtitles, setSubtitles] = useState('');
  // State to track if recording is active
  const [isRecording, setIsRecording] = useState(false);
  // Ref to access the video element
  const videoRef = useRef(null);
  // Ref to store the media stream
  const mediaStreamRef = useRef(null);
  // State to hold mock subtitles for demonstration
  const [mockSubtitles, setMockSubtitles] = useState([
    "Hello!",
    "This is a demonstration.",
    "Captions will appear here.",
    "The video feed is above.",
    "System is processing...",
    "Almost live transcription.",
    "React and Component Structure.", // Updated text
    "Waiting for more input...",
  ]);
  const [currentSubtitleIndex, setCurrentSubtitleIndex] = useState(0);

  // Function to start video capture
  const startVideo = async () => {
    try {
      // Request video stream without audio
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 }, // Request HD resolution if possible
          height: { ideal: 720 }
        },
        audio: false // Explicitly disable audio
      });
      mediaStreamRef.current = stream; // Store the stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Mute the video element to be sure no audio plays
        videoRef.current.muted = true;
        videoRef.current.play().catch(err => {
          console.error("Error playing video:", err);
          // Handle potential autoplay restrictions
        });
      }
      setIsRecording(true);
      setSubtitles(''); // Clear previous subtitles
      setCurrentSubtitleIndex(0); // Reset mock subtitle index
      console.log("Video stream started");
      // In a real app, you'd start sending video data to the backend here
    } catch (err) {
      console.error("Error accessing camera:", err);
      // Handle errors (e.g., user denied permission)
      alert("Could not access the camera. Please ensure permission is granted and no other application is using it.");
    }
  };

  // Function to stop video capture
  const stopVideo = () => {
    if (mediaStreamRef.current) {
      // Stop all tracks in the stream
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null; // Clear the stored stream
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null; // Remove stream from video element
    }
    setIsRecording(false);
    console.log("Video stream stopped");
    // In a real app, you'd stop sending video data here
  };

  // Simulate receiving subtitles when recording
  useEffect(() => {
    let intervalId = null;
    if (isRecording) {
      // Update subtitles every 3 seconds for demonstration
      intervalId = setInterval(() => {
        setSubtitles(prev =>
          // Append new mock subtitle, keeping the last few lines
          `${prev}\n${mockSubtitles[currentSubtitleIndex % mockSubtitles.length]}`.split('\n').slice(-5).join('\n')
        );
        setCurrentSubtitleIndex(prevIndex => prevIndex + 1);
      }, 3000);
    } else {
      // Optional: Clear subtitles immediately or fade them out
            // setSubtitles(''); // Clear subtitles when not recording
    }

    // Cleanup interval on component unmount or when recording stops
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRecording, mockSubtitles, currentSubtitleIndex]); // Rerun effect if recording status or subtitles change

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-800 to-gray-900 text-gray-100 p-6 font-sans">
      <h1 className="text-3xl md:text-4xl font-bold mb-8 text-teal-400 tracking-tight">
        Insignia
      </h1>

      
      <div className="w-full max-w-xl md:max-w-2xl bg-gray-800 bg-opacity-70 backdrop-filter backdrop-blur-lg rounded-lg border border-gray-700 shadow-xl overflow-hidden mb-8">
        {/* Video and Subtitle displays are now separate components */}
        <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
        <SubtitleDisplay subtitles={subtitles} isRecording={isRecording} />
      </div>


      <Controls isRecording={isRecording} onStart={startVideo} onStop={stopVideo} />

    </div>
  );
}

export default App;