import React, { useState, useRef, useEffect } from 'react';
import VideoDisplay from './components/VideoDisplay';
import SubtitleDisplay from './components/SubtitleDisplay';
import Controls from './components/Controls';

function App() {
  const [subtitles, setSubtitles] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const [mockSubtitles, setMockSubtitles] = useState([
    "Hello there!",
    "Welcome to this live caption demo.",
    "This interface is now responsive.",
    "Video and captions stack on mobile.",
    "System is processing audio input...",
    "Transcribing in near real-time.",
    "Modern UI with React & Tailwind.",
    "Awaiting further speech...",
    "Components adapt to screen size.",
    "Layout is constrained on very large screens." // New mock text
  ]);
  const [currentSubtitleIndex, setCurrentSubtitleIndex] = useState(0);

  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 }, // Aim for HD
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.play().catch(err => {
          console.error("Error playing video:", err);
        });
      }
      setIsRecording(true);
      setSubtitles('');
      setCurrentSubtitleIndex(0);
      console.log("Video stream started");
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access the camera. Please ensure permission is granted and no other application is using it.");
    }
  };

  const stopVideo = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsRecording(false);
    console.log("Video stream stopped");
  };

  useEffect(() => {
    let intervalId = null;
    if (isRecording) {
      intervalId = setInterval(() => {
        setSubtitles(prev =>
          `${prev}\n${mockSubtitles[currentSubtitleIndex % mockSubtitles.length]}`.split('\n').slice(-10).join('\n')
        );
        setCurrentSubtitleIndex(prevIndex => prevIndex + 1);
      }, 2200);
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRecording, mockSubtitles, currentSubtitleIndex]);

  return (
    // Outermost container for the entire app viewport
    <div className="flex flex-col items-center h-screen bg-gray-900 text-gray-100 font-sans">
      
      {/* Centering and Max-Width Container */}
      {/* This container will take full width up to a certain max-width, then center itself. */}
      {/* It also handles the vertical flex layout and padding for its children. */}
      <div className="flex flex-col w-full max-w-screen-2xl h-full p-2 sm:p-3 md:p-4 lg:p-6"> {/* Added max-w-screen-2xl, adjust as needed (e.g., screen-xl) */}

        {/* Main Content Area: Video and Subtitles */}
        <div className="flex flex-col md:flex-row flex-1 overflow-hidden rounded-lg md:rounded-xl shadow-xl md:shadow-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700/50">
          
          {/* Video Display Area */}
          <div className="flex-grow flex items-center justify-center p-1.5 sm:p-2 md:p-3 lg:p-4 bg-black/50 relative md:rounded-l-xl">
             <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
          </div>

          {/* Subtitle Display */}
          <div className="w-full md:w-72 lg:w-80 xl:w-96 md:flex-shrink-0 bg-gray-800/80 md:border-l border-gray-700/60 flex flex-col mt-2 md:mt-0 md:rounded-r-xl overflow-hidden">
            <div className="h-64 sm:h-72 md:h-full"> 
              <SubtitleDisplay subtitles={subtitles} isRecording={isRecording} />
            </div>
          </div>
        </div>

        {/* Controls - footer */}
        <div className="mt-2 sm:mt-3 md:mt-4 p-2.5 sm:p-3 md:p-3.5 bg-gray-800/70 backdrop-blur-md rounded-lg md:rounded-xl shadow-lg border border-gray-700/50">
          <Controls isRecording={isRecording} onStart={startVideo} onStop={stopVideo} />
        </div>
      </div> {/* End of Centering and Max-Width Container */}
    </div> // End of Outermost container
  );
}

export default App;