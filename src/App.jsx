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
    "Hello there!", // Freshened mock text
    "Welcome to this live caption demo.",
    "Notice the updated styling.",
    "Video feed is on the left.",
    "System is processing audio input...",
    "Transcribing in near real-time.",
    "Modern UI with React & Tailwind.",
    "Awaiting further speech...",
    "Components are now more rounded.",
    "Spacing has been adjusted."
  ]);
  const [currentSubtitleIndex, setCurrentSubtitleIndex] = useState(0);

  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 } // Request a smoother frame rate
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
      setSubtitles(''); // Clear previous subtitles
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
      }, 2200); // Slightly adjusted interval
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRecording, mockSubtitles, currentSubtitleIndex]);

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100 font-sans p-3 sm:p-4 md:p-6"> {/* Added padding to the whole app */}
      
      {/* Main Content Area: Video and Subtitles */}
      <div className="flex flex-1 overflow-hidden rounded-xl shadow-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700/50"> {/* Rounded corners, shadow, subtle background */}
        
        {/* Video Display Area */}
        <div className="flex-grow flex items-center justify-center p-2 md:p-3 lg:p-4 bg-black/50 relative"> {/* Padding inside, darker bg */}
          {/* VideoDisplay component will be here, its own styling will apply for rounded video */}
           <VideoDisplay videoRef={videoRef} isRecording={isRecording} />
        </div>

        {/* Subtitle Display - sidebar on the right */}
        {/* Increased width slightly, added padding inside the subtitle box */}
        <div className="w-80 sm:w-96 md:w-[400px] flex-shrink-0 bg-gray-800/80 border-l border-gray-700/60 flex flex-col">
          <SubtitleDisplay subtitles={subtitles} isRecording={isRecording} />
        </div>
      </div>

      {/* Controls - footer */}
      {/* Added margin-top for spacing, and padding inside the controls bar */}
      <div className="mt-3 sm:mt-4 md:mt-6 p-3 sm:p-4 bg-gray-800/70 backdrop-blur-md rounded-xl shadow-lg border border-gray-700/50">
        <Controls isRecording={isRecording} onStart={startVideo} onStop={stopVideo} />
      </div>
    </div>
  );
}

export default App;