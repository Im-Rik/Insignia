// src/hooks/useVideoStream.js - More Robust Version
import { useRef, useState, useCallback } from 'react';

const useVideoStream = () => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  const startVideo = useCallback(async () => {
    console.log('[useVideoStream] Attempting to start video...');
    if (isRecording || !videoRef.current) {
      console.log('[useVideoStream] Aborting start: already recording or videoRef is null.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
      console.log('[useVideoStream] ✅ Camera stream acquired.');

      mediaStreamRef.current = stream;
      const videoElement = videoRef.current;
      videoElement.srcObject = stream;
      
      // These attributes are very important for autoplay on all browsers
      videoElement.muted = true;
      videoElement.playsInline = true;

      console.log('[useVideoStream] Attaching stream and calling video.play()...');
      await videoElement.play();
      console.log('[useVideoStream] ✅ Video playback successfully started.');

      setIsRecording(true);
    } catch (err) {
      console.error("❌ CRITICAL ERROR in useVideoStream:", err);
      alert(`Camera access failed: ${err.message}. Please check browser permissions and refresh the page.`);
      setIsRecording(false);
    }
  }, [isRecording]); // Added isRecording to dependency array

  const stopVideo = useCallback(() => {
    console.log('[useVideoStream] Stopping video...');
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }
    mediaStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsRecording(false);
  }, []);

  return { videoRef, isRecording, startVideo, stopVideo };
};

export default useVideoStream;