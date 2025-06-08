//hooks.useVideoStream.js

import { useRef, useState } from 'react';

const useVideoStream = () => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Camera access failed. Check permissions.");
    }
  };

  const stopVideo = () => {
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsRecording(false);
  };

  return { videoRef, isRecording, startVideo, stopVideo };
};

export default useVideoStream;
