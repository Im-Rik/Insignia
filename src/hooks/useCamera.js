// hooks/useCamera.js
import { useRef } from 'react';

const useCamera = () => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const startVideo = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    mediaStreamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }
  };

  const stopVideo = () => {
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  return { videoRef, startVideo, stopVideo };
};

export default useCamera;
