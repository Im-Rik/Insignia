// hooks/useMediaPipe.js
import { useEffect, useRef } from 'react';
import { Holistic } from '@mediapipe/holistic';

const PROCESS_EVERY_N_FRAMES = 3;

const useMediaPipe = (videoRef, isActive, onResults) => {
  const holisticRef = useRef(null);
  const frameCountRef = useRef(0);
  const animationIdRef = useRef(null);
  const isProcessingRef = useRef(false);

  useEffect(() => {
    if (!isActive || !videoRef.current) {
      return; // Exit early if not active or ref is not set
    }

    const videoElement = videoRef.current;

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults((results) => {
      isProcessingRef.current = false;
      onResults(results);
    });

    holisticRef.current = holistic;

    const processFrame = async () => {
      if (!holisticRef.current || !videoElement) return;

      frameCountRef.current++;
      
      if (frameCountRef.current % PROCESS_EVERY_N_FRAMES === 0 && !isProcessingRef.current) {
        isProcessingRef.current = true;
        try {
          await holisticRef.current.send({ image: videoElement });
        } catch (error) {
          console.error('MediaPipe processing error:', error);
          isProcessingRef.current = false;
        }
      }

      animationIdRef.current = requestAnimationFrame(processFrame);
    };

    // More robust way to start processing
    const startProcessing = () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      animationIdRef.current = requestAnimationFrame(processFrame);
    };

    // Listen for the 'canplay' event to safely start processing
    videoElement.addEventListener('canplay', startProcessing);

    // Cleanup function
    return () => {
      videoElement.removeEventListener('canplay', startProcessing);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (holisticRef.current) {
        holisticRef.current.close();
      }
    };
  }, [isActive, onResults, videoRef]); // Dependencies remain the same
};

export default useMediaPipe;