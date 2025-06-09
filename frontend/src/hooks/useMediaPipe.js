// src/hooks/useMediaPipe.js - FINAL PRODUCTION-READY VERSION
import { useEffect, useRef } from 'react';
import { Holistic } from '@mediapipe/holistic';

const useMediaPipe = (videoRef, isActive, onResults) => {
  const holisticRef = useRef(null);
  const animationFrameIdRef = useRef(null);
  
  useEffect(() => {
    // This function runs the main processing loop
    const processFrameLoop = async () => {
      // Safety check: if the hook is no longer active, stop the loop.
      if (!isActive || !holisticRef.current) {
        return;
      }
      
      const videoElement = videoRef.current;
      
      // Poll until the video is ready to be processed
      if (videoElement && videoElement.readyState >= 3) { // HAVE_FUTURE_DATA
        try {
          await holisticRef.current.send({ image: videoElement });
        } catch (error) {
          console.error("MediaPipe failed to send frame:", error);
          // Stop the loop if a critical error occurs
          return;
        }
      }
      
      // Request the next frame to continue the loop
      animationFrameIdRef.current = requestAnimationFrame(processFrameLoop);
    };

    if (isActive) {
      if (!holisticRef.current) {
        // Initialize Holistic once
        const holistic = new Holistic({
          locateFile: (file) => `/${file}`, // Use local files from the /public directory
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

        holistic.onResults(onResults);
        holisticRef.current = holistic;
      }

      // Start the processing loop
      processFrameLoop();

    }

    // Cleanup function runs when isActive becomes false or the component unmounts
    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [isActive, onResults, videoRef]);
};

export default useMediaPipe;