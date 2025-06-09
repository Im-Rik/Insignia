// components/MediaPipeOverlay.jsx
import React, { useRef, useEffect, useState } from 'react';
import { POSE_CONNECTIONS, HAND_CONNECTIONS } from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

function MediaPipeOverlay({ results, videoRef, isProcessing }) {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastDrawTimeRef = useRef(0);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  // Effect to set the canvas size based on the video dimensions
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateCanvasSize = () => {
      if (video.videoWidth && video.videoHeight) {
        setCanvasSize({ width: video.videoWidth, height: video.videoHeight });
      }
    };

    // Check if video dimensions are already available
    if (video.videoWidth && video.videoHeight) {
      updateCanvasSize();
    } else {
      // Otherwise, wait for the 'loadedmetadata' event
      video.addEventListener('loadedmetadata', updateCanvasSize);
      return () => video.removeEventListener('loadedmetadata', updateCanvasSize);
    }
  }, [videoRef]);

  // Main effect for drawing the landmarks
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return; // Exit if canvas is not yet available

    // If processing has stopped, clear the canvas and do nothing else.
    if (!isProcessing) {
      // Cancel any leftover animation frame from a previous run
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      const canvasCtx = canvas.getContext('2d');
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      return; // Stop execution for this effect
    }

    // Don't draw if we have no results or the canvas isn't sized yet
    if (!results || canvasSize.width === 0) return;

    // Cancel any pending animation frame before starting a new one
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const draw = (timestamp) => {
      // Throttle drawing to a maximum of ~30fps
      if (timestamp - lastDrawTimeRef.current < 33) {
        animationFrameRef.current = requestAnimationFrame(draw);
        return;
      }
      lastDrawTimeRef.current = timestamp;

      const canvasCtx = canvas.getContext('2d', { alpha: true });
      
      // Ensure canvas dimensions match the desired size
      if (canvas.width !== canvasSize.width || canvas.height !== canvasSize.height) {
        canvas.width = canvasSize.width;
        canvas.height = canvasSize.height;
      }

      // Clear the canvas before each draw
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

      // Save context state for transformation
      canvasCtx.save();
      // Apply a horizontal flip for the mirror effect
      canvasCtx.translate(canvas.width, 0);
      canvasCtx.scale(-1, 1);

      // Draw Pose landmarks
      if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { 
          color: '#00FF00', 
          lineWidth: 4 
        });
        drawLandmarks(canvasCtx, results.poseLandmarks, { 
          color: '#FF0000', 
          lineWidth: 2,
          radius: 4
        });
      }

      // Draw Left Hand landmarks
      if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { 
          color: '#CC0000', 
          lineWidth: 5 
        });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { 
          color: '#00FF00', 
          lineWidth: 2,
          radius: 3
        });
      }

      // Draw Right Hand landmarks
      if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { 
          color: '#00CC00', 
          lineWidth: 5 
        });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { 
          color: '#FF0000', 
          lineWidth: 2,
          radius: 3
        });
      }

      // Restore the context to its original state
      canvasCtx.restore();
    };

    // Start the drawing loop
    animationFrameRef.current = requestAnimationFrame(draw);

    // Cleanup function to cancel the animation frame when the component unmounts or dependencies change
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [results, canvasSize, isProcessing]); // Dependency array includes isProcessing

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
      style={{ imageRendering: 'crisp-edges' }}
    />
  );
}

// Use React.memo to prevent re-renders if props haven't changed
export default React.memo(MediaPipeOverlay);