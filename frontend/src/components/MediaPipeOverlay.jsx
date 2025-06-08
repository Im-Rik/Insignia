// components/MediaPipeOverlay.jsx
import React, { useRef, useEffect, useState } from 'react';
import { POSE_CONNECTIONS, FACEMESH_TESSELATION, HAND_CONNECTIONS } from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

function MediaPipeOverlay({ results, videoRef }) {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastDrawTimeRef = useRef(0);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  
  // Only update canvas size when video dimensions change
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    
    const updateCanvasSize = () => {
      if (video.videoWidth && video.videoHeight) {
        setCanvasSize({ width: video.videoWidth, height: video.videoHeight });
      }
    };
    
    // Check video dimensions
    if (video.videoWidth && video.videoHeight) {
      updateCanvasSize();
    } else {
      video.addEventListener('loadedmetadata', updateCanvasSize);
      return () => video.removeEventListener('loadedmetadata', updateCanvasSize);
    }
  }, [videoRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !results || canvasSize.width === 0) return;

    // Cancel any pending animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const draw = (timestamp) => {
      // Limit drawing to 30fps max
      if (timestamp - lastDrawTimeRef.current < 33) {
        animationFrameRef.current = requestAnimationFrame(draw);
        return;
      }
      lastDrawTimeRef.current = timestamp;

      const canvasCtx = canvas.getContext('2d', { alpha: true });
      
      // Set canvas size only if changed
      if (canvas.width !== canvasSize.width || canvas.height !== canvasSize.height) {
        canvas.width = canvasSize.width;
        canvas.height = canvasSize.height;
      }

      // Clear canvas
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

      // Setup transform for mirror effect
      canvasCtx.save();
      canvasCtx.translate(canvas.width, 0);
      canvasCtx.scale(-1, 1);

      // Draw pose landmarks
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

      // Draw face landmarks (simplified - skip tessellation for performance)
      if (results.faceLandmarks && false) { // Disable face mesh for performance
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION, { 
          color: '#C0C0C070', 
          lineWidth: 1 
        });
      }

      // Draw hand landmarks
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

      canvasCtx.restore();
    };

    // Start drawing
    animationFrameRef.current = requestAnimationFrame(draw);

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [results, canvasSize]);

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
      style={{ imageRendering: 'crisp-edges' }}
    />
  );
}

export default React.memo(MediaPipeOverlay);