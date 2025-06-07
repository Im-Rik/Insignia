import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import io from 'socket.io-client';

const SignRecognition = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null); // Hidden canvas for non-mirrored MediaPipe processing
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  
  const [isConnected, setIsConnected] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [rawPrediction, setRawPrediction] = useState(null);
  const [rawConfidence, setRawConfidence] = useState(0);
  const [bufferSize, setBufferSize] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [isContinuous, setIsContinuous] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);

  // Extract keypoints in the EXACT same format as the training data
  const extractKeypoints = useCallback((results) => {
    // This function EXACTLY matches the training code format!
    
    // Pose (33 landmarks x [x, y, z, visibility]) = 132 features
    let pose = [];
    if (results.poseLandmarks) {
      results.poseLandmarks.forEach(landmark => {
        pose.push(landmark.x, landmark.y, landmark.z, landmark.visibility);
      });
    } else {
      // Fill with zeros if no pose detected (same as training)
      pose = new Array(33 * 4).fill(0);
    }
    
    // Face (468 landmarks x [x, y, z]) = 1404 features
    let face = [];
    if (results.faceLandmarks) {
      results.faceLandmarks.forEach(landmark => {
        face.push(landmark.x, landmark.y, landmark.z);
      });
    } else {
      // Fill with zeros if no face detected (same as training)
      face = new Array(468 * 3).fill(0);
    }
    
    // Left hand (21 landmarks x [x, y, z]) = 63 features
    let leftHand = [];
    if (results.leftHandLandmarks) {
      results.leftHandLandmarks.forEach(landmark => {
        leftHand.push(landmark.x, landmark.y, landmark.z);
      });
    } else {
      // Fill with zeros if no left hand detected (same as training)
      leftHand = new Array(21 * 3).fill(0);
    }
    
    // Right hand (21 landmarks x [x, y, z]) = 63 features
    let rightHand = [];
    if (results.rightHandLandmarks) {
      results.rightHandLandmarks.forEach(landmark => {
        rightHand.push(landmark.x, landmark.y, landmark.z);
      });
    } else {
      // Fill with zeros if no right hand detected (same as training)
      rightHand = new Array(21 * 3).fill(0);
    }
    
    // Concatenate in the EXACT same order as training: pose + face + lh + rh
    const keypoints = [...pose, ...face, ...leftHand, ...rightHand];
    
    // Verify we have exactly 1662 features (132 + 1404 + 63 + 63 = 1662)
    if (keypoints.length !== 1662) {
      console.warn(`Expected 1662 keypoints, got ${keypoints.length}`);
    }
    
    return keypoints;
  }, []);

  // Initialize MediaPipe Holistic
  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      }
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    holistic.onResults((results) => {
      if (!isRecording) return;

      // Draw results on the DISPLAY canvas (mirrored for user)
      const displayCanvas = canvasRef.current;
      const displayCtx = displayCanvas.getContext('2d');
      
      displayCanvas.width = videoRef.current.videoWidth;
      displayCanvas.height = videoRef.current.videoHeight;
      
      displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
      // Draw mirrored video for user display
      displayCtx.save();
      displayCtx.scale(-1, 1);
      displayCtx.drawImage(videoRef.current, -displayCanvas.width, 0, displayCanvas.width, displayCanvas.height);
      displayCtx.restore();
      
      // Draw landmarks on the mirrored display canvas
      if (results.poseLandmarks) {
        drawLandmarksMirrored(displayCtx, results.poseLandmarks, 'blue', displayCanvas.width);
      }
      if (results.leftHandLandmarks) {
        drawLandmarksMirrored(displayCtx, results.leftHandLandmarks, 'red', displayCanvas.width);
      }
      if (results.rightHandLandmarks) {
        drawLandmarksMirrored(displayCtx, results.rightHandLandmarks, 'green', displayCanvas.width);
      }

      // Extract keypoints from NON-MIRRORED results (real-world orientation)
      // MediaPipe already processed the original video orientation
      const keypoints = extractKeypoints(results);
      if (socketRef.current && isConnected) {
        socketRef.current.emit('keypoints', { keypoints });
      }
    });

    holisticRef.current = holistic;

    return () => {
      if (holisticRef.current) {
        holisticRef.current.close();
      }
    };
  }, [extractKeypoints, isRecording, isConnected]);

  // Initialize camera
  useEffect(() => {
    if (videoRef.current && holisticRef.current) {
      // Create hidden canvas for non-mirrored MediaPipe processing
      const hiddenCanvas = document.createElement('canvas');
      hiddenCanvasRef.current = hiddenCanvas;
      
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (holisticRef.current && isRecording) {
            // Draw video to hidden canvas in ORIGINAL orientation (non-mirrored)
            const video = videoRef.current;
            hiddenCanvas.width = video.videoWidth;
            hiddenCanvas.height = video.videoHeight;
            const hiddenCtx = hiddenCanvas.getContext('2d');
            hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
            
            // Send the NON-MIRRORED canvas to MediaPipe
            await holisticRef.current.send({ image: hiddenCanvas });
          }
        },
        width: 640,
        height: 480
      });

      camera.start();
      cameraRef.current = camera;

      return () => {
        if (cameraRef.current) {
          cameraRef.current.stop();
        }
      };
    }
  }, [isRecording]);

  // Initialize WebSocket connection
  useEffect(() => {
    const socket = io('http://localhost:5000');
    
    socket.on('connect', () => {
      setIsConnected(true);
      setError(null);
      console.log('Connected to server');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from server');
    });

    socket.on('prediction', (data) => {
      setPrediction(data.prediction);
      setConfidence(data.confidence);
      setRawPrediction(data.raw_prediction || data.prediction);
      setRawConfidence(data.raw_confidence || data.confidence);
      setBufferSize(data.buffer_size);
      setFrameCount(data.frame_count || 0);
      setIsContinuous(data.is_continuous || false);
      
      // Add to prediction history for visualization
      setPredictionHistory(prev => {
        const newHistory = [...prev, {
          prediction: data.prediction,
          confidence: data.confidence,
          timestamp: Date.now()
        }];
        // Keep only last 10 predictions
        return newHistory.slice(-10);
      });
    });

    socket.on('buffer_status', (data) => {
      setBufferSize(data.size);
      setFrameCount(data.frame_count || 0);
    });

    socket.on('error', (data) => {
      setError(data.message);
      console.error('Server error:', data.message);
    });

    socket.on('status', (data) => {
      console.log('Status:', data.message);
    });

    socketRef.current = socket;

    return () => {
      socket.disconnect();
    };
  }, []);

  // Helper function to draw landmarks on mirrored display
  const drawLandmarksMirrored = (ctx, landmarks, color, canvasWidth) => {
    ctx.fillStyle = color;
    landmarks.forEach(landmark => {
      // Mirror the x coordinate for display
      const x = canvasWidth - (landmark.x * canvasWidth);
      const y = landmark.y * ctx.canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Helper function to draw landmarks (non-mirrored, for reference)
  const drawLandmarks = (ctx, landmarks, color) => {
    ctx.fillStyle = color;
    landmarks.forEach(landmark => {
      const x = landmark.x * ctx.canvas.width;
      const y = landmark.y * ctx.canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const startRecording = () => {
    setIsRecording(true);
    setPrediction(null);
    setConfidence(0);
    setRawPrediction(null);
    setRawConfidence(0);
    setBufferSize(0);
    setFrameCount(0);
    setError(null);
    setIsContinuous(false);
    setPredictionHistory([]);
    
    // Clear buffer on server
    if (socketRef.current) {
      socketRef.current.emit('clear_buffer');
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const clearBuffer = () => {
    if (socketRef.current) {
      socketRef.current.emit('clear_buffer');
    }
    setPrediction(null);
    setConfidence(0);
    setRawPrediction(null);
    setRawConfidence(0);
    setBufferSize(0);
    setFrameCount(0);
    setIsContinuous(false);
    setPredictionHistory([]);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Sign Language Recognition</h1>
      
      {/* Connection Status */}
      <div style={{ marginBottom: '20px' }}>
        <span style={{ 
          color: isConnected ? 'green' : 'red',
          fontWeight: 'bold'
        }}>
          {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
      </div>

      {/* Video and Canvas */}
      <div style={{ position: 'relative', marginBottom: '20px' }}>
        <video
          ref={videoRef}
          style={{
            width: '640px',
            height: '480px',
            transform: 'scaleX(-1)', // Mirror for user display only
            position: 'absolute',
            opacity: 0.7 // Slightly transparent so landmarks are visible
          }}
          autoPlay
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          style={{
            width: '640px',
            height: '480px',
            border: '2px solid #333',
            position: 'relative'
          }}
        />
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '5px 10px',
          borderRadius: '4px',
          fontSize: '12px'
        }}>
          📺 Mirrored Display | 🔄 Real-world Processing
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={startRecording}
          disabled={!isConnected || isRecording}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: isRecording ? '#ccc' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: isRecording ? 'not-allowed' : 'pointer'
          }}
        >
          {isRecording ? 'Recording...' : 'Start Recording'}
        </button>
        
        <button
          onClick={stopRecording}
          disabled={!isRecording}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: !isRecording ? '#ccc' : '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: !isRecording ? 'not-allowed' : 'pointer'
          }}
        >
          Stop Recording
        </button>

        <button
          onClick={clearBuffer}
          disabled={!isConnected}
          style={{
            padding: '10px 20px',
            backgroundColor: '#ff9800',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: isConnected ? 'pointer' : 'not-allowed'
          }}
        >
          Clear Buffer
        </button>
      </div>

      {/* Status Display */}
      <div style={{ marginBottom: '20px' }}>
        <h3>Status:</h3>
        <p><strong>Buffer Size:</strong> {bufferSize} / 60 frames</p>
        <p><strong>Total Frames:</strong> {frameCount}</p>
        <p><strong>Mode:</strong> {isContinuous ? '🔄 Continuous Recognition' : '⏳ Buffering'}</p>
        <div style={{
          width: '300px',
          height: '20px',
          backgroundColor: '#f0f0f0',
          borderRadius: '10px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${(bufferSize / 60) * 100}%`,
            height: '100%',
            backgroundColor: bufferSize === 60 ? '#4CAF50' : '#2196F3',
            transition: 'width 0.2s'
          }} />
        </div>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div style={{
          padding: '15px',
          backgroundColor: '#e8f5e8',
          border: '1px solid #4CAF50',
          borderRadius: '5px',
          marginBottom: '20px'
        }}>
          <h3>🎯 Current Prediction:</h3>
          <p style={{ 
            fontSize: '28px', 
            fontWeight: 'bold', 
            color: '#2e7d32',
            margin: '10px 0'
          }}>
            {prediction}
          </p>
          <p style={{ fontSize: '16px', color: '#666' }}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
          
          {/* Show raw prediction if different from smoothed */}
          {rawPrediction && rawPrediction !== prediction && (
            <div style={{ 
              marginTop: '10px', 
              padding: '8px', 
              backgroundColor: '#f5f5f5', 
              borderRadius: '4px',
              fontSize: '14px'
            }}>
              <strong>Raw:</strong> {rawPrediction} ({(rawConfidence * 100).toFixed(1)}%)
            </div>
          )}
          
          {isContinuous && (
            <div style={{ 
              marginTop: '8px', 
              color: '#4CAF50', 
              fontSize: '14px',
              fontWeight: 'bold'
            }}>
              🔄 Live Recognition Active
            </div>
          )}
        </div>
      )}

      {/* Prediction History */}
      {predictionHistory.length > 0 && (
        <div style={{
          padding: '15px',
          backgroundColor: '#f8f9fa',
          border: '1px solid #dee2e6',
          borderRadius: '5px',
          marginBottom: '20px'
        }}>
          <h3>📊 Recent Predictions:</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {predictionHistory.slice(-5).map((pred, index) => (
              <div key={index} style={{
                padding: '4px 8px',
                backgroundColor: index === predictionHistory.length - 1 ? '#4CAF50' : '#e0e0e0',
                color: index === predictionHistory.length - 1 ? 'white' : 'black',
                borderRadius: '12px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                {pred.prediction} ({(pred.confidence * 100).toFixed(0)}%)
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div style={{
          padding: '15px',
          backgroundColor: '#ffebee',
          border: '1px solid #f44336',
          borderRadius: '5px',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#d32f2f' }}>Error:</h3>
          <p style={{ color: '#d32f2f' }}>{error}</p>
        </div>
      )}

      {/* Instructions */}
      <div style={{
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '5px'
      }}>
        <h3>🚀 Instructions (Continuous Recognition):</h3>
        <ol>
          <li>Make sure you're connected to the server</li>
          <li>Click "Start Recording" to begin capturing sign language</li>
          <li>Wait for buffer to fill (60 frames) - you'll see "🔄 Continuous Recognition" when ready</li>
          <li><strong>Now perform different signs continuously!</strong> The system will predict every few frames</li>
          <li>Watch predictions update in real-time as you change gestures</li>
          <li>Smoothed predictions reduce flickering for stable recognition</li>
        </ol>
        
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          backgroundColor: '#e3f2fd', 
          borderRadius: '4px' 
        }}>
          <strong>✨ Try this:</strong> Do "ok" → "hospital" → "go" in sequence and watch the predictions change live!
        </div>
        
        <div style={{ 
          marginTop: '10px', 
          padding: '10px', 
          backgroundColor: '#fff3e0', 
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          <strong>📺 Display vs Processing:</strong><br/>
          • <strong>What you see:</strong> Mirrored video (like a mirror - familiar experience)<br/>
          • <strong>What AI analyzes:</strong> Real-world orientation (your actual left/right hand positions)<br/>
          • This ensures the model receives data in the same format it was trained on!
        </div>
        
        <div style={{ 
          marginTop: '10px', 
          padding: '10px', 
          backgroundColor: '#f3e5f5', 
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          <strong>🔬 Sampling Method:</strong><br/>
          • <strong>Training:</strong> Uniform sampling (60 frames from entire video)<br/>
          • <strong>Real-time:</strong> Sliding window (most recent 60 consecutive frames)<br/>
          • Keypoint extraction is identical to training for compatibility!
        </div>
      </div>
    </div>
  );
};

export default SignRecognition;