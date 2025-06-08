// utils/keypointUtils.js

// Pre-allocate arrays to avoid repeated memory allocation
const POSE_SIZE = 33 * 4;
const FACE_SIZE = 468 * 3;
const HAND_SIZE = 21 * 3;
const TOTAL_SIZE = POSE_SIZE + FACE_SIZE + HAND_SIZE * 2;

// Reusable buffer to avoid creating new arrays every frame
let keypointBuffer = new Float32Array(TOTAL_SIZE);

export const extractAllKeypoints = (results) => {
  let offset = 0;

  // Pose landmarks (33 landmarks x [x, y, z, visibility])
  if (results.poseLandmarks) {
    for (let i = 0; i < results.poseLandmarks.length; i++) {
      const landmark = results.poseLandmarks[i];
      keypointBuffer[offset++] = landmark.x;
      keypointBuffer[offset++] = landmark.y;
      keypointBuffer[offset++] = landmark.z;
      keypointBuffer[offset++] = landmark.visibility || 0;
    }
  } else {
    // Fill with zeros if no pose landmarks
    keypointBuffer.fill(0, offset, offset + POSE_SIZE);
    offset += POSE_SIZE;
  }

  // Face landmarks (468 landmarks x [x, y, z])
  if (results.faceLandmarks) {
    for (let i = 0; i < results.faceLandmarks.length; i++) {
      const landmark = results.faceLandmarks[i];
      keypointBuffer[offset++] = landmark.x;
      keypointBuffer[offset++] = landmark.y;
      keypointBuffer[offset++] = landmark.z;
    }
  } else {
    // Fill with zeros if no face landmarks
    keypointBuffer.fill(0, offset, offset + FACE_SIZE);
    offset += FACE_SIZE;
  }

  // Left Hand landmarks (21 landmarks x [x, y, z])
  if (results.leftHandLandmarks) {
    for (let i = 0; i < results.leftHandLandmarks.length; i++) {
      const landmark = results.leftHandLandmarks[i];
      keypointBuffer[offset++] = landmark.x;
      keypointBuffer[offset++] = landmark.y;
      keypointBuffer[offset++] = landmark.z;
    }
  } else {
    // Fill with zeros if no left hand landmarks
    keypointBuffer.fill(0, offset, offset + HAND_SIZE);
    offset += HAND_SIZE;
  }

  // Right Hand landmarks (21 landmarks x [x, y, z])
  if (results.rightHandLandmarks) {
    for (let i = 0; i < results.rightHandLandmarks.length; i++) {
      const landmark = results.rightHandLandmarks[i];
      keypointBuffer[offset++] = landmark.x;
      keypointBuffer[offset++] = landmark.y;
      keypointBuffer[offset++] = landmark.z;
    }
  } else {
    // Fill with zeros if no right hand landmarks
    keypointBuffer.fill(0, offset, offset + HAND_SIZE);
    offset += HAND_SIZE;
  }

  // Return a slice of the buffer (creates a new typed array view)
  return keypointBuffer.slice(0, TOTAL_SIZE);
};

// Utility function to get keypoint statistics (for debugging)
export const getKeypointStats = (keypointsArray) => {
  const nonZeroCount = keypointsArray.reduce((count, val) => count + (val !== 0 ? 1 : 0), 0);
  const avgValue = keypointsArray.reduce((sum, val) => sum + Math.abs(val), 0) / keypointsArray.length;
  
  return {
    totalKeypoints: keypointsArray.length,
    nonZeroKeypoints: nonZeroCount,
    averageValue: avgValue,
    hasPose: keypointsArray.slice(0, POSE_SIZE).some(v => v !== 0),
    hasFace: keypointsArray.slice(POSE_SIZE, POSE_SIZE + FACE_SIZE).some(v => v !== 0),
    hasLeftHand: keypointsArray.slice(POSE_SIZE + FACE_SIZE, POSE_SIZE + FACE_SIZE + HAND_SIZE).some(v => v !== 0),
    hasRightHand: keypointsArray.slice(POSE_SIZE + FACE_SIZE + HAND_SIZE).some(v => v !== 0)
  };
};