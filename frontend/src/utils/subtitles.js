// Mock subtitles for testing
export const mockSubtitles = [
  "Hello there!",
  "Welcome to this live caption demo.",
  "This interface is now responsive.",
  "Sign language is being recognized.",
  "The model is processing your gestures.",
  "Accuracy varies with lighting conditions.",
  "Keep your hands in frame.",
  "Multiple modes are available.",
  "Developer mode shows technical details.",
  "Upload videos for batch processing.",
  "Real-time translation is active.",
  "MediaPipe landmarks detected.",
  "Backend connection established.",
  "Processing gesture sequences.",
  "Translation confidence is high."
];

// Mock MediaPipe connection points for hand landmarks
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
  [5, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
  [9, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17]                                   // Palm
];

// Mock accuracy ranges for different scenarios
export const mockAccuracyRanges = {
  goodLighting: { min: 85, max: 98 },
  normalLighting: { min: 70, max: 85 },
  poorLighting: { min: 50, max: 70 }
};

// Mock gesture categories for testing
export const mockGestures = [
  { id: 'hello', name: 'Hello', confidence: 0.92 },
  { id: 'thanks', name: 'Thank you', confidence: 0.88 },
  { id: 'yes', name: 'Yes', confidence: 0.95 },
  { id: 'no', name: 'No', confidence: 0.93 },
  { id: 'please', name: 'Please', confidence: 0.87 },
  { id: 'sorry', name: 'Sorry', confidence: 0.85 },
  { id: 'help', name: 'Help', confidence: 0.90 },
  { id: 'stop', name: 'Stop', confidence: 0.94 }
];