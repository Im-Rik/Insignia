# app.py (Final version with Frame-by-Frame processing for Memory Efficiency)

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import ffmpeg
import logging
import traceback
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ---------- Basic App Setup ----------------------
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=50 * 1024 * 1024,
    ping_timeout=20,
    ping_interval=10
)

# --- Configuration and Helper functions ---
WEIGHTS_PATH = "sign_recognizer_best.pth"
TARGET_FRAMES = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FOLDER = "temp_processing"
DEBUG_CLIP_FOLDER = "debug_clips"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(DEBUG_CLIP_FOLDER, exist_ok=True)
mp_holistic = mp.solutions.holistic

def extract_keypoints(res):
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]).flatten() if res.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

# --- Model Definition ---
class SignRecognizer(nn.Module):
    def __init__(self, in_dim=1662, n_cls=15, hidden=128, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_cls))
    def forward(self, x):
        h, _ = self.lstm(x); α = torch.softmax(self.attn(h).squeeze(-1), 1); ctx = (h * α.unsqueeze(-1)).sum(1); return self.head(self.norm(ctx))

# --- Load Model ---
print("⌛ Loading model and weights...")
try:
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE); classes = ckpt["classes"]; model = SignRecognizer(1662, len(classes)).to(DEVICE); model.load_state_dict(ckpt["state_dict"]); model.eval(); print(f"✅ Model loaded successfully — {len(classes)} classes on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {e}"); model, classes = None, []

# --- Background Task ---
def process_video_in_background(main_video_path, segments, sid):
    print(f"--- [BACKGROUND TASK] Starting for client {sid} ---")
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for segment in segments:
                segment_id = segment['id']
                start_time = segment['start']
                end_time = segment['end']
                print(f"  - Processing segment {segment_id} ({start_time:.2f}s -> {end_time:.2f}s)...")
                
                clip_path = os.path.join(DEBUG_CLIP_FOLDER, f"segment_{segment_id}.webm")

                try:
                    (
                        ffmpeg
                        .input(main_video_path, ss=start_time, to=end_time)
                        .output(clip_path, loglevel="quiet", **{'c:v': 'libvpx-vp9'})
                        .run(overwrite_output=True)
                    )
                    print(f"  - ✅ Segment clip saved for debugging: {clip_path}")

                    cap = cv2.VideoCapture(clip_path)
                    
                    # --- MEMORY EFFICIENCY REFACTOR ---
                    # We will no longer store all frames in memory.
                    # Instead, we process frame-by-frame.
                    all_keypoints = []
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break # End of video clip
                        
                        # Process one frame to get keypoints
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        all_keypoints.append(extract_keypoints(results))
                        
                        # Yield control to keep the server responsive
                        socketio.sleep(0)
                    cap.release()

                    if not all_keypoints:
                        raise RuntimeError("Could not extract any keypoints from the segment clip.")

                    # Now, resample the list of KEYPOINTS, not frames.
                    if len(all_keypoints) > TARGET_FRAMES:
                        indices = np.linspace(0, len(all_keypoints) - 1, TARGET_FRAMES, dtype=int)
                        final_keypoints = [all_keypoints[i] for i in indices]
                    else:
                        final_keypoints = all_keypoints
                    
                    # Pad if necessary
                    if len(final_keypoints) < TARGET_FRAMES:
                        padding = np.zeros((TARGET_FRAMES - len(final_keypoints), 1662), dtype=np.float32)
                        final_keypoints = np.vstack([np.array(final_keypoints, dtype=np.float32), padding])
                    
                    # Make prediction
                    seq = torch.from_numpy(np.array(final_keypoints, dtype=np.float32)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(seq), 1)[0].cpu().numpy()
                    
                    prediction = classes[int(probs.argmax())]
                    confidence = float(probs.max())
                    
                    print(f"  - ✅ Prediction: '{prediction}'. Emitting result to {sid}")
                    socketio.emit('single_prediction_result', 
                                  {'segmentId': segment_id, 'prediction': prediction, 'confidence': confidence},
                                  room=sid)

                except Exception as e:
                    error_message = f"ERROR processing segment {segment_id}: {e}"
                    print(f"  - ❌ {error_message}")
                    traceback.print_exc()
                    socketio.emit('prediction_error', {'error': str(e), 'segmentId': segment_id}, room=sid)
                
                socketio.sleep(0)
    
    finally:
        if os.path.exists(main_video_path):
            os.remove(main_video_path)
            print(f"🗑️ Main video deleted: {main_video_path}")
        print(f"--- [BACKGROUND TASK] Finished for client {sid} ---")

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('process_video_with_segments')
def handle_video_processing(data):
    sid = request.sid
    print(f"\n--- [REQUEST HANDLER] Received event from {sid}. Dispatching to background task. ---")
    video_blob = data['video']
    segments = data['segments']
    main_video_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.webm")
    with open(main_video_path, 'wb') as f:
        f.write(video_blob)
    print(f"💾 Main video saved to: {main_video_path}. Starting background processing...")
    socketio.start_background_task(
        process_video_in_background, 
        main_video_path, 
        segments, 
        sid
    )

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server with Eventlet...")
    socketio.run(app, host='0.0.0.0', port=5000)