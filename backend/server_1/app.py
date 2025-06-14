# app.py (Simplified)

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from threading import Thread
import multiprocessing
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ---------- Basic App Setup ----------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024 # Allow large file uploads
)

# --- Configuration and Helper functions (No changes here) ---
WEIGHTS_PATH = "sign_recognizer_best.pth"
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FOLDER = "temp_videos" # Only one temp folder is needed now
os.makedirs(TEMP_FOLDER, exist_ok=True)
mp_holistic = mp.solutions.holistic

def extract_all_keypoints(res):
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]).flatten() if res.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

def sample_n(frames, n=60):
    if not frames: raise RuntimeError("No frames read from video!")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]

class SignRecognizer(nn.Module):
    def __init__(self, in_dim=1662, n_cls=10, hidden=128, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_cls))
    def forward(self, x):
        h, _ = self.lstm(x); α = torch.softmax(self.attn(h).squeeze(-1), 1); ctx = (h * α.unsqueeze(-1)).sum(1); return self.head(self.norm(ctx))

print("⌛ Loading model and weights...")
try:
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE); classes = ckpt["classes"]; model = SignRecognizer(1662, len(classes)).to(DEVICE); model.load_state_dict(ckpt["state_dict"]); model.eval(); print(f"✅ Model loaded successfully — {len(classes)} classes")
except Exception as e:
    print(f"❌ Error loading model: {e}"); model, classes = None, []

# --- Child process to run prediction (No major changes here) ---
def run_prediction(video_path, sid, q):
    try:
        q.put({'sid': sid, 'event': 'status_update', 'data': {'message': 'Sampling video frames...'}})
        cap = cv2.VideoCapture(video_path)
        frames = [];
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()
        frames = sample_n(frames, SEQ_LEN)
        
        q.put({'sid': sid, 'event': 'status_update', 'data': {'message': 'Extracting keypoints...', 'progress': 25}})
        kps = []
        with mp_holistic.Holistic() as holo:
            for i, frm in enumerate(frames):
                res = holo.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                kps.append(extract_all_keypoints(res))
                if (i + 1) % 10 == 0:
                       progress = 25 + int(((i+1)/len(frames)) * 50) # Keypoint extraction is 50% of the work
                       q.put({'sid': sid, 'event': 'status_update', 'data': {'message': f'Processed {i+1}/{len(frames)} frames...', 'progress': progress}})
        
        q.put({'sid': sid, 'event': 'status_update', 'data': {'message': 'Running model inference...', 'progress': 85}})
        seq = torch.from_numpy(np.stack(kps)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(seq), 1)[0].cpu().numpy()
        prediction = classes[int(probs.argmax())]
        prob_vec = {c: f"{p:.4f}" for c, p in zip(classes, probs)}
        
        q.put({
            'sid': sid, 'event': 'prediction_result',
            'data': {'prediction': prediction, 'probabilities': prob_vec}
        })
    except Exception as e:
        q.put({'sid': sid, 'event': 'prediction_error', 'data': {'error': str(e)}})
    finally:
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.remove(video_path)

# --- REMOVED: Video conversion function is no longer needed ---
# --- REMOVED: /videos/<filename> route is no longer needed ---

@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('predict_video')
def handle_video_prediction(video_data):
    """Receives video, saves it temporarily, and starts the prediction process."""
    sid = request.sid
    print(f"▶️ Received video for prediction from client {sid}...")
    
    # Save the uploaded video to a temporary file
    temp_video_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.mp4")
    with open(temp_video_path, 'wb') as f:
        f.write(video_data)

    # Start ONLY the prediction process in the background
    pred_process = multiprocessing.Process(target=run_prediction, args=(temp_video_path, sid, q))
    pred_process.start()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

def queue_listener(q):
    """Listens to the queue and emits results back to the correct client."""
    while True:
        try:
            message = q.get()
            socketio.emit(
                message['event'],
                message['data'],
                room=message['sid']
            )
        except Exception as e:
            print(f"❌ Error in queue listener: {e}")

if __name__ == '__main__':
    q = multiprocessing.Queue()
    listener = Thread(target=queue_listener, args=(q,))
    listener.daemon = True
    listener.start()
    print("🚀 Starting Flask-SocketIO server (Prediction-Only Mode)...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)