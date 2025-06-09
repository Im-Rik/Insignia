import os
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# (All the setup code is the same...)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
socketio = SocketIO(app, cors_allowed_origins="*")
WEIGHTS_PATH = "sign_recognizer_best.pth"
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.0
PREDICTION_INTERVAL_MS = 400
COOLDOWN_PERIOD_S = 2.0
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
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = SignRecognizer(1662, len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅ Model loaded successfully — {len(classes)} classes")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, classes = None, []

client_data = {}

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"✅ Client connected: {sid}")
    client_data[sid] = {
        'buffer': deque(maxlen=SEQ_LEN),
        'last_prediction_time': 0,
        'last_emit_time': 0
    }

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"❌ Client disconnected: {sid}")
    if sid in client_data:
        del client_data[sid]

@socketio.on('live_keypoints')
def handle_live_keypoints(kps):
    # --- NEW FAILSAFE PRINT STATEMENT ---
    print("--- handle_live_keypoints function entered ---")
    
    sid = request.sid
    # This check is the most likely reason for the silence.
    if not model or sid not in client_data:
        # This new print statement will tell us exactly why it's stopping.
        print(f"--- Exiting early. Model loaded: {model is not None}. Client data exists: {sid in client_data}")
        return

    try:
        # (The rest of the function is identical to the previous version)
        current_time_ms = int(time.time() * 1000)
        current_time_s = time.time()
        client_data[sid]['buffer'].append(np.array(kps))
        
        if len(client_data[sid]['buffer']) < SEQ_LEN:
            return
        
        if (current_time_ms - client_data[sid]['last_prediction_time']) < PREDICTION_INTERVAL_MS:
            return
        
        client_data[sid]['last_prediction_time'] = current_time_ms

        if (current_time_s - client_data[sid]['last_emit_time']) < COOLDOWN_PERIOD_S:
            return
        
        with torch.no_grad():
            seq = torch.from_numpy(np.stack(client_data[sid]['buffer'])).unsqueeze(0).to(DEVICE).float()
            probs = torch.softmax(model(seq), 1)[0]
            confidence = probs.max().item()
            prediction_idx = probs.argmax().item()
            prediction = classes[prediction_idx]
            
            print(f"Raw Prediction: '{prediction}', Confidence: {confidence:.4f}")

            if confidence > CONFIDENCE_THRESHOLD:
                print(f"---> EMITTING '{prediction}' to frontend (Confidence > {CONFIDENCE_THRESHOLD})")
                emit('prediction_result', {'prediction': prediction, 'confidence': f"{confidence:.2f}"})
                client_data[sid]['last_emit_time'] = current_time_s

    except Exception as e:
        print(f"❌ Error during keypoint handling: {e}")


if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server in REAL-TIME mode...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)