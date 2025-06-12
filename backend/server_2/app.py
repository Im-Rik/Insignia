import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ===================== Settings from the working demo =====================
# These parameters control the prediction stability and confidence.
WIN_SIZE = 60           # Sliding-window length (number of frames)
SMOOTH = 10             # Number of consistent predictions required for a sign to be recognized
CONF_THRESH = 0.6       # Minimum confidence for each frame in the smoothing window
WEIGHTS_PATH = "sign_recognizer_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Flask & SocketIO Setup =====================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# ===================== Model Definition =====================
# This structure is identical to your friend's working script.
class SignRecognizer(nn.Module):
    def __init__(self, in_dim=1662, n_cls=35, hidden=128, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden, n_cls)
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        α = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        ctx = (h * α.unsqueeze(-1)).sum(dim=1)
        return self.head(self.norm(ctx))

# ===================== Model Loading =====================
print("⌛ Loading model and weights...")
try:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    CLASSES = ckpt["classes"]
    
    # Initialize model with the correct number of classes from the checkpoint
    model = SignRecognizer(n_cls=len(CLASSES)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    print(f"✅ Model loaded successfully — {len(CLASSES)} classes detected.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, CLASSES = None, []

# ===================== Real-time Logic =====================
# Stores data for each connected client
client_data = {}

@socketio.on('connect')
def handle_connect():
    """Initializes data structures for a new client."""
    sid = request.sid
    print(f"✅ Client connected: {sid}")
    client_data[sid] = {
        'seq': deque(maxlen=WIN_SIZE),  # Stores keypoint sequence
        'hist': deque(maxlen=SMOOTH)    # Stores prediction history for smoothing
    }

@socketio.on('disconnect')
def handle_disconnect():
    """Clears data for a disconnected client."""
    sid = request.sid
    print(f"❌ Client disconnected: {sid}")
    if sid in client_data:
        del client_data[sid]

@socketio.on('live_keypoints')
def handle_live_keypoints(kps):
    """
    Receives keypoints, performs inference, and emits a stable prediction.
    """
    sid = request.sid
    # Ensure the model is loaded and the client is registered
    if not model or sid not in client_data:
        return

    try:
        data = client_data[sid]
        # Append new keypoints to the client's sequence buffer
        data['seq'].append(np.array(kps))

        # Only proceed if we have a full window of frames
        if len(data['seq']) < WIN_SIZE:
            return

        # ---- 1. Inference ----
        # Stack the sequence into a batch and get model prediction
        X = torch.from_numpy(np.stack(data['seq'])).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            probs = torch.softmax(model(X), dim=1)[0].cpu().numpy()
        
        pred = int(probs.argmax())
        conf = float(probs[pred])

        # ---- 2. Update History ----
        # Add the latest prediction and confidence to the history buffer
        data['hist'].append((pred, conf))

        # ---- 3. Check for Stability and Confidence ----
        displayed_word = "NA"
        display_confidence = 0.0

        # Only check for stability if the history buffer is full
        if len(data['hist']) == SMOOTH:
            preds_in_hist, confs_in_hist = zip(*data['hist'])

            # Condition 1: All predictions in the history must be the same
            is_consistent = all(p == pred for p in preds_in_hist)
            # Condition 2: All confidences must be above the threshold
            is_confident = all(c >= CONF_THRESH for c in confs_in_hist)

            if is_consistent and is_confident:
                displayed_word = CLASSES[pred]
                display_confidence = np.mean(confs_in_hist)

        # ---- 4. Emit Result ----
        # Send the current state ("NA" or the stable sign) to the frontend
        emit('prediction_result', {
            'prediction': displayed_word,
            'confidence': f"{display_confidence:.2f}"
        })

    except Exception as e:
        print(f"❌ Error during keypoint handling for SID {sid}: {e}")

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server with real-time smoothing...")
    # use_reloader=False is important to prevent the model from loading twice in debug mode
    socketio.run(app, host='0.0.0.0', port=5010, debug=True, use_reloader=False)