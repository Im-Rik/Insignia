import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

WIN_SIZE = 60
SMOOTH = 10
CONF_THRESH = 0.6
WEIGHTS_PATH = "sign_recognizer_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

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

print("Loading model and weights...")
try:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    CLASSES = ckpt["classes"]
    
    model = SignRecognizer(n_cls=len(CLASSES)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    print(f"Model loaded successfully — {len(CLASSES)} classes detected.")
except Exception as e:
    print(f"Error loading model: {e}")
    model, CLASSES = None, []

client_data = {}

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    client_data[sid] = {
        'seq': deque(maxlen=WIN_SIZE),
        'hist': deque(maxlen=SMOOTH)
    }

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in client_data:
        del client_data[sid]

@socketio.on('live_keypoints')
def handle_live_keypoints(kps):
    sid = request.sid
    if not model or sid not in client_data:
        return

    try:
        data = client_data[sid]
        data['seq'].append(np.array(kps))

        if len(data['seq']) < WIN_SIZE:
            return

        X = torch.from_numpy(np.stack(data['seq'])).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            probs = torch.softmax(model(X), dim=1)[0].cpu().numpy()
        
        pred = int(probs.argmax())
        conf = float(probs[pred])

        data['hist'].append((pred, conf))

        displayed_word = "NA"
        display_confidence = 0.0

        if len(data['hist']) == SMOOTH:
            preds_in_hist, confs_in_hist = zip(*data['hist'])

            is_consistent = all(p == pred for p in preds_in_hist)
            is_confident = all(c >= CONF_THRESH for c in confs_in_hist)

            if is_consistent and is_confident:
                displayed_word = CLASSES[pred]
                display_confidence = np.mean(confs_in_hist)

        emit('prediction_result', {
            'prediction': displayed_word,
            'confidence': f"{display_confidence:.2f}"
        })

    except Exception as e:
        print(f"Error during keypoint handling for SID {sid}: {e}")

if __name__ == '__main__':
    print(" [######] (Sliding window) Real-time prediction server")
    print(" [######] Starting server on http://0.0.0.0:5010")
    socketio.run(app, host='0.0.0.0', port=5010, debug=True, use_reloader=False)