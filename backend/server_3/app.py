# app.py (Classification Server - Final HTTP/SSE Version)

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
import json
import queue
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# =============================================================================
# 1. FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app) # Allow all origins for simplicity

# --- Configuration (Unchanged) ---
WEIGHTS_PATH = "sign_recognizer_best.pth"
TARGET_FRAMES = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FOLDER = "temp_processing"
DEBUG_CLIP_FOLDER = "debug_clips"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(DEBUG_CLIP_FOLDER, exist_ok=True)
mp_holistic = mp.solutions.holistic

# --- NEW: In-memory queue to pass results between threads ---
# This dictionary will hold a queue for each processing job.
# Key: job_id, Value: queue.Queue()
job_queues = {}

# =============================================================================
# 2. MODEL AND KEYPOINT LOGIC (Unchanged)
# =============================================================================

def extract_keypoints(res):
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]).flatten() if res.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

class SignRecognizer(nn.Module):
    def __init__(self, in_dim=1662, n_cls=15, hidden=128, drop=0.4):
        super().__init__(); self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True); self.attn = nn.Linear(hidden * 2, 1); self.norm = nn.LayerNorm(hidden * 2); self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_cls))
    def forward(self, x):
        h, _ = self.lstm(x); α = torch.softmax(self.attn(h).squeeze(-1), 1); ctx = (h * α.unsqueeze(-1)).sum(1); return self.head(self.norm(ctx))

print("⌛ Loading model and weights...")
try:
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE); classes = ckpt["classes"]; model = SignRecognizer(1662, len(classes)).to(DEVICE); model.load_state_dict(ckpt["state_dict"]); model.eval(); print(f"✅ Model loaded successfully — {len(classes)} classes on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {e}"); model, classes = None, []


# =============================================================================
# 3. BACKGROUND PROCESSING (Now uses a Queue)
# =============================================================================

# In your classification server app.py, replace the old function with this new one.

def process_video_in_background(main_video_path, segments, job_id):
    print(f"--- [BACKGROUND THREAD] Starting for job {job_id} ---")
    q = job_queues.get(job_id)
    if not q:
        return

    try:
        # No need to open the main video with OpenCV here anymore.
        # We will let ffmpeg handle all extraction.
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for segment in segments:
                segment_id = segment['id']
                start_time = segment['start']
                end_time = segment['end']
                
                # Use a unique path for the temporary clip for this specific segment
                clip_path = os.path.join(TEMP_FOLDER, f"job_{job_id}_segment_{segment_id}.webm")
                
                print(f"  - Extracting segment {segment_id} ({start_time:.2f}s -> {end_time:.2f}s) using ffmpeg...")

                try:
                    # <<< CHANGE: Reverted to the reliable ffmpeg extraction method
                    # This creates a frame-accurate temporary clip for the segment.
                    (
                        ffmpeg
                        .input(main_video_path, ss=start_time, to=end_time)
                        .output(clip_path, loglevel="quiet", **{'c:v': 'libvpx-vp9', 'an': None}) # Ensure no audio
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    )

                    all_keypoints = []
                    
                    # <<< CHANGE: Open the newly created, accurate clip with OpenCV
                    cap = cv2.VideoCapture(clip_path)
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break  # End of the segment clip

                        # Process the frame as before
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        all_keypoints.append(extract_keypoints(results))
                    cap.release()

                    if not all_keypoints:
                        raise RuntimeError("Could not extract any keypoints from the segment clip.")

                    # --- Resampling, Padding, and Prediction logic remains identical ---
                    indices = np.linspace(0, len(all_keypoints) - 1, TARGET_FRAMES, dtype=int) if len(all_keypoints) > TARGET_FRAMES else range(len(all_keypoints))
                    final_keypoints = [all_keypoints[i] for i in indices]

                    if len(final_keypoints) < TARGET_FRAMES:
                        padding = np.zeros((TARGET_FRAMES - len(final_keypoints), 1662), dtype=np.float32)
                        final_keypoints = np.vstack([np.array(final_keypoints, dtype=np.float32), padding])

                    seq = torch.from_numpy(np.array(final_keypoints, dtype=np.float32)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(seq), 1)[0].cpu().numpy()
                    
                    prediction, confidence = classes[int(probs.argmax())], float(probs.max())
                    result_data = {'segmentId': segment_id, 'prediction': prediction, 'confidence': confidence}
                    
                    q.put(result_data)
                    print(f"  - ✅ Prediction: '{prediction}'. Pushed to queue for job {job_id}")

                except Exception as e:
                    error_message = f"ERROR processing segment {segment_id}: {e}"
                    print(f"  - ❌ {error_message}"); traceback.print_exc()
                    q.put({'error': str(e), 'segmentId': segment_id})
                
                finally:
                    # <<< CHANGE: Clean up the temporary segment clip
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

    finally:
        # Signal that the job is finished
        q.put(None)
        
        # Clean up the main uploaded video
        if os.path.exists(main_video_path):
            os.remove(main_video_path)
            print(f"🗑️ Main video deleted: {main_video_path}")
        print(f"--- [BACKGROUND THREAD] Finished for job {job_id} ---")


# =============================================================================
# 4. HTTP API ENDPOINTS
# =============================================================================

@app.route('/classify', methods=['POST'])
def handle_classification_request():
    if 'video' not in request.files: return jsonify({"error": "No video file provided"}), 400
    if 'segments' not in request.form: return jsonify({"error": "No segments data provided"}), 400

    video_file = request.files['video']
    segments_json = request.form['segments']
    segments = json.loads(segments_json)

    main_video_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.webm")
    video_file.save(main_video_path)
    
    job_id = str(uuid.uuid4())
    job_queues[job_id] = queue.Queue()

    # Start the processing in a separate thread
    thread = threading.Thread(target=process_video_in_background, args=(main_video_path, segments, job_id))
    thread.start()
    
    print(f"✅ Job {job_id} created and started. Sending job_id to client.")
    return jsonify({"job_id": job_id}), 202 # 202 Accepted

@app.route('/stream/<job_id>')
def stream_results(job_id):
    q = job_queues.get(job_id)
    if not q: return jsonify({"error": "Invalid job ID"}), 404

    def generate():
        while True:
            # Wait for a result to appear in the queue
            result = q.get()
            if result is None: # This is our signal that the job is done
                break
            # Format the data as a Server-Sent Event
            yield f"data: {json.dumps(result)}\n\n"
        # Clean up the queue for this job
        del job_queues[job_id]
        print(f"🎉 Stream for job {job_id} closed.")

    # Return a streaming response
    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    print("🚀 Starting Flask server for Classification...")
    app.run(host='0.0.0.0', port=5000, threaded=True)