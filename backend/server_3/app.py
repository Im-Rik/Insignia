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
# Suppress standard Flask logging to keep the terminal clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app) # Allow all origins for simplicity

# --- Configuration ---
WEIGHTS_PATH = "sign_recognizer_best.pth"
TARGET_FRAMES = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FOLDER = "temp_processing"
os.makedirs(TEMP_FOLDER, exist_ok=True)
mp_holistic = mp.solutions.holistic

# --- In-memory queue to pass results between threads ---
# This dictionary will hold a queue for each processing job.
# Key: job_id, Value: queue.Queue()
job_queues = {}

# =============================================================================
# 2. MODEL AND KEYPOINT LOGIC
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
        h, _ = self.lstm(x); alpha = torch.softmax(self.attn(h).squeeze(-1), 1); ctx = (h * alpha.unsqueeze(-1)).sum(1); return self.head(self.norm(ctx))

print("[INFO] Loading model and weights...")
try:
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = SignRecognizer(1662, len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # CORRECTED THIS LINE:
    print(f"[SUCCESS] Model loaded successfully: {len(classes)} classes on {str(DEVICE).upper()} device.")
except Exception as e:
    print(f"[ERROR] Critical error loading model: {e}")
    model, classes = None, []


# =============================================================================
# 3. BACKGROUND VIDEO PROCESSING
# =============================================================================

def process_video_in_background(main_video_path, segments, job_id):
    print(f"\n--- [THREAD START] Processing job {job_id} ---")
    q = job_queues.get(job_id)
    if not q:
        print(f"[ERROR] Could not find job queue for {job_id}. Aborting thread.")
        return

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for i, segment in enumerate(segments):
                segment_id = segment['id']
                start_time = segment['start']
                end_time = segment['end']
                clip_path = os.path.join(TEMP_FOLDER, f"job_{job_id}_segment_{segment_id}.webm")
                
                print(f"  [Segment {i+1}/{len(segments)}] Extracting clip ({start_time:.2f}s -> {end_time:.2f}s)...")

                try:
                    # Use ffmpeg to create a precise, temporary clip for the segment
                    (
                        ffmpeg
                        .input(main_video_path, ss=start_time, to=end_time)
                        .output(clip_path, loglevel="quiet", **{'c:v': 'libvpx-vp9', 'an': None}) # no audio
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    )

                    all_keypoints = []
                    # Open the newly created clip with OpenCV for frame processing
                    cap = cv2.VideoCapture(clip_path)
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break # End of the segment clip

                        # Process frame with MediaPipe
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        all_keypoints.append(extract_keypoints(results))
                    cap.release()

                    if not all_keypoints:
                        raise RuntimeError("Could not extract any keypoints from the segment clip.")

                    # Resample frames to match the model's expected input size
                    indices = np.linspace(0, len(all_keypoints) - 1, TARGET_FRAMES, dtype=int) if len(all_keypoints) > TARGET_FRAMES else range(len(all_keypoints))
                    final_keypoints = [all_keypoints[i] for i in indices]

                    # Add padding if there are fewer frames than the target
                    if len(final_keypoints) < TARGET_FRAMES:
                        padding = np.zeros((TARGET_FRAMES - len(final_keypoints), 1662), dtype=np.float32)
                        final_keypoints = np.vstack([np.array(final_keypoints, dtype=np.float32), padding])

                    # Get prediction from the model
                    seq = torch.from_numpy(np.array(final_keypoints, dtype=np.float32)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(seq), 1)[0].cpu().numpy()
                    
                    prediction = classes[int(probs.argmax())]
                    confidence = float(probs.max())
                    result_data = {'segmentId': segment_id, 'prediction': prediction, 'confidence': confidence}
                    
                    # Put the successful result into the queue for the client
                    q.put(result_data)
                    print(f"  [Segment {i+1}/{len(segments)}] Prediction: '{prediction}'. Pushed to queue.")

                except Exception as e:
                    error_message = f"Failed to process segment {segment_id}: {e}"
                    print(f"  [ERROR] {error_message}")
                    traceback.print_exc()
                    # Put the error into the queue so the client knows something went wrong
                    q.put({'error': str(e), 'segmentId': segment_id})
                
                finally:
                    # Clean up the temporary segment clip
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

    finally:
        # Signal that the job is finished by putting 'None' in the queue
        q.put(None)
        
        # Clean up the main uploaded video
        if os.path.exists(main_video_path):
            os.remove(main_video_path)
            print(f"[CLEANUP] Main video deleted: {os.path.basename(main_video_path)}")
        
        print(f"--- [THREAD END] Finished job {job_id} ---")


# =============================================================================
# 4. HTTP API ENDPOINTS
# =============================================================================

@app.route('/classify', methods=['POST'])
def handle_classification_request():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    if 'segments' not in request.form:
        return jsonify({"error": "No segments data provided"}), 400

    video_file = request.files['video']
    segments = json.loads(request.form['segments'])

    # Save the uploaded video to a temporary file
    filename = f"{uuid.uuid4()}.webm"
    main_video_path = os.path.join(TEMP_FOLDER, filename)
    video_file.save(main_video_path)
    
    # Create a unique job ID and a queue for its results
    job_id = str(uuid.uuid4())
    job_queues[job_id] = queue.Queue()

    # Start the processing in a separate thread to avoid blocking the request
    thread = threading.Thread(target=process_video_in_background, args=(main_video_path, segments, job_id))
    thread.start()
    
    print(f"\n[API] Job '{job_id}' created for {len(segments)} segments. Handing off to background thread.")
    # Immediately respond to the client with the job ID
    return jsonify({"job_id": job_id}), 202 # 202 Accepted

@app.route('/stream/<job_id>')
def stream_results(job_id):
    q = job_queues.get(job_id)
    if not q:
        return jsonify({"error": "Invalid or expired job ID"}), 404

    # This function will be a generator for the streaming response
    def generate():
        print(f"[STREAM] Client connected to stream for job {job_id}.")
        while True:
            # Block and wait for a result to appear in the queue
            result = q.get()
            
            # The 'None' object signals the end of the job
            if result is None:
                break
            
            # Format the data as a Server-Sent Event and yield it
            yield f"data: {json.dumps(result)}\n\n"
        
        # Clean up the queue for this job from memory
        if job_id in job_queues:
            del job_queues[job_id]
        print(f"[STREAM] Stream closed for job {job_id}. Queue deleted.")

    # Return the generator function wrapped in a Flask Response
    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    print("\n--- Starting Flask Classification Server ---")
    app.run(host='0.0.0.0', port=5020, threaded=True)