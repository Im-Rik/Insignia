# ss_server.py (Version 5: With User's Tuned Segmentation Logic)

import os
import cv2
import numpy as np
import time
import mediapipe as mp
import multiprocessing as mp_cpu
import uuid
import ffmpeg
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# =============================================================================
# 1. FLASK APPLICATION SETUP (Unchanged)
# =============================================================================

app = Flask(__name__)
CORS(app)

TEMP_UPLOADS_FOLDER = "temp_uploads"
PROCESSED_FOLDER = "processed_clips"
os.makedirs(TEMP_UPLOADS_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# =============================================================================
# 2. GESTURE SEGMENTATION LOGIC (Updated with your new code)
# =============================================================================

# --- Your new, tuned configuration ---
SCALE          = 0.5
FLOW_TH_PIX    = 45.0
SPARSE_POINTS  = 200
SPARSE_QUALITY = 0.01
POSE_IVL       = 2
POSE_TH        = 0.18
IDLE_SPLIT     = 3
MIN_LEN        = 50
MAX_LEN        = 80 # Your new hyperparameter

# --- Your new sparse_flow_mag function ---
def sparse_flow_mag(prev_g: np.ndarray, next_g: np.ndarray) -> float:
    pts = cv2.goodFeaturesToTrack(prev_g, maxCorners=SPARSE_POINTS, qualityLevel=SPARSE_QUALITY, minDistance=10, blockSize=7)
    if pts is None or len(pts) < 10:
        return 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, next_g, pts, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nxt is None or st is None:
        return 0.0
    good_old, good_new = pts[st == 1], nxt[st == 1]
    if len(good_old) < 5:
        return 0.0
    mags = np.linalg.norm(good_new - good_old, axis=1)
    mags = mags[mags < np.percentile(mags, 90)]
    return float(np.mean(mags)) if mags.size else 0.0

# --- Your new process_batch function ---
def process_batch(frame_idx, path, scale):
    cap = cv2.VideoCapture(path)
    out = []
    for i in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok1, f1 = cap.read(); ok2, f2 = cap.read()
        if not (ok1 and ok2):
            out.append((i, 0.0)); continue
        g1 = cv2.cvtColor(cv2.resize(f1, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        out.append((i, sparse_flow_mag(g1, g2)))
    cap.release(); return out

# --- Your new main segmentation function ---
def detect_gesture_segments_sparse(src: str) -> List[Tuple[int, int]]:
    cap = cv2.VideoCapture(src)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if tot < 2: return []
    idxs = list(range(tot - 1))
    n_workers = max(1, mp_cpu.cpu_count() - 1)
    batch = max(1, len(idxs) // n_workers)
    batches = [idxs[i:i + batch] for i in range(0, len(idxs), batch)]

    # 1) parallel sparse flow
    flow_vals = [0.0] * tot
    with ProcessPoolExecutor(n_workers) as ex:
        # Using ex.map for more concise parallel execution
        for res in ex.map(process_batch, batches, [src] * len(batches), [SCALE] * len(batches)):
            for idx, mag in res:
                if idx < tot: flow_vals[idx] = mag

    # 2) pose gate
    frames, pose_raised = [], []
    cap = cv2.VideoCapture(src)
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f); pose_raised.append(False)
    cap.release()

    mp_pose = mp.solutions.pose
    LHIP, RHIP, LWRI, RWRI = 23, 24, 15, 16

    def wrist_up(lm):
        if lm[LHIP].visibility < .3 or lm[RHIP].visibility < .3: return False
        cy = (lm[LHIP].y + lm[RHIP].y) / 2
        torso = abs(lm[LHIP].y - lm[RHIP].y) + 1e-6
        return any((cy - lm[w].y) > POSE_TH * torso for w in (LWRI, RWRI) if lm[w].visibility > .3)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        for i in range(0, len(frames), POSE_IVL):
            sm = cv2.resize(frames[i], None, fx=SCALE, fy=SCALE)
            res = pose.process(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))
            # Robust check for landmarks
            flag = wrist_up(res.pose_landmarks.landmark) if res.pose_landmarks else False
            for j in range(i, min(len(frames), i + POSE_IVL)):
                pose_raised[j] = flag

    # 3) combine flow + pose, idle-split with length clamp
    segs, start, idle = [], None, 0
    for idx in range(len(frames)):
        gest = (flow_vals[idx] > FLOW_TH_PIX) or pose_raised[idx]
        if gest:
            idle = 0
            if start is None:
                start = idx
        else:
            idle += 1
            if start is not None and idle >= IDLE_SPLIT:
                end = idx - IDLE_SPLIT + 1
                segs.append((start, end)); start = None
        # Your new hard split logic
        if start is not None and idx - start + 1 >= MAX_LEN:
            segs.append((start, idx + 1)); start, idle = None, 0

    if start is not None:
        segs.append((start, len(frames)))

    return [(s, e) for s, e in segs if e - s >= MIN_LEN]


# =============================================================================
# 3. VIDEO DRAWING & API ENDPOINTS (Unchanged)
# =============================================================================

def draw_segments_on_video(input_path, output_path, time_segments):
    if not time_segments:
        import shutil
        shutil.copy(input_path, output_path)
        print("No segments to draw, copied original video.")
        return
    filter_chains = [f"between(t,{seg['start']},{seg['end']})" for seg in time_segments]
    enable_condition = "+".join(filter_chains)
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('drawbox', x=10, y=10, width='iw-20', height='ih-20', color='green@0.5', thickness=4, enable=enable_condition)
            .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='fast', movflags='frag_keyframe+empty_moov')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Debug video saved to {output_path} using FFmpeg.")
    except ffmpeg.Error as e:
        print('FFmpeg stdout:', e.stdout.decode('utf8'))
        print('FFmpeg stderr:', e.stderr.decode('utf8'))
        raise e

@app.route('/processed/<path:filename>')
def serve_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/segment', methods=['POST'])
def auto_segment_video():
    if 'video' not in request.files: return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    
    temp_filename = secure_filename(f"temp_{uuid.uuid4()}.mp4")
    temp_path = os.path.join(TEMP_UPLOADS_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        # The robust, original method of getting frame count
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # We need to use the original script's method of getting frame count
        # as the new `detect_gesture_segments_sparse` relies on it.
        frame_segments = detect_gesture_segments_sparse(temp_path)
        print(f"Found {len(frame_segments)} segments (frames): {frame_segments}")

        time_segments = [{"start": s / fps, "end": e / fps} for s, e in frame_segments]
        
        processed_filename = f"processed_{uuid.uuid4()}.mp4"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        
        draw_segments_on_video(temp_path, processed_path, time_segments)
        video_url = f"/processed/{processed_filename}"

        return jsonify({"segments": time_segments, "processed_video_url": video_url}), 200

    except Exception as e:
        print(f"--- ERROR PROCESSING VIDEO ---")
        traceback.print_exc()
        return jsonify({"error": f"Failed to process video on server: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    mp_cpu.freeze_support()
    print("🚀 Starting Segmentation Server on http://127.0.0.1:8000")
    app.run(host='127.0.0.1', port=8000, debug=True, use_reloader=False)