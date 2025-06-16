import os
import cv2
import numpy as np
import mediapipe as mp
import multiprocessing as mp_cpu
import uuid
import traceback
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- 1. FLASK APPLICATION SETUP ---

app = Flask(__name__)
CORS(app)
TEMP_UPLOADS_FOLDER = "temp_uploads"
os.makedirs(TEMP_UPLOADS_FOLDER, exist_ok=True)

# --- 2. GESTURE SEGMENTATION LOGIC ---

SCALE = 0.5
FLOW_TH_PIX = 45.0
SPARSE_POINTS = 200
SPARSE_QUALITY = 0.01
POSE_IVL = 2
POSE_TH = 0.18
IDLE_SPLIT = 3
MIN_LEN = 50
MAX_LEN = 80

def sparse_flow_mag(prev_g: np.ndarray, next_g: np.ndarray) -> float:
    pts = cv2.goodFeaturesToTrack(prev_g, maxCorners=SPARSE_POINTS, qualityLevel=SPARSE_QUALITY, minDistance=10, blockSize=7)
    if pts is None or len(pts) < 10: return 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, next_g, pts, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nxt is None or st is None: return 0.0
    good_old, good_new = pts[st == 1], nxt[st == 1]
    if len(good_old) < 5: return 0.0
    mags = np.linalg.norm(good_new - good_old, axis=1)
    mags = mags[mags < np.percentile(mags, 90)]
    return float(np.mean(mags)) if mags.size else 0.0

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

def detect_gesture_segments_sparse(src: str) -> List[Tuple[int, int]]:
    cap = cv2.VideoCapture(src)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if tot < 2: return []
    idxs = list(range(tot - 1))
    n_workers = max(1, mp_cpu.cpu_count() - 1)
    batch = max(1, len(idxs) // n_workers)
    batches = [idxs[i:i + batch] for i in range(0, len(idxs), batch)]

    flow_vals = [0.0] * tot
    with ProcessPoolExecutor(n_workers) as ex:
        for res in ex.map(process_batch, batches, [src] * len(batches), [SCALE] * len(batches)):
            for idx, mag in res:
                if idx < tot: flow_vals[idx] = mag

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
            flag = wrist_up(res.pose_landmarks.landmark) if res.pose_landmarks else False
            for j in range(i, min(len(frames), i + POSE_IVL)):
                pose_raised[j] = flag

    segs, start, idle = [], None, 0
    for idx in range(len(frames)):
        gest = (flow_vals[idx] > FLOW_TH_PIX) or pose_raised[idx]
        if gest:
            idle = 0
            if start is None: start = idx
        else:
            idle += 1
            if start is not None and idle >= IDLE_SPLIT:
                segs.append((start, idx - IDLE_SPLIT + 1)); start = None
        if start is not None and idx - start + 1 >= MAX_LEN:
            segs.append((start, idx + 1)); start, idle = None, 0

    if start is not None:
        segs.append((start, len(frames)))

    return [(s, e) for s, e in segs if e - s >= MIN_LEN]


# --- 3. API ENDPOINT ---

@app.route('/segment', methods=['POST'])
def auto_segment_video():
    if 'video' not in request.files: return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    original_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(original_filename)[1].lower()
    
    unique_basename = f"temp_{uuid.uuid4()}"
    
    temp_path = os.path.join(TEMP_UPLOADS_FOLDER, f"{unique_basename}{file_extension}")
    file.save(temp_path)

    path_to_process = temp_path
    converted_path = None

    if file_extension != ".mp4":
        print(f"File is not MP4 ({file_extension}). Converting to MP4...")
        converted_path = os.path.join(TEMP_UPLOADS_FOLDER, f"{unique_basename}.mp4")
        
        try:
            command = [
                'ffmpeg',
                '-i', temp_path,
                '-an',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-y',
                converted_path
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully converted {temp_path} to {converted_path}")
            path_to_process = converted_path
        except subprocess.CalledProcessError as e:
            print(f"--- FFMPEG CONVERSION ERROR ---")
            print(f"FFmpeg stdout: {e.stdout}")
            print(f"FFmpeg stderr: {e.stderr}")
            return jsonify({"error": f"Failed to convert video on server."}), 500
        except FileNotFoundError:
            print(f"--- FFMPEG NOT FOUND ---")
            return jsonify({"error": "FFmpeg is not installed or not found in system PATH."}), 500
    
    try:
        cap = cv2.VideoCapture(path_to_process)
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open the video file: {path_to_process}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        frame_segments = detect_gesture_segments_sparse(path_to_process)
        print(f"Found {len(frame_segments)} segments (frames): {frame_segments}")

        time_segments = [{"start": s / fps, "end": e / fps} for s, e in frame_segments]

        return jsonify({"segments": time_segments}), 200

    except Exception as e:
        print(f"--- ERROR PROCESSING VIDEO ---")
        traceback.print_exc()
        return jsonify({"error": f"Failed to process video on server: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)

if __name__ == '__main__':
    mp_cpu.freeze_support()
    print(" [######] Video (gesture) segmentation server")
    print(" [######] Starting server on http://127.0.0.1:8000")
    app.run(host='127.0.0.1', port=8000, debug=True, use_reloader=False)