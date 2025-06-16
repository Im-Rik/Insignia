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
import groq
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- 1. SETUP ---

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app)

WEIGHTS_PATH = "sign_recognizer_best.pth"
TARGET_FRAMES = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_FOLDER = "temp_processing"
os.makedirs(TEMP_FOLDER, exist_ok=True)
mp_holistic = mp.solutions.holistic

try:
    groq_client = groq.Groq(api_key="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")
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Groq client: {e}. Make sure GROQ_API_KEY is set.")
    groq_client = None

job_queues = {}

# --- 2. SIGN RECOGNITION MODEL ---

def extract_keypoints(res):
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]).flatten() if res.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

class SignRecognizer(nn.Module):
    def __init__(self, in_dim=1662, n_cls=15, hidden=128, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_cls))
    
    def forward(self, x):
        h, _ = self.lstm(x)
        alpha = torch.softmax(self.attn(h).squeeze(-1), 1)
        ctx = (h * alpha.unsqueeze(-1)).sum(1)
        return self.head(self.norm(ctx))

print("Loading model and weights...")
try:
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = SignRecognizer(1662, len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded successfully: {len(classes)} classes on {str(DEVICE).upper()} device.")
except Exception as e:
    print(f"Critical error loading model: {e}")
    model, classes = None, []

# --- 3. BACKGROUND VIDEO PROCESSING ---

def process_video_in_background(main_video_path, segments, job_id):
    print(f"Processing job {job_id}")
    q = job_queues.get(job_id)
    if not q:
        print(f"Could not find job queue for {job_id}. Aborting thread.")
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
                    (
                        ffmpeg
                        .input(main_video_path, ss=start_time, to=end_time)
                        .output(clip_path, loglevel="quiet", **{'c:v': 'libvpx-vp9', 'an': None})
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    )

                    all_keypoints = []
                    cap = cv2.VideoCapture(clip_path)
                    while True:
                        ok, frame = cap.read()
                        if not ok: break
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        all_keypoints.append(extract_keypoints(results))
                    cap.release()

                    if not all_keypoints:
                        raise RuntimeError("Could not extract any keypoints from the segment clip.")

                    indices = np.linspace(0, len(all_keypoints) - 1, TARGET_FRAMES, dtype=int) if len(all_keypoints) > TARGET_FRAMES else range(len(all_keypoints))
                    final_keypoints_list = [all_keypoints[i] for i in indices]

                    if len(final_keypoints_list) < TARGET_FRAMES:
                        padding = np.zeros((TARGET_FRAMES - len(final_keypoints_list), 1662), dtype=np.float32)
                        final_keypoints = np.vstack([np.array(final_keypoints_list, dtype=np.float32), padding])
                    else:
                        final_keypoints = np.array(final_keypoints_list, dtype=np.float32)

                    seq = torch.from_numpy(final_keypoints).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(seq), 1)[0].cpu().numpy()
                    
                    prediction = classes[int(probs.argmax())]
                    confidence = float(probs.max())
                    result_data = {'segmentId': segment_id, 'prediction': prediction, 'confidence': confidence}
                    
                    q.put(result_data)
                    print(f"  [Segment {i+1}/{len(segments)}] Prediction: '{prediction}'. Pushed to queue.")

                except Exception as e:
                    error_message = f"Failed to process segment {segment_id}: {e}"
                    print(f"  [ERROR] {error_message}")
                    traceback.print_exc()
                    q.put({'error': str(e), 'segmentId': segment_id})
                
                finally:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

    finally:
        q.put(None)
        if os.path.exists(main_video_path):
            os.remove(main_video_path)
            print(f"Main video deleted: {os.path.basename(main_video_path)}")
        print(f"Finished job {job_id}")


# --- 4. API ENDPOINTS ---

@app.route('/classify', methods=['POST'])
def handle_classification_request():
    if 'video' not in request.files: return jsonify({"error": "No video file provided"}), 400
    if 'segments' not in request.form: return jsonify({"error": "No segments data provided"}), 400

    video_file = request.files['video']
    segments = json.loads(request.form['segments'])
    
    filename = f"{uuid.uuid4()}.webm"
    main_video_path = os.path.join(TEMP_FOLDER, filename)
    video_file.save(main_video_path)
    
    job_id = str(uuid.uuid4())
    job_queues[job_id] = queue.Queue()

    thread = threading.Thread(target=process_video_in_background, args=(main_video_path, segments, job_id))
    thread.start()
    
    print(f"Job '{job_id}' created for {len(segments)} segments.")
    return jsonify({"job_id": job_id}), 202

@app.route('/stream/<job_id>')
def stream_results(job_id):
    q = job_queues.get(job_id)
    if not q:
        return jsonify({"error": "Invalid or expired job ID"}), 404

    def generate():
        print(f"Client connected to stream for job {job_id}.")
        while True:
            result = q.get()
            if result is None:
                yield "event: complete\ndata: {}\n\n"
                break
            
            yield f"data: {json.dumps(result)}\n\n"
        
        if job_id in job_queues:
            del job_queues[job_id]
        print(f"Stream closed for job {job_id}. Queue deleted.")

    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence_endpoint():
    if not groq_client:
        return jsonify({"error": "Groq client is not initialized. Check server logs."}), 503

    try:
        data = request.get_json()
        glosses = data.get('glosses')
        context_sentences = data.get('context_sentences')

        if not isinstance(glosses, list) or not isinstance(context_sentences, list):
            return jsonify({"error": "Inputs 'glosses' and 'context_sentences' must be lists."}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON format in request body."}), 400

    print(f"LLM request received. Glosses: {glosses}, Context: {context_sentences}")

    system_prompt = """
    You are a medical communication assistant designed to help deaf and mute individuals communicate their health concerns effectively. Your task is to generate clear, contextually appropriate sentences based on gesture inputs and emoji selections.

## Context
You are part of a web application that recognizes Indian Sign Language (ISL) gestures and allows users to select symptom-related emojis. Your role is to combine these inputs into coherent sentences that can help medical professionals understand the patient's condition.

## Available Inputs

### Gesture Words (from ISL recognition):
"Baby", "Bill", "Boy", "Brother", "cold", "Daughter", "Deaf", "Doctor", "dry", "Family", "Father", "Girl", "healthy", "Heavy", "High", "Hospital", "Medicine", "Money", "Mother", "Patient", "sick", "Sister", "Son", "Weak"

### Symptom Emojis:
"headache", "backpain", "neck pain", "stomach ache", "running nose", "watery nose", "sick", "tired", "dizziness", "dry mouth"

## Your Task
Generate a natural, grammatically correct sentence that:
1. Incorporates the recognized gesture words meaningfully
2. Includes the selected emoji symptoms appropriately
3. Creates a clear message suitable for medical communication
4. Maintains a respectful and professional tone

## Input Format
You will receive input in this format:
{
  "gestures": ["list", "of", "recognized", "words"],
  "emojis": ["list", "of", "selected", "symptoms"]
}

## Output Requirements
- Generate ONE clear sentence that combines the inputs logically
- If referring to family members, use them as the subject when appropriate
- Always prioritize medical clarity over complex grammar
- If the combination seems unclear, create the most medically relevant interpretation
- ## Important Notes
- If only gestures are provided without emojis, still create a meaningful medical sentence
- If only emojis are provided, create a sentence describing the symptoms
- Always assume the communication is for medical purposes
- Keep sentences simple and direct for clear understanding
- Your response MUST NOT contain any explanations, conversational text, or markdown.
- Output ONLY the final sentence.
- Do not wrap the sentence in quotation marks.
"""
    
    user_content = f"Glosses: {json.dumps(glosses)}\nContext: {json.dumps(context_sentences)}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=150,
        )
        generated_sentence = chat_completion.choices[0].message.content.strip()
        
        if generated_sentence.startswith('"') and generated_sentence.endswith('"'):
            generated_sentence = generated_sentence[1:-1]

        print(f"Groq response: '{generated_sentence}'")
        
        return jsonify({"generated_sentence": generated_sentence})

    except Exception as e:
        print(f"Groq API call failed: {e}")
        return jsonify({"error": "Failed to generate sentence from LLM."}), 500

if __name__ == '__main__':
    print(" [######] Classification and LLM Server")
    print(" [######] Starting server on http://0.0.0.0:5020")
    app.run(host='0.0.0.0', port=5020, threaded=True)