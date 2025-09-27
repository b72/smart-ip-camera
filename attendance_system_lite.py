import cv2
import threading
import queue
import os
import requests

# Set the INSIGHTFACE_HOME environment variable
os.environ["INSIGHTFACE_HOME"] = os.path.abspath("models")

from insightface.app import FaceAnalysis
import numpy as np

from datetime import datetime

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

API_ENDPOINT = "http://127.0.0.1:8000/incident"
EMPLOYEES_DIR = "employees"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
MODEL_PATH = os.getenv("MODEL_PATH", "models/buffalo_sc")

# Initialize face detector
face_app = FaceAnalysis(
            name= os.getenv("MODEL_NAME"), 
            providers= [os.getenv("EXEC_PROVIDER")],
            model_path= MODEL_PATH
            )
face_app.prepare(ctx_id=0, det_size=(800, 800))

# Load employee embeddings
employee_embeddings = {}
for emp_id in os.listdir(EMPLOYEES_DIR):
    emp_folder = os.path.join(EMPLOYEES_DIR, emp_id)
    if not os.path.isdir(emp_folder):
        continue
    for img_name in os.listdir(emp_folder):
        img_path = os.path.join(emp_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_app.get(img)
        if faces:
            embedding = faces[0].embedding
            employee_embeddings.setdefault(emp_id, []).append(embedding)

# Average embeddings per employee
for emp_id in employee_embeddings:
    employee_embeddings[emp_id] = np.mean(employee_embeddings[emp_id], axis=0)

cap = cv2.VideoCapture(os.getenv("VIDEO_SOURCE"))

frame_queue = queue.Queue(maxsize=1)
last_faces = []
lock = threading.Lock()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def send_to_api(data):
    try:
        requests.post(API_ENDPOINT, json=data, timeout=3)
    except Exception as e:
        print(f"API error: {e}")

def detect_faces():
    global last_faces
    while True:
        frame = frame_queue.get()
        faces = face_app.get(frame)
        results = []
        for face in faces:
            match_id = None
            match_sim = 0
            for emp_id, emp_emb in employee_embeddings.items():
                sim = cosine_similarity(face.embedding, emp_emb)
                if sim > SIMILARITY_THRESHOLD and sim > match_sim:
                    match_id = emp_id
                    match_sim = sim
            if match_id:
                data = {
                    "employee_id": match_id,
                    "timestamp": str(datetime.now()),  # <-- change here
                    "similarity": float(match_sim),
                    "image_path": ""
                }
                threading.Thread(target=send_to_api, args=(data,), daemon=True).start()
            results.append(face.bbox.astype(int))
        with lock:
            last_faces = results

# Start detection thread
threading.Thread(target=detect_faces, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if frame_queue.empty():
        frame_queue.put(frame.copy())
    with lock:
        for bbox in last_faces:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()