from time import perf_counter

import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle
import time

# Setup GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
face_model = YOLO('yolo_detection_model/yolov11s-face.pt')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load face DB
with open('face_db.pkl', 'rb') as f:
    face_db = pickle.load(f)

# Face recognition cache
face_id_to_name = {}
face_id_to_conf = {}
log_data = []

def detect_faces_with_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    next_face_id = 0
    face_tracks = {}
    last_seen = {}
    max_distance = 40
    max_history = 20
    max_missing_frames = 10

    cv2.namedWindow("Tracked Faces", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracked Faces", 1280, 720)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('Detection1280_Recognition_Tracking_colab.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps
        t0=time.time()
        results = face_model(frame, verbose=False, imgsz=1280)
        detect_time=time.time()-t0
        detections = results[0].boxes
        boxes = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append((x1, y1, x2, y2, conf, centroid))

        used_ids = set()
        for x1, y1, x2, y2, conf, centroid in boxes:
            matched_id = None
            min_distance = float('inf')

            for fid, history in face_tracks.items():
                if not history or fid in used_ids:
                    continue
                prev_centroid = history[-1]
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < min_distance and dist < max_distance:
                    min_distance = dist
                    matched_id = fid

            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1

            used_ids.add(matched_id)

            if matched_id not in face_tracks:
                face_tracks[matched_id] = deque(maxlen=max_history)
            face_tracks[matched_id].append(centroid)
            last_seen[matched_id] = frame_count

            # Only recognize if unknown or first time
            face_crop = frame[y1:y2, x1:x2]
            should_recognize = matched_id not in face_id_to_name or face_id_to_name[matched_id] == "Unknown"
            t1=perf_counter()
            if face_crop.size != 0 and should_recognize:
                try:
                    face_resized = cv2.resize(face_crop, (160, 160))
                    face_tensor = torch.tensor(np.array(face_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

                    with torch.no_grad():
                        face_emb = facenet(face_tensor).squeeze().cpu().numpy()

                    best_match = "Unknown"
                    best_score = 1.0

                    for name, emb in face_db.items():
                        dist = cosine(face_emb, emb)
                        if dist < best_score:
                            best_score = dist
                            best_match = name

                    if best_score < 0.65:
                        face_id_to_name[matched_id] = best_match
                        face_id_to_conf[matched_id] = best_score
                    else:
                        face_id_to_name[matched_id] = "Unknown"
                        face_id_to_conf[matched_id] = best_score

                except:
                    face_id_to_name[matched_id] = "Error"
                    face_id_to_conf[matched_id] = 1.0
            recog_time = perf_counter() - t1
            name = face_id_to_name.get(matched_id, "Unknown")
            conf = face_id_to_conf.get(matched_id, 1.0)
            label = f'{name} ({matched_id}) {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Cleanup old IDs
        to_delete = [fid for fid in last_seen if frame_count - last_seen[fid] > max_missing_frames]
        for fid in to_delete:
            face_tracks.pop(fid, None)
            last_seen.pop(fid, None)
            face_id_to_name.pop(fid, None)
            face_id_to_conf.pop(fid, None)

        # FPS
        elapsed_time = time.perf_counter() - start_time
        fps = 1 / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f'FaceDet: {detect_time * 1000:.1f}ms', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        cv2.putText(frame, f'Recog: {recog_time * 1000:.1f}ms', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)



        cv2.putText(frame, f'TotalFrame: {elapsed_time * 1000:.1f}ms', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out.write(frame)
        cv2.imshow("Tracked Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()


# Run it
detect_faces_with_tracking("Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv")
