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
from time import perf_counter

# Setup GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
face_model = YOLO('../yolo_detection_model/yolov11s-face.pt')  # face detection model
attention_model = YOLO('../yolo_attention_model/attention14june.pt')  # your custom attention model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
log_data = []
perf_log=[]
# Load face DB
with open('../face_db.pkl', 'rb') as f:
    face_db = pickle.load(f)

# Recognition cache
face_id_to_name = {}
face_id_to_conf = {}

# Color mapping for attention
color_map = {
    "attentive": (0, 255, 0),
    "inattentive": (0, 0, 255),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

# Padding utility
def get_padded_crop(frame, x1, y1, x2, y2, padding=80):
    h, w = frame.shape[:2]
    pad_x1 = max(0, x1 - padding)
    pad_y1 = max(0, y1 - padding)
    pad_x2 = min(w, x2 + padding)
    pad_y2 = min(h, y2 + padding)
    return frame[pad_y1:pad_y2, pad_x1:pad_x2]

def detect_faces_with_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and output file
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
    #out = cv2.VideoWriter('allModels&tracking.mp4', fourcc, fps, (frame_width, frame_height))


    frame_count = 0
    next_face_id = 0
    face_tracks = {}
    last_seen = {}
    max_distance = 40
    max_history = 20
    max_missing_frames = 10

    #cv2.namedWindow("Tracked Faces", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Tracked Faces", 1280, 720)

    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_count += 1
        timestamp = frame_count / fps
        t0 = perf_counter()

        results = face_model(frame, verbose=True, imgsz=1280)
        detection_model_time = perf_counter() - t0
        detections = results[0].boxes
        boxes = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append((x1, y1, x2, y2, conf, centroid))
            log_data.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                # "class": model.names[cls],
                "confidence": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,

            })

        used_ids = set()
        cumulative_detection_time = 0
        tracking_start = perf_counter()
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

            face_crop = frame[y1:y2, x1:x2]
            should_recognize = matched_id not in face_id_to_name or face_id_to_name[matched_id] == "Unknown"
            t2 = perf_counter()

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

            # === Attention classification ===
            recognition_time = perf_counter() - t2
            attention_label = "Unknown"
            t1 = perf_counter()

            try:
                t6=perf_counter()
                padded_crop = get_padded_crop(frame, x1, y1, x2, y2, padding=80)
                padded_time=perf_counter()-t6
                attn_results = attention_model.predict(padded_crop, imgsz=160, verbose=True)[0]
                if attn_results.boxes.shape[0] > 0:
                    cls_id = int(attn_results.boxes.cls[0])
                    attention_label = attention_model.names[cls_id]
            except:
                attention_label = "Error"
            attention_time = perf_counter() - t1
            cumulative_detection_time += attention_time
            # === Draw bounding box with color ===
            name = face_id_to_name.get(matched_id, "Unknown")
            conf = face_id_to_conf.get(matched_id, 1.0)
            label = f'{name} ({matched_id}) {conf:.2f} | {attention_label}'
            color = color_map.get(attention_label, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Cleanup old IDs
        to_delete = [fid for fid in last_seen if frame_count - last_seen[fid] > max_missing_frames]
        for fid in to_delete:
            face_tracks.pop(fid, None)
            last_seen.pop(fid, None)
            face_id_to_name.pop(fid, None)
            face_id_to_conf.pop(fid, None)
        tracking_end = perf_counter()-tracking_start
        # FPS display
        #fps_display = 1 / elapsed_time
        # cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.putText(frame, f'FaceDet: {detection_model_time * 1000:.1f}ms', (10, 90),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
        # cv2.putText(frame, f'Recog: {recognition_time * 1000:.1f}ms', (10, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        # cv2.putText(frame, f'Attention: {attention_time * 1000:.1f}ms', (10, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        # cv2.putText(frame, f'FrameTime: {elapsed_time * 1000:.1f}ms', (10, 180),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # cv2.putText(frame, f'cumulative attention time: {cumulative_detection_time * 1000:.1f}ms', (10, 220),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        #
        t3=perf_counter()
        cv2.imshow("Tracked Faces", frame)
        writing_showing_frame_time=perf_counter()-t3
        #out.write(frame)
        elapsed_time = time.perf_counter() - start_time
        perf_log.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "num_detections": len(results[0].boxes),
            "detection_time_ms": round(detection_model_time * 1000, 1),
            "attention_time_ms": round(attention_time * 1000, 1),
            "attention_padding_time_ms": round(padded_time * 1000, 1),
            "showing_time_ms": round(writing_showing_frame_time * 1000, 2),
            "tracking_time_ms": round(tracking_end * 1000, 1),
            "recognition_time_ms": round(recognition_time * 1000, 1),
            "total_frame_time_ms": round(elapsed_time * 1000, 1)
        })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #out.release()
    df_detections = pd.DataFrame(log_data)
    df_perf = pd.DataFrame(perf_log)
    with pd.ExcelWriter("allModels&Tracking_1280.xlsx", engine="openpyxl") as writer:
        df_detections.to_excel(writer, sheet_name="Detections", index=False)
        df_perf.to_excel(writer, sheet_name="Performance", index=False)
    cap.release()
    cv2.destroyAllWindows()


# Run
detect_faces_with_tracking("../Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv")
