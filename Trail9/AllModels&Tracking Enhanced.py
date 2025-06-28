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
face_model = YOLO('../yolo_detection_model/yolov11s-face.pt')
attention_model = YOLO('../yolo_attention_model/attention14june.pt')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
detect_log_data = []
attention_log_data = []
perf_log=[]
# Load face DB
with open('../face_db.pkl', 'rb') as f:
    face_db = pickle.load(f)

face_id_to_name = {}
face_id_to_conf = {}

color_map = {
    "attentive": (0, 255, 0),
    "inattentive": (0, 0, 255),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def detect_faces_with_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #out = cv2.VideoWriter('optimized_attention_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count, next_face_id = 0, 0
    face_tracks, last_seen = {}, {}
    max_distance, max_history, max_missing_frames = 40, 20, 10

    cv2.namedWindow("Tracked Faces", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracked Faces", 1280, 720)

    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_count += 1
        timestamp = frame_count / fps
        t0 = perf_counter()

        results = face_model(frame, verbose=False, imgsz=1280)
        detection_model_time = perf_counter() - t0

        detections = results[0].boxes

        boxes = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append((x1, y1, x2, y2, conf, centroid))
            detect_log_data.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                # "class": model.names[cls],
                "confidence": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,

            })

        # Run attention model once per frame
        t_attn = perf_counter()

        attn_results = attention_model(frame, verbose=False, imgsz=1280)[0].boxes
        attention_time = perf_counter() - t_attn

        attn_boxes = []
        for attn_box in attn_results:
            ax1, ay1, ax2, ay2 = map(int, attn_box.xyxy[0])
            attn_class = int(attn_box.cls[0])
            att_conf = float(attn_box.conf[0])
            attn_boxes.append(((ax1, ay1, ax2, ay2), attn_class))
            attention_log_data.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                # "class": model.names[cls],
                "confidence": round(att_conf, 3),
                "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2,

            })
        t1=perf_counter()
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

            face_crop = frame[y1:y2, x1:x2]
            should_recognize = matched_id not in face_id_to_name or face_id_to_name[matched_id] == "Unknown"
            t_recog = perf_counter()

            if face_crop.size != 0 and should_recognize:
                try:
                    face_resized = cv2.resize(face_crop, (160, 160))
                    face_tensor = torch.tensor(np.array(face_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        face_emb = facenet(face_tensor).squeeze().cpu().numpy()
                    best_match, best_score = "Unknown", 1.0
                    for name, emb in face_db.items():
                        dist = cosine(face_emb, emb)
                        if dist < best_score:
                            best_score, best_match = dist, name
                    if best_score < 0.65:
                        face_id_to_name[matched_id], face_id_to_conf[matched_id] = best_match, best_score
                    else:
                        face_id_to_name[matched_id], face_id_to_conf[matched_id] = "Unknown", best_score
                except:
                    face_id_to_name[matched_id], face_id_to_conf[matched_id] = "Error", 1.0
            recognition_time=perf_counter() - t_recog
            # Assign attention label by IoU
            attention_label = "inattentive"
            t2 = perf_counter()
            best_iou = 0
            for (ax1, ay1, ax2, ay2), cls in attn_boxes:
                iou = compute_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    attention_label = attention_model.names[cls]
            iou_time=perf_counter() - t2
            name = face_id_to_name.get(matched_id, "Unknown")
            conf = face_id_to_conf.get(matched_id, 1.0)
            label = f'{name} ({matched_id}) {conf:.2f} | {attention_label}'
            color = color_map.get(attention_label, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Remove old IDs
        to_delete = [fid for fid in last_seen if frame_count - last_seen[fid] > max_missing_frames]
        for fid in to_delete:
            face_tracks.pop(fid, None)
            last_seen.pop(fid, None)
            face_id_to_name.pop(fid, None)
            face_id_to_conf.pop(fid, None)
        tracking_time = perf_counter() - t1
       # fps_display = 1 / (time.perf_counter() - start_time)
        #cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # FPS and timing overlays
        # fps_display = 1 / elapsed_time
        #
        # cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #
        # cv2.putText(frame, f'FaceDet: {detection_model_time * 1000:.1f}ms', (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
        #
        # cv2.putText(frame, f'Recog: {recognition_time * 1000:.1f}ms', (10, 90),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        #
        # cv2.putText(frame, f'Attention: {attention_time * 1000:.1f}ms', (10, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        #
        # cv2.putText(frame, f'TotalFrame: {elapsed_time * 1000:.1f}ms', (10, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        #out.write(frame)
        t3=perf_counter()
        cv2.imshow("Tracked Faces", frame)
        writing_showing_frame_time=perf_counter()-t3
        elapsed_time = time.perf_counter() - start_time
        perf_log.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "num_detections": len(results[0].boxes),
            "detection_time_ms": round(detection_model_time * 1000, 1),
            "attention_time_ms": round(attention_time * 1000, 1),

            "showing_time_ms": round(writing_showing_frame_time * 1000, 2),
            "iou_time_ms": round(iou_time * 1000, 1),
            "tracking_time_ms": round(tracking_time * 1000, 1),
            "recognition_time_ms": round(recognition_time * 1000, 1),
            "total_frame_time_ms": round(elapsed_time * 1000, 1)
        })
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #out.release()
    cap.release()
    df_detections = pd.DataFrame(detect_log_data)
    df_attention = pd.DataFrame(attention_log_data)
    df_perf = pd.DataFrame(perf_log)
    with pd.ExcelWriter("allModels&Tracking_Enhanced_1280.xlsx", engine="openpyxl") as writer:
        df_detections.to_excel(writer, sheet_name="Detections", index=False)
        df_attention.to_excel(writer, sheet_name="Attention", index=False)

        df_perf.to_excel(writer, sheet_name="Performance", index=False)
    cv2.destroyAllWindows()

# Run
detect_faces_with_tracking("../Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv")
