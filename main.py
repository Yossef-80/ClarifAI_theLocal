import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from scipy.spatial.distance import cosine
import pickle

# Load the face database from file
with open('face_db.pkl', 'rb') as f:
    face_db = pickle.load(f)

# Optional: check how many people are in the database
print(f"✅ Loaded Face DB with {len(face_db)} people: {list(face_db.keys())}")

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device =", device)

# Load models
face_model = YOLO('yolo_detection_model/yolov11s-face.pt').to(device)
attention_model = YOLO('yolo_attention_model/attention14june.pt').to(device)
recognition_model = InceptionResnetV1(classify=False).eval().to(device)
recognition_model.load_state_dict(torch.load('face_embedding_vggface2.pth', map_location=device), strict=False)
# Face DB already built
print(f"✅ Face DB loaded with {len(face_db)} people")

# Color coding for attention
attention_labels = ["attentive", "phone", "unattentive"]
attention_colors = {
    "attentive": (0, 255, 0),
    "phone": (0, 255, 255),
    "unattentive": (0, 0, 255),
    "Unknown": (128, 128, 128)
}

# Recognition function
def recognize_face(cropped_face):
    try:
        img = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)).resize((160, 160))
        img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            emb = recognition_model(img_tensor).cpu().numpy().squeeze()

        min_dist = 1.0
        identity = "Unknown"
        for name, db_emb in face_db.items():
            dist = cosine(emb, db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
        return identity if min_dist < 0.65 else "Unknown"
    except:
        return "Error"

# Main processing
def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    data_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        # Face detection
        face_results = face_model(frame, verbose=False, imgsz=1280)[0]

        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue
            h, w = face_crop.shape[:2]
            pad_percent = 0.15  # 10% padding

            # Calculate padding amounts
            pad_top = int(h * pad_percent)
            pad_bottom = int(h * pad_percent)
            pad_left = int(w * pad_percent)
            pad_right = int(w * pad_percent)

            # Apply padding
            face_crop_padded = cv2.copyMakeBorder(
                face_crop,
                pad_top, pad_bottom,
                pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # Black padding
            )


            # Face recognition per frame
            name = recognize_face(face_crop)

            # Attention detection
            attn_label = "unattentive"
            attn_result = attention_model(face_crop_padded, conf=0.2, imgsz=160, verbose=False)[0]
            if attn_result.boxes is not None and len(attn_result.boxes) > 0:
                best_box = max(attn_result.boxes, key=lambda b: b.conf[0])
                class_id = int(best_box.cls[0])
                attn_label = attention_labels[class_id]

            # Color based on attention
            color = attention_colors.get(attn_label, (255, 255, 255))

            # Draw on frame
            label = f"{name} | {attn_label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log
            data_log.append([timestamp, name, attn_label, round(conf, 2), x1, y1, x2, y2])

        if output_path:
            out.write(frame)

    cap.release()
    if output_path:
        out.release()

    # Save log
    df = pd.DataFrame(data_log, columns=["Timestamp", "Name", "Attention", "Confidence", "X1", "Y1", "X2", "Y2"])
    df.to_csv("/content/drive/MyDrive/AI_work/14june/Kindergarten Reader Theater_output_framewise_recognition_attention.csv", index=False)
    print("✅ Video processing done and CSV saved.")

# Run it
video_path = "Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv"
output_path = "/content/drive/MyDrive/AI_work/14june/Kindergarten Reader Theater_output_colored_attention_recognition.mp4"


def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    data_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        # Face detection
        face_results = face_model(frame, verbose=False)[0]

        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop and pad face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            h, w = face_crop.shape[:2]
            pad = lambda x: int(x * 0.15)
            face_crop_padded = cv2.copyMakeBorder(
                face_crop,
                pad(h), pad(h), pad(w), pad(w),
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Face recognition
            name = recognize_face(face_crop)

            # Attention classification
            attn_label = "unattentive"
            attn_result = attention_model(face_crop_padded, conf=0.2, imgsz=160, verbose=False)[0]
            if attn_result.boxes is not None and len(attn_result.boxes) > 0:
                best_box = max(attn_result.boxes, key=lambda b: b.conf[0])
                class_id = int(best_box.cls[0])
                attn_label = attention_labels[class_id]

            # Draw on frame
            color = attention_colors.get(attn_label, (255, 255, 255))
            label = f"{name} | {attn_label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save info to log
            data_log.append([timestamp, name, attn_label, round(conf, 2), x1, y1, x2, y2])

        # 👁️ Show live stream
        cv2.imshow("Live Attention & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Optional: Save logs
    df = pd.DataFrame(data_log, columns=["Timestamp", "Name", "Attention", "Confidence", "X1", "Y1", "X2", "Y2"])
    df.to_csv("framewise_log.csv", index=False)
    print("✅ Finished live processing and saved log.")


process_video_stream(video_path)
