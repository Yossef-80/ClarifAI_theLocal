import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from scipy.spatial.distance import cosine
import time
import pickle

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_model = YOLO('yolo_detection_model/yolov11s-face.pt')
attention_model = YOLO('yolo_attention_model/attention14june.pt')
recognition_model = InceptionResnetV1(classify=False).eval().to(device)
recognition_model.load_state_dict(torch.load('face_embedding_vggface2.pth', map_location=device), strict=False)

# Load face DB
with open('face_db.pkl', 'rb') as f:
    face_db = pickle.load(f)
print(f"✅ Loaded Face DB with {len(face_db)} people: {list(face_db.keys())}")
log_data = []

# Recognition function
def recognize_face_fast(cropped_face):
    try:
        img = cv2.resize(cropped_face, (160, 160))
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = img.to(device)
        with torch.no_grad():
            emb = recognition_model(img).cpu().numpy().squeeze()

        identity, min_dist = "Unknown", 1.0
        for name, db_emb in face_db.items():
            dist = cosine(emb, db_emb)
            if dist < min_dist:
                min_dist, identity = dist, name
        return identity if min_dist < 0.65 else "Unknown"
    except:
        return "Error"

# Attention labels
attention_labels = ["attentive", "phone", "unattentive"]
attention_colors = {
    "attentive": (0, 255, 0),
    "phone": (0, 255, 255),
    "unattentive": (0, 0, 255),
    "Unknown": (128, 128, 128)
}

# Start video
cap = cv2.VideoCapture('Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv')
while True:
    start_total = time.time()
    read_start = time.time()
    ret, frame = cap.read()
    read_end = time.time()
    if not ret:
        break

    # Face detection
    detect_start = time.time()
    face_results = face_model.predict(frame, imgsz=640, conf=0.3, device=device, verbose=False)[0]
    detect_end = time.time()

    embed_total, attn_total, draw_total = 0, 0, 0
    if face_results.boxes:
        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            face_crop = frame[y1:y2, x1:x2]

            # Recognition
            start_embed = time.time()
            name = recognize_face_fast(face_crop)
            end_embed = time.time()
            embed_total += end_embed - start_embed

            # Attention
            start_attn = time.time()
            attn_result = attention_model.predict(face_crop, imgsz=160, device=device, verbose=False)[0]
            attn_label = "unattentive"
            if attn_result.boxes:
                best_box = max(attn_result.boxes, key=lambda b: b.conf[0])
                attn_class = int(best_box.cls[0])
                attn_label = attention_labels[attn_class]
            end_attn = time.time()
            attn_total += end_attn - start_attn

            # Draw
            start_draw = time.time()
            color = attention_colors.get(attn_label, (255, 255, 255))
            label = f"{name} | {attn_label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            end_draw = time.time()
            draw_total += end_draw - start_draw

    total_end = time.time()
    total_time = total_end - start_total
    fps = 1 / total_time if total_time else 0

    print(f"\u23f1 Total: {total_time:.3f}s | Read: {read_end - read_start:.3f}s | Detect: {detect_end - detect_start:.3f}s | "
          f"Embed: {embed_total:.3f}s | Attn: {attn_total:.3f}s | Draw: {draw_total:.3f}s | FPS: {fps:.2f}")

    # Optimize display
    small_disp = cv2.resize(frame, (640, 360))
    cv2.imshow("Real-Time Attention + Recognition", small_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
