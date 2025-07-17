from time import perf_counter

import cv2
import time
from ultralytics import YOLO
import pandas as pd
# Load your custom YOLO model
model = YOLO('../yolo_attention_model/old attention model Try V1.pt')

# Load video (use 0 for webcam or path to a file)
video_path = "../Source Video/combined videos.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))
log_data = []
perf_log=[]
# Define codec and output file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
out = cv2.VideoWriter('attention_weights_16july2025_AI_VIDEO_AttentionOnly_1280p_train_1280.mp4', fourcc, fps, (frame_width, frame_height))
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    start_time = time.perf_counter()  # ⏱️ Start timing
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    ret, frame = cap.read()
    if not ret:
        break
    attention_start=perf_counter()
    # Run YOLO detection on GPU
    results = model.predict(frame, imgsz=1280, conf=0.3, device='cuda')[0]
    attention_end=perf_counter()-attention_start
    # Draw results
    t0=perf_counter()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f'{model.names[cls]} {conf:.2f}'
        t3=perf_counter()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        write_detection_frame_time = perf_counter() - t3
        log_data.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            # "class": model.names[cls],
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "box_drawing_time": round(write_detection_frame_time * 1000, 1),
        })

    boxes_drawing_time=perf_counter()-t0
    # 🧮 Calculate and show FPS
    # fps = 1 / (end_time - start_time)
    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # cv2.putText(frame, f'Attention: {attention_end * 1000:.1f}ms', (10, 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    # cv2.putText(frame, f'FrameTime: {(end_time-start_time) * 1000:.1f}ms', (10, 180),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Display frame
    t1=perf_counter()
    cv2.imshow('YOLO Attention', frame)
    writing_showing_frame_time=perf_counter()-t1
    end_time = time.perf_counter()

    out.write(frame)
    perf_log.append({
        "frame": frame_idx,
        "timestamp_sec": round(timestamp, 2),
        "num_detections": len(results.boxes),
        "attention_time_ms": round(attention_end * 1000, 1),
        "drawing_time_ms": round(boxes_drawing_time * 1000, 1),
        "showing_time_ms": round(writing_showing_frame_time * 1000, 1),
        "total_frame_time_ms": round((end_time-start_time) * 1000, 1)
    })
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
df_detections = pd.DataFrame(log_data)
df_perf = pd.DataFrame(perf_log)
with pd.ExcelWriter("attention_weights_16july2025_AI_VIDEO_AttentionOnly_1280p_train_1280.xlsx", engine="openpyxl") as writer:
    df_detections.to_excel(writer, sheet_name="Detections", index=False)
    df_perf.to_excel(writer, sheet_name="Performance", index=False)
cap.release()
cv2.destroyAllWindows()
