from time import perf_counter

import cv2
import time
from ultralytics import YOLO
import pandas as pd
#from Attention_detection_recognition_withTimeMeasure import detect_end

# Load your custom YOLO model
model = YOLO('../yolo_detection_model/yolov11s-face.pt')

# Load video (use 0 for webcam or path to a file)
video_path = '../Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv'
cap = cv2.VideoCapture(video_path)
log_data = []
perf_log=[]

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
#out = cv2.VideoWriter('detection_only1280.mp4', fourcc, fps, (frame_width, frame_height))
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    start_time = time.perf_counter()  # ⏱️ Start timing

    ret, frame = cap.read()
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if not ret:
        break
    detect_start=perf_counter()
    # Run YOLO detection on GPU
    results = model.predict(frame, imgsz=1280, conf=0.3, device='cuda')[0]
    detect_ended=perf_counter()-detect_start
    t0=perf_counter()
    # Draw results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f'{model.names[cls]} {conf:.2f}'
        t3=perf_counter()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        write_detection_frame_time=perf_counter()-t3
        log_data.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            # "class": model.names[cls],
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "box_drawing_time": round(write_detection_frame_time* 1000, 1),
        })
    boxes_drawing_time=perf_counter()-t0
    # 🧮 Calculate and show FPS
    # fps = 1 / (end_time - start_time)
    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # cv2.putText(frame, f'Detection: {detect_ended * 1000:.1f}ms', (10, 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    # cv2.putText(frame, f'FrameTime: {(end_time - start_time) * 1000:.1f}ms', (10, 180),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    t1=perf_counter()
    #out.write(frame)
    # Display frame
    cv2.imshow('YOLO Detection', frame)
    writing_showing_frame_time=perf_counter()-t1
    end_time = time.perf_counter()
    frame_time = end_time - start_time
    perf_log.append({
        "frame": frame_idx,
        "timestamp_sec": round(timestamp, 2),
        "num_detections": len(results.boxes),
        "detection_time_ms": round(detect_ended * 1000, 1),
        "drawing_time_ms": round(boxes_drawing_time * 1000, 1),
        "showing_time_ms": round(writing_showing_frame_time * 1000, 1),
        "total_frame_time_ms": round(frame_time * 1000, 1)
    })
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Convert logs to DataFrames
df_detections = pd.DataFrame(log_data)
df_perf = pd.DataFrame(perf_log)

# Write both DataFrames to separate sheets in a single XLSX file
with pd.ExcelWriter("yolo_detection_only_1280.xlsx", engine="openpyxl") as writer:
    df_detections.to_excel(writer, sheet_name="Detections", index=False)
    df_perf.to_excel(writer, sheet_name="Performance", index=False)
#out.release()
cap.release()
cv2.destroyAllWindows()
