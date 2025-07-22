import cv2
import torch
import numpy as np
import pandas as pd
import time
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "../../Source Video/combined videos.mp4"
    output_path = "allModels&Tracking_Enhanced_attention_weights_16july2025_trained960p_on_960P_detect.xlsx.mp4"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    face_db_path = '../../AI_VID_face_db.pkl'

    # Temporary capture to get video properties
    temp_cap = cv2.VideoCapture(video_path)
    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
    temp_cap.release()

    capture = VideoCaptureHandler(video_path, output_path, frame_width, frame_height, fps)
    detector = FaceDetector(face_model_path, device)
    attention = AttentionDetector(attention_model_path, device)
    recognizer = FaceRecognizer(face_db_path, device)
    tracker = FaceTracker()
    display = DisplayManager(color_map)

    detect_log_data = []
    attention_log_data = []
    perf_log = []

    frame_count = 0
    while True:
        frame_start_time = time.perf_counter()
        # Frame reading timing
        t_read = time.perf_counter()
        ret, frame = capture.read()
        frame_read_time = time.perf_counter() - t_read
        if not ret:
            break
        frame_idx = capture.get_frame_idx()
        frame_count += 1
        timestamp = frame_count / fps

        # Detection timing
        t0 = time.perf_counter()
        detected = detector.detect(frame)
        detection_time = time.perf_counter() - t0
        boxes = [d['box'] for d in detected]
        centroids = [d['centroid'] for d in detected]
        for d in detected:
            x1, y1, x2, y2 = d['box']
            conf = d['conf']
            detect_log_data.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "confidence": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        # Attention timing
        t1 = time.perf_counter()
        attn_detections = attention.detect(frame)
        attention_time = time.perf_counter() - t1
        attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)
        for attn in attn_detections:
            ax1, ay1, ax2, ay2 = attn['box']
            att_conf = attn['conf']
            attention_log_data.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "confidence": round(att_conf, 3),
                "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2,
            })

        # Tracking & Recognition timing
        t2 = time.perf_counter()
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = tracker.update(detected_for_tracker, frame_count, recognizer, frame)
        tracking_time = time.perf_counter() - t2

        # Drawing timing
        t3 = time.perf_counter()
        frame = display.draw(frame, tracked_faces, attention_labels, attention.names)
        drawing_time = time.perf_counter() - t3

        # Writing timing
        t4 = time.perf_counter()
        # capture.write(frame)
        writing_time = time.perf_counter() - t4

        total_frame_time = time.perf_counter() - frame_start_time
        measured_sum = frame_read_time + detection_time + attention_time + tracking_time + drawing_time + writing_time
        other_time = total_frame_time - measured_sum

        perf_log.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "num_detections": len(boxes),
            "frame_read_time_ms": round(frame_read_time * 1000, 2),
            "detection_time_ms": round(detection_time * 1000, 2),
            "attention_time_ms": round(attention_time * 1000, 2),
            "tracking_time_ms": round(tracking_time * 1000, 2),
            "drawing_time_ms": round(drawing_time * 1000, 2),
            "writing_time_ms": round(writing_time * 1000, 2),
            "other_time_ms": round(other_time * 1000, 2),
            "total_frame_time_ms": round(total_frame_time * 1000, 2)
        })

        # Optionally print the timings for this frame
        print(f"Frame {frame_idx}: Read {frame_read_time*1000:.2f}ms | Detection {detection_time*1000:.2f}ms | Attention {attention_time*1000:.2f}ms | Tracking {tracking_time*1000:.2f}ms | Drawing {drawing_time*1000:.2f}ms | Writing {writing_time*1000:.2f}ms | Other {other_time*1000:.2f}ms | Total {total_frame_time*1000:.2f}ms")

        cv2.imshow("Tracked Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    df_detections = pd.DataFrame(detect_log_data)
    df_attention = pd.DataFrame(attention_log_data)
    df_perf = pd.DataFrame(perf_log)
    # with pd.ExcelWriter("allModels&Tracking_Enhanced_attention_weights_16july2025_trained960p_on_960P_detect.xlsx", engine="openpyxl") as writer:
    #     df_detections.to_excel(writer, sheet_name="Detections", index=False)
    #     df_attention.to_excel(writer, sheet_name="Attention", index=False)
    #     df_perf.to_excel(writer, sheet_name="Performance", index=False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
