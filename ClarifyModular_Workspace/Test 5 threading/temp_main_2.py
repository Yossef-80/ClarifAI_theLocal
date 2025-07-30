import base64
import cv2
import torch
import numpy as np
import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager

import socketio

# Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# FastAPI app
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Combine both with ASGI app
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

client_tasks = {}

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    task = asyncio.create_task(stream_video(sid))
    client_tasks[sid] = task

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)
    task = client_tasks.pop(sid, None)
    if task:
        task.cancel()

async def stream_video(sid):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        video_path = "../../Source Video/combined videos.mp4"
        face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
        attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
        face_db_path = '../../AI_VID_face_db.pkl'

        capture = VideoCaptureHandler(video_path)
        frame_width, frame_height = capture.get_frame_size()
        fps = capture.get_fps()
        detector = FaceDetector(face_model_path, device)
        attention = AttentionDetector(attention_model_path, device)
        recognizer = FaceRecognizer(face_db_path, device)
        tracker = FaceTracker()
        display = DisplayManager(color_map)

        frame_count = 0
        attn_stats = []
        frame_times = []
        while True:
            frame_start = time.perf_counter()
            ret, frame = capture.read()
            if not ret:
                break
            frame_idx = capture.get_frame_idx()
            frame_count += 1
            timestamp = frame_count / fps

            detected = detector.detect(frame)
            boxes = [d['box'] for d in detected]
            centroids = [d['centroid'] for d in detected]

            attn_detections = attention.detect(frame)
            attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)

            detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
            tracked_faces = tracker.update(detected_for_tracker, frame_count, recognizer, frame)

            frame = display.draw(frame, tracked_faces, attention_labels, attention.names)

            # Compute metrics for this frame
            attentive_count = sum(1 for label in attention_labels if label == 'attentive')
            total_count = len(attention_labels)
            attn_stats.append((attentive_count, total_count))

            frame_end = time.perf_counter()
            frame_times.append(frame_end - frame_start)

            # Only send metrics and alert every second
            if frame_count % fps == 0:
                if attn_stats:
                    last_attn, last_total = attn_stats[-1]
                    percent_val = int((last_attn / last_total) * 100) if last_total > 0 else 0
                    # Print frame time and FPS
                    if frame_times:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                        print(f"[PERF] Avg frame time: {avg_frame_time*1000:.2f} ms | FPS: {current_fps:.2f}")
                    frame_times = []
                    # Emit metrics
                    await sio.emit('classroom_metrics', {
                        "attention": percent_val,
                        "comprehension": percent_val,  # or your actual value
                        "active": last_attn
                    }, to=sid)
                    # Emit alert
                    if percent_val <= 50:
                        alert_type = "danger"
                        alert_msg = f"ALERT: Only {last_attn} of {last_total} students are attentive! ({percent_val}%)"
                    elif percent_val <= 70:
                        alert_type = "warning"
                        alert_msg = f"Warning: {last_attn} of {last_total} students are attentive. ({percent_val}%)"
                    else:
                        alert_type = "success"
                        alert_msg = f"Good: {last_attn} of {last_total} students are attentive. ({percent_val}%)"
                    await sio.emit('alert', {
                        "type": alert_type,
                        "time": time.strftime("%H:%M:%S"),
                        "message": alert_msg
                    }, to=sid)
                attn_stats = []

            # Only send every 5th frame (video)
            if frame_count % 5 == 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # Lower quality for speed
                _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                await sio.emit('video_frame', {"frame": b64}, to=sid)

            #await asyncio.sleep(1 / fps)

    except asyncio.CancelledError:
        print(f"Stream for {sid} cancelled.")
    finally:
        capture.release()
