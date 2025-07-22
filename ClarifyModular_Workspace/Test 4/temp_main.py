import eventlet
eventlet.monkey_patch()

import cv2
import torch
import numpy as np
import time
import base64
from flask import Flask, request, Response
from flask_socketio import SocketIO, emit
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager
import threading

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Add a set to track connected Socket.IO clients
connected_clients = set()

def encode_frame(frame):
    max_height = 480
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

streaming_task = None

# Target max FPS for streaming (adjust as needed)
MAX_FPS = 10
FRAME_INTERVAL = 1.0 / MAX_FPS

def video_processing(sid):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "../../Source Video/combined videos.mp4"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    face_db_path = '../../AI_VID_face_db.pkl'

    capture = VideoCaptureHandler(video_path, None, 0, 0, 0)
    fps = capture.get_fps()
    detector = FaceDetector(face_model_path, device)
    attention = AttentionDetector(attention_model_path, device)
    recognizer = FaceRecognizer(face_db_path, device)
    tracker = FaceTracker()
    display = DisplayManager(color_map)

    frame_count = 0
    last_emit_time = time.time()
    attn_stats = []
    last_metrics_emit = time.time()

    while True:
        loop_start = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        frame_idx = capture.get_frame_idx()
        frame_count += 1
        timestamp = frame_count / fps

        detected = detector.detect(frame)
        boxes = [d['box'] for d in detected]
        attn_detections = attention.detect(frame)
        attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = tracker.update(detected_for_tracker, frame_count, recognizer, frame)
        frame = display.draw(frame, tracked_faces, attention_labels, attention.names)

        # Count attentiveness for stats
        attn_this_frame = sum(1 for label in attention_labels if label == 'attentive')
        total_this_frame = len(attention_labels)
        attn_stats.append((attn_this_frame, total_this_frame))

        # Emit video frame
        frame_b64 = encode_frame(frame)
        socketio.emit('video_frame', frame_b64, room=sid)

        # Every second, emit attentiveness stats
        now = time.time()
        if now - last_emit_time >= 1.0:
            sum_attn = sum(x for x, _ in attn_stats)
            sum_total = sum(y for _, y in attn_stats)
            percent = (sum_attn / sum_total * 100) if sum_total > 0 else 0
            data = {
                "time": time.strftime("%H:%M:%S", time.gmtime(timestamp)),
                "attentive": sum_attn,
                "total": sum_total,
                "percent": percent
            }
            socketio.emit('attentiveness', data, room=sid)
            attn_stats = []
            last_emit_time = now

        # --- Classroom Metrics ---
        attn_this_frame = sum(1 for label in attention_labels if label == 'attentive')
        total_this_frame = len(attention_labels)
        attn_stats.append((attn_this_frame, total_this_frame))
        now = time.time()
        if now - last_metrics_emit >= 1.0:
            sum_attn = sum(x for x, _ in attn_stats)
            sum_total = sum(y for _, y in attn_stats)
            attention_rate = int((sum_attn / sum_total * 100) if sum_total > 0 else 0)
            comprehension_rate = attention_rate  # Placeholder: use same as attention
            active_students = sum_attn  # Number of attentive students in the last second
            metrics = {
                'attention': attention_rate,
                'comprehension': comprehension_rate,
                'active': active_students
            }
            # Emit to all connected clients
            for sid in list(connected_clients):
                socketio.emit('classroom_metrics', metrics, room=sid)
            attn_stats = []
            last_metrics_emit = now

        # Frame pacing: sleep to target max FPS
        elapsed = time.time() - loop_start
        if elapsed < FRAME_INTERVAL:
            eventlet.sleep(FRAME_INTERVAL - elapsed)
        else:
            eventlet.sleep(0)  # Yield to eventlet

    capture.release()
    cv2.destroyAllWindows()

# MJPEG generator for fast HTTP video streaming

def generate_mjpeg():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "../../Source Video/combined videos.mp4"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    face_db_path = '../../AI_VID_face_db.pkl'

    capture = VideoCaptureHandler(video_path, None, 0, 0, 0)
    detector = FaceDetector(face_model_path, device)
    attention = AttentionDetector(attention_model_path, device)
    recognizer = FaceRecognizer(face_db_path, device)
    tracker = FaceTracker()
    display = DisplayManager(color_map)

    target_fps = 20
    frame_interval = 1.0 / target_fps
    last_metrics_emit = time.time()
    last_alert_emit = time.time()

    while True:
        now = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        detected = detector.detect(frame)
        boxes = [d['box'] for d in detected]
        attn_detections = attention.detect(frame)
        attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = tracker.update(detected_for_tracker, 0, recognizer, frame)
        frame = display.draw(frame, tracked_faces, attention_labels, attention.names)

        # Resize and encode as JPEG (fast, small)
        max_height = 480
        h, w = frame.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        # --- Metrics and Alerts ---
        attn_this_frame = sum(1 for label in attention_labels if label == 'attentive')
        total_this_frame = len(attention_labels)
        attention_rate = int((attn_this_frame / total_this_frame * 100) if total_this_frame > 0 else 0)
        comprehension_rate = attention_rate  # Placeholder
        active_students = attn_this_frame

        # Emit metrics every second
        if now - last_metrics_emit >= 1.0 and len(connected_clients) > 0:
            metrics = {
                'attention': attention_rate,
                'comprehension': comprehension_rate,
                'active': active_students
            }
            print("Emitting classroom_metrics:", metrics, "to", list(connected_clients), flush=True)
            for sid in list(connected_clients):
                socketio.emit('classroom_metrics', metrics, room=sid)
            last_metrics_emit = now

        # Emit alerts every second
        if now - last_alert_emit >= 1.0 and len(connected_clients) > 0:
            if attention_rate < 50:
                alert_type = 'danger'
            elif attention_rate < 70:
                alert_type = 'warning'
            else:
                alert_type = 'success'
            alert = {
                'time': time.strftime('%H:%M:%S', time.gmtime(now)),
                'message': f'Attention: {attn_this_frame} attentive students ({attention_rate}%)',
                'type': alert_type
            }
            print("Emitting alert:", alert, "to", list(connected_clients), flush=True)
            for sid in list(connected_clients):
                socketio.emit('alert', alert, room=sid)
            last_alert_emit = now

        elapsed = time.time() - now
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    capture.release()
    cv2.destroyAllWindows()

# Start the background threads on server startup
processing_thread = threading.Thread(target=generate_mjpeg, daemon=True)
processing_thread.start()

@socketio.on('start_stream')
def handle_start_stream():
    global streaming_task
    sid = request.sid
    if streaming_task is None:
        streaming_task = socketio.start_background_task(video_processing, sid)
        emit('stream_status', {'status': 'started'})
    else:
        emit('stream_status', {'status': 'already_running'})

@socketio.on('disconnect')
def handle_disconnect():
    global streaming_task
    streaming_task = None
    print('Client disconnected')

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    connected_clients.add(sid)
    print('Client connected:', sid, flush=True)

@app.route('/video_feed')
def video_feed():
    """HTTP MJPEG endpoint for fast video streaming"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8000)
