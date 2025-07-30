import cv2
import torch
import time
import base64
from flask import Flask,request
from flask_socketio import SocketIO, emit
import eventlet

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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def encode_frame(frame):
    # Resize frame to max height 480 if larger, keeping aspect ratio
    max_height = 480
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

# Store background task reference to avoid multiple streams
streaming_task = None

def video_processing(sid):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "../../Source Video/combined videos.mp4"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    face_db_path = '../../AI_VID_face_db.pkl'

    # temp_cap = cv2.VideoCapture(video_path)
    # frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
    # temp_cap.release()

    capture = VideoCaptureHandler(video_path)
    detector = FaceDetector(face_model_path, device)
    attention = AttentionDetector(attention_model_path, device)
    recognizer = FaceRecognizer(face_db_path, device)
    tracker = FaceTracker()
    display = DisplayManager(color_map)
    fps=capture.get_fps()
    frame_count = 0
    while True:
        loop_start = time.time()
        t_read_start = time.time()
        ret, frame = capture.read()
        t_read_end = time.time()
        print(f"[Timing] Frame read: {(t_read_end - t_read_start) * 1000:.1f} ms")
        if not ret:
            break
        frame_idx = capture.get_frame_idx()
        frame_count += 1
        timestamp = frame_count / fps

        t0 = time.time()
        # Detection
        detected = detector.detect(frame)
        t1 = time.time()
        print(f"[Timing] Detection: {(t1 - t0) * 1000:.1f} ms")

        t_box_start = time.time()
        boxes = [d['box'] for d in detected]
        centroids = [d['centroid'] for d in detected]
        t_box_end = time.time()
        print(f"[Timing] Box/Centroid extraction: {(t_box_end - t_box_start) * 1000:.1f} ms")

        t2 = time.time()
        # Attention
        attn_detections = attention.detect(frame)
        t3 = time.time()
        print(f"[Timing] Attention: {(t3 - t2) * 1000:.1f} ms")

        t_attlabel_start = time.time()
        attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)
        t_attlabel_end = time.time()
        print(f"[Timing] Attention label extraction: {(t_attlabel_end - t_attlabel_start) * 1000:.1f} ms")

        t4 = time.time()
        # Tracking & Recognition
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = tracker.update(detected_for_tracker, frame_count, recognizer, frame)
        t5 = time.time()
        print(f"[Timing] Tracking & Recognition: {(t5 - t4) * 1000:.1f} ms")

        t_draw_start = time.time()
        # Drawing
        frame = display.draw(frame, tracked_faces, attention_labels, attention.names)
        t_draw_end = time.time()
        print(f"[Timing] Drawing: {(t_draw_end - t_draw_start) * 1000:.1f} ms")

        t_enc_start = time.time()
        # --- EMIT VIDEO FRAME ---
        frame_b64 = encode_frame(frame)
        t_enc_end = time.time()
        print(f"[Timing] Encoding: {(t_enc_end - t_enc_start) * 1000:.1f} ms")

        t_emit_start = time.time()
        socketio.emit('video_frame', frame_b64, room=sid)
        t_emit_end = time.time()
        print(f"[Timing] Emit: {(t_emit_end - t_emit_start) * 1000:.1f} ms")

        # --- EMIT ALERTS (example logic) ---
        t_alert_start = time.time()
        if frame_count % 100 == 0:
            alert = {
                "time": time.strftime("%H:%M:%S", time.gmtime(timestamp)),
                "message": f"Frame {frame_count}: Example alert!",
                "type": "warning"
            }
            socketio.emit('alert', alert, room=sid)
        t_alert_end = time.time()
        print(f"[Timing] Alert emit: {(t_alert_end - t_alert_start) * 1000:.1f} ms")

        t_trans_start = time.time()
        # --- EMIT TRANSCRIPTION (example logic) ---
        if frame_count % 60 == 0:
            transcription = {
                "time": time.strftime("%H:%M:%S", time.gmtime(timestamp)),
                "text": f"Transcription at frame {frame_count}",
                "score": 80
            }
            socketio.emit('transcription', transcription, room=sid)
        t_trans_end = time.time()
        print(f"[Timing] Transcription emit: {(t_trans_end - t_trans_start) * 1000:.1f} ms")

        loop_end = time.time()
        print(f"[Timing] Total frame time: {(loop_end - loop_start) * 1000:.1f} ms\n")
        eventlet.sleep(1 / fps)

    capture.release()
    cv2.destroyAllWindows()

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
    streaming_task = None  # This will not stop the thread, but you can add logic to break the loop if needed
    print('Client disconnected')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8000)