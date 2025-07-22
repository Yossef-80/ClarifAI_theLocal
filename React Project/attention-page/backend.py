import cv2
import torch
import time
import base64
from flask import Flask
from flask_socketio import SocketIO
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
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def video_processing():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "../../Source Video/combined videos.mp4"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    face_db_path = '../../AI_VID_face_db.pkl'

    temp_cap = cv2.VideoCapture(video_path)
    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
    temp_cap.release()

    capture = VideoCaptureHandler(video_path, None, frame_width, frame_height, fps)
    detector = FaceDetector(face_model_path, device)
    attention = AttentionDetector(attention_model_path, device)
    recognizer = FaceRecognizer(face_db_path, device)
    tracker = FaceTracker()
    display = DisplayManager(color_map)

    frame_count = 0
    while True:
        frame_start_time = time.perf_counter()
        ret, frame = capture.read()
        if not ret:
            break
        frame_idx = capture.get_frame_idx()
        frame_count += 1
        timestamp = frame_count / fps

        # Detection
        detected = detector.detect(frame)
        boxes = [d['box'] for d in detected]
        centroids = [d['centroid'] for d in detected]

        # Attention
        attn_detections = attention.detect(frame)
        attention_labels = attention.get_attention_labels(boxes, frame, attn_detections)

        # Tracking & Recognition
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = tracker.update(detected_for_tracker, frame_count, recognizer, frame)

        # Drawing
        frame = display.draw(frame, tracked_faces, attention_labels, attention.names)

        # --- EMIT VIDEO FRAME ---
        frame_b64 = encode_frame(frame)
        socketio.emit('video_frame', frame_b64)

        # --- EMIT ALERTS (example logic) ---
        if frame_count % 100 == 0:
            alert = {
                "time": time.strftime("%H:%M:%S", time.gmtime(timestamp)),
                "message": f"Frame {frame_count}: Example alert!",
                "type": "warning"
            }
            socketio.emit('alert', alert)

        # --- EMIT TRANSCRIPTION (example logic) ---
        if frame_count % 60 == 0:
            transcription = {
                "time": time.strftime("%H:%M:%S", time.gmtime(timestamp)),
                "text": f"Transcription at frame {frame_count}",
                "score": 80
            }
            socketio.emit('transcription', transcription)

        # Control frame rate for demo
        eventlet.sleep(1 / fps)

        # Optionally show the frame locally
        cv2.imshow("Tracked Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == "__main__":
    socketio.start_background_task(video_processing)
    socketio.run(app, host='0.0.0.0', port=8000) 