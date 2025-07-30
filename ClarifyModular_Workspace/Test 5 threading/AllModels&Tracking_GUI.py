import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QPushButton, QGridLayout, QTextEdit, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import pickle
from collections import deque
import time

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
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

class AllModelsTrackingGUI(QWidget):
    def __init__(self, video_path, face_db_path, face_model_path, attention_model_path):
        super().__init__()
        self.setWindowTitle("ClarifAI Modular - All Models & Tracking GUI")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #f7f7fa;
                color: #222;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 15px;
            }
            QLabel, QProgressBar, QListWidget, QTextEdit {
                border-radius: 10px;
            }
            QPushButton {
                background-color: #e2e8f0;
                color: #222;
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
            QProgressBar {
                background: #e2e8f0;
                border: 1px solid #e2e8f0;
                height: 22px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38b2ac, stop:1 #4299e1);
                border-radius: 10px;
            }
            QListWidget {
                background: #fff;
                border: none;
                padding: 8px;
            }
            QTextEdit {
                background: #fff;
                border: none;
                padding: 8px;
            }
        """)
        # Section Titles
        self.video_title = QLabel("🎥 Video Stream")
        self.video_title.setStyleSheet("font-size:17px;font-weight:600;padding:4px 0 8px 0;color:#222;")
        self.alerts_title = QLabel("🚨 Alerts")
        self.alerts_title.setStyleSheet("font-size:17px;font-weight:600;padding:4px 0 8px 0;color:#222;")
        self.transcription_title = QLabel("📝 Transcription")
        self.transcription_title.setStyleSheet("font-size:17px;font-weight:600;padding:4px 0 8px 0;color:#222;")
        self.metrics_title = QLabel("📊 Metrics")
        self.metrics_title.setStyleSheet("font-size:17px;font-weight:600;padding:4px 0 8px 0;color:#222;")
        # Video (Top Left)
        self.video_label = QLabel("Loading video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #f1f5f9; color: #222; border-radius: 12px; font-size: 18px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Alerts (Top Right)
        self.alerts_widget = QListWidget()
        self.alerts_widget.setStyleSheet("background: #fff; color: #222; font-size: 15px; border-radius: 10px;")
        self.alerts_widget.setAlternatingRowColors(False)
        # Transcription (Bottom Left)
        self.transcription_widget = QTextEdit()
        self.transcription_widget.setReadOnly(True)
        self.transcription_widget.setPlaceholderText("Transcription will appear here.")
        self.transcription_widget.setStyleSheet("background: #fff; color: #222; font-size: 15px; border-radius: 10px;")
        # Metrics (Bottom Right)
        self.attention_bar = QProgressBar()
        self.comprehension_bar = QProgressBar()
        self.active_label = QLabel("Active Students: 0")
        self.attention_bar.setFormat("Attention: %p%")
        self.comprehension_bar.setFormat("Comprehension: %p%")
        self.attention_bar.setMaximum(100)
        self.comprehension_bar.setMaximum(100)
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(self.attention_bar)
        metrics_layout.addWidget(self.comprehension_bar)
        metrics_layout.addWidget(self.active_label)
        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)
        # Start/Stop buttons (below metrics)
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumWidth(90)
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumWidth(90)
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        metrics_layout.addLayout(btn_layout)
        # Grid Layout
        grid = QGridLayout()
        # Top left: Video
        video_vbox = QVBoxLayout()
        video_vbox.addWidget(self.video_title)
        video_vbox.addWidget(self.video_label)
        grid.addLayout(video_vbox, 0, 0)
        # Top right: Alerts
        alerts_vbox = QVBoxLayout()
        alerts_vbox.addWidget(self.alerts_title)
        alerts_vbox.addWidget(self.alerts_widget)
        grid.addLayout(alerts_vbox, 0, 1)
        # Bottom left: Transcription
        trans_vbox = QVBoxLayout()
        trans_vbox.addWidget(self.transcription_title)
        trans_vbox.addWidget(self.transcription_widget)
        grid.addLayout(trans_vbox, 1, 0)
        # Bottom right: Metrics
        metrics_vbox = QVBoxLayout()
        metrics_vbox.addWidget(self.metrics_title)
        metrics_vbox.addWidget(metrics_widget)
        grid.addLayout(metrics_vbox, 1, 1)
        grid.setColumnStretch(0, 7)  # 70%
        grid.setColumnStretch(1, 3)  # 30%
        grid.setRowStretch(0, 3)  # 60%
        grid.setRowStretch(1, 2)  # 40%
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        self.setLayout(grid)

        # Model and video setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.video_path = video_path
        self.face_model_path = face_model_path
        self.attention_model_path = attention_model_path
        self.face_db_path = face_db_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.face_model = YOLO(self.face_model_path)
        self.attention_model = YOLO(self.attention_model_path)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        with open(self.face_db_path, 'rb') as f:
            self.face_db = pickle.load(f)
        self.face_id_to_name = {}
        self.face_id_to_conf = {}
        self.face_tracks = {}
        self.last_seen = {}
        self.max_distance = 40
        self.max_history = 20
        self.max_missing_frames = 10
        self.frame_count = 0
        self.attn_stats = []
        self.last_alert_level = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.running = False
        self.buffer_size = 10
        self.frame_buffer = []

    def add_alert(self, message, alert_type="danger"):
        time_str = QDateTime.currentDateTime().toString("hh:mm:ss")
        item = QListWidgetItem(f"[{time_str}] {message}")
        if alert_type == "success":
            item.setBackground(QColor("#c6f6d5"))
            item.setForeground(QColor("#22543d"))
        elif alert_type == "warning":
            item.setBackground(QColor("#fefcbf"))
            item.setForeground(QColor("#744210"))
        else:
            item.setBackground(QColor("#fed7d7"))
            item.setForeground(QColor("#742a2a"))
        self.alerts_widget.addItem(item)
        self.alerts_widget.scrollToBottom()

    def start_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        self.attn_stats = []
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(int(1000 / self.fps))

    def stop_video(self):
        self.running = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def next_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            self.video_label.setText("Video ended.")
            return
        self.frame_count += 1
        # Detection
        t0 = time.perf_counter()
        results = self.face_model(frame, verbose=False, imgsz=1280)
        detection_time = time.perf_counter() - t0
        detections = results[0].boxes
        boxes = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append((x1, y1, x2, y2, conf, centroid))
        # Attention
        t1 = time.perf_counter()
        attn_results = self.attention_model(frame, verbose=False, imgsz=960)[0].boxes
        attention_time = time.perf_counter() - t1
        attn_boxes = []
        for attn_box in attn_results:
            ax1, ay1, ax2, ay2 = map(int, attn_box.xyxy[0])
            attn_class = int(attn_box.cls[0])
            att_conf = float(attn_box.conf[0])
            attn_boxes.append(((ax1, ay1, ax2, ay2), attn_class))
        # Tracking & Recognition
        used_ids = set()
        face_attention_labels = []
        for x1, y1, x2, y2, conf, centroid in boxes:
            matched_id = None
            min_distance = float('inf')
            for fid, history in self.face_tracks.items():
                if not history or fid in used_ids:
                    continue
                prev_centroid = history[-1]
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < min_distance and dist < self.max_distance:
                    min_distance = dist
                    matched_id = fid
            if matched_id is None:
                matched_id = len(self.face_tracks)
            used_ids.add(matched_id)
            if matched_id not in self.face_tracks:
                self.face_tracks[matched_id] = deque(maxlen=self.max_history)
            self.face_tracks[matched_id].append(centroid)
            self.last_seen[matched_id] = self.frame_count
            face_crop = frame[y1:y2, x1:x2]
            should_recognize = matched_id not in self.face_id_to_name or self.face_id_to_name[matched_id] == "Unknown"
            if face_crop.size != 0 and should_recognize:
                try:
                    face_resized = cv2.resize(face_crop, (160, 160))
                    face_tensor = torch.tensor(np.array(face_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                    with torch.no_grad():
                        face_emb = self.facenet(face_tensor).squeeze().cpu().numpy()
                    best_match, best_score = "Unknown", 1.0
                    for name, emb in self.face_db.items():
                        dist = np.linalg.norm(face_emb - emb)
                        if dist < best_score:
                            best_score, best_match = dist, name
                    if best_score < 0.65:
                        self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = best_match, best_score
                    else:
                        self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = "Unknown", best_score
                except:
                    self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = "Error", 1.0
            # Assign attention label by IoU
            attention_label = "inattentive"
            best_iou = 0
            for (ax1, ay1, ax2, ay2), cls in attn_boxes:
                iou = compute_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                if iou > best_iou and iou > 0.08:
                    best_iou = iou
                    attention_label = self.attention_model.names[cls]
                    if attention_label == "unattentive":
                        attention_label = "inattentive"
            face_attention_labels.append(attention_label)
            name = self.face_id_to_name.get(matched_id, "Unknown")
            conf = self.face_id_to_conf.get(matched_id, 1.0)
            label = f'{name} ({matched_id}) {conf:.2f} | {attention_label}'
            color = color_map.get(attention_label, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Remove old IDs
        to_delete = [fid for fid in self.last_seen if self.frame_count - self.last_seen[fid] > self.max_missing_frames]
        for fid in to_delete:
            self.face_tracks.pop(fid, None)
            self.last_seen.pop(fid, None)
            self.face_id_to_name.pop(fid, None)
            self.face_id_to_conf.pop(fid, None)
        # Metrics and alerts
        attentive_count = sum(1 for label in face_attention_labels if label == 'attentive')
        total_count = len(face_attention_labels)
        percent_val = int((attentive_count / total_count) * 100) if total_count > 0 else 0
        self.attn_stats.append((attentive_count, total_count))
        if self.frame_count % self.fps == 0:
            # Use the last frame's stats for alert and metrics, or average over the second
            if self.attn_stats:
                last_attn, last_total = self.attn_stats[-1]
                # Or, for average over the second:
                # sum_attn = sum(x for x, _ in self.attn_stats)
                # sum_total = sum(y for _, y in self.attn_stats)
                # percent_val = int((sum_attn / sum_total) * 100) if sum_total > 0 else 0
                percent_val = int((last_attn / last_total) * 100) if last_total > 0 else 0
                self.attention_bar.setValue(percent_val)
                self.comprehension_bar.setValue(percent_val)
                self.active_label.setText(f"Active Students: {last_attn}")
                # Alert logic (as before)
                attentive_ratio = last_attn / last_total if last_total > 0 else 0
                if attentive_ratio <= 0.5:
                    alert_type = "danger"
                    alert_msg = f"ALERT: Only {last_attn} of {last_total} students are attentive! ({percent_val}%)"
                elif attentive_ratio <= 0.7:
                    alert_type = "warning"
                    alert_msg = f"Warning: {last_attn} of {last_total} students are attentive. ({percent_val}%)"
                else:
                    alert_type = "success"
                    alert_msg = f"Good: {last_attn} of {last_total} students are attentive. ({percent_val}%)"
                self.add_alert(alert_msg, alert_type)
                self.last_alert_level = alert_type
            self.attn_stats = []
        # Display frame
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "../../Source Video/combined videos.mp4"
    face_db_path = "../../AI_VID_face_db.pkl"
    face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
    attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
    window = AllModelsTrackingGUI(video_path, face_db_path, face_model_path, attention_model_path)
    window.show()
    sys.exit(app.exec_()) 