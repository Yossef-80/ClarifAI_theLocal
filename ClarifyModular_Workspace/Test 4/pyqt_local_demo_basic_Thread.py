import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QPushButton, QGridLayout, QTextEdit, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager
import time
import pandas as pd
import threading
import time


color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

# Global flag to control threads
threads_running = True

# Create events for synchronization
event2 = threading.Event()
event3 = threading.Event()
event4 = threading.Event()

def thread1():
    global threads_running
    while threads_runn  ing:
        print("T1")
        time.sleep(0.1)  # 10 seconds
        if not threads_running:
            break
        #print("Thread 1: Done sleeping, triggering Thread 2")
        event2.set()

def thread2():
    global threads_running
    while threads_running:
        event2.wait()  # Wait until thread1 is done
        if not threads_running:
            break
        event2.clear()  # Reset the event
        print("T2")
        event3.set()

def thread3():
    global threads_running
    while threads_running:
        event3.wait()
        if not threads_running:
            break
        event3.clear()  # Reset the event
        print("T3")
        event4.set()

def thread4():
    global threads_running
    while threads_running:
        event4.wait()
        if not threads_running:
            break
        event4.clear()  # Reset the event
        print("T4")


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClarifAI Modular - Local Video Processing Demo")
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

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.running = False

        # Video/model setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.video_path = "../../Source Video/combined videos.mp4"
        self.face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
        self.attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
        self.face_db_path = '../../AI_VID_face_db.pkl'
        self.capture = None
        self.detector = None
        self.attention = None
        self.recognizer = None
        self.tracker = None
        self.display = None
        #self.fps = 30
        self.frame_count = 0
        self.attn_stats = []
        self.buffer_size = 10  # Number of frames to buffer
        self.frame_buffer = []  # Rolling buffer for processed frames
        self.last_alert_level = None  # Track last alert type
        self.timing_log = []  # For per-frame timing

    def add_alert(self, message, alert_type="danger"):
        time_str = QDateTime.currentDateTime().toString("hh:mm:ss")
        item = QListWidgetItem(f"[{time_str}] {message}")
        # Modern, subtle colors for light theme
        if alert_type == "success":
            item.setBackground(QColor("#c6f6d5"))  # light green
            item.setForeground(QColor("#22543d"))
        elif alert_type == "warning":
            item.setBackground(QColor("#fefcbf"))  # light yellow
            item.setForeground(QColor("#744210"))
        else:
            item.setBackground(QColor("#fed7d7"))  # light red
            item.setForeground(QColor("#742a2a"))
        self.alerts_widget.addItem(item)
        self.alerts_widget.scrollToBottom()


    def start_video(self):
        self.capture = VideoCaptureHandler(self.video_path)
        self.fps = self.capture.get_fps()
        self.detector = FaceDetector(self.face_model_path, self.device)
        self.attention = AttentionDetector(self.attention_model_path, self.device)
        self.recognizer = FaceRecognizer(self.face_db_path, self.device)
        self.tracker = FaceTracker()
        self.display = DisplayManager(color_map)
        self.frame_count = 0
        self.attn_stats = []
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(int(20))

        # Create threads as daemon threads (non-blocking)
        t1 = threading.Thread(target=thread1, daemon=True)
        t2 = threading.Thread(target=thread2, daemon=True)
        t3 = threading.Thread(target=thread3, daemon=True)
        t4 = threading.Thread(target=thread4, daemon=True)

        # Start threads
        t1.start()
        t2.start()
        t3.start()
        t4.start()

    def stop_video(self):
        global threads_running
        threads_running = False  # Stop the background threads
        
        # Clear events to unblock any waiting threads
        event2.set()
        event3.set()
        event4.set()
        
        self.running = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Write timing log to Excel
        if self.timing_log:
            df_timing = pd.DataFrame(self.timing_log)
            try:
                excel_path = 'pyqt_timings.xlsx'
                df_timing.to_excel(excel_path, index=False)
                print(f"Timing log saved to {excel_path}")
            except Exception as e:
                print(f"Excel logging error: {e}")

    def next_frame(self):
        if not self.running or not self.capture:
            return
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        ret, frame = self.capture.read()
        t1 = time.perf_counter()
        if not ret:
            self.stop_video()
            self.video_label.setText("Video ended.")
            return
        self.frame_count += 1
        print(f"[Frame] {self.frame_count} | Target FPS: {self.fps}")
        # Timing for detection
        start_det = time.perf_counter()
        detected = self.detector.detect(frame)
        end_det = time.perf_counter()
        det_time_ms = (end_det - start_det) * 1000
        print(f"[Timing] Detection model: {det_time_ms:.2f} ms")
        boxes = [d['box'] for d in detected]
        # Timing for attention
        start_attn = time.perf_counter()
        attn_detections = self.attention.detect(frame)
        end_attn = time.perf_counter()
        attn_time_ms = (end_attn - start_attn) * 1000
        print(f"[Timing] Attention model: {attn_time_ms:.2f} ms")
        # Timing for get_attention_labels
        start_labels = time.perf_counter()
        attention_labels = self.attention.get_attention_labels(boxes, frame, attn_detections)
        end_labels = time.perf_counter()
        labels_time_ms = (end_labels - start_labels) * 1000
        print(f"[Timing] Attention labels: {labels_time_ms:.2f} ms")
        # Timing for tracker update
        start_tracker = time.perf_counter()
        detected_for_tracker = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) for d in detected]
        tracked_faces = self.tracker.update(detected_for_tracker, self.frame_count, self.recognizer, frame)
        end_tracker = time.perf_counter()
        tracker_time_ms = (end_tracker - start_tracker) * 1000
        print(f"[Timing] Tracker update: {tracker_time_ms:.2f} ms")
        # Timing for display draw
        start_draw = time.perf_counter()
        processed_frame = self.display.draw(frame, tracked_faces, attention_labels, self.attention.names)
        end_draw = time.perf_counter()
        draw_time_ms = (end_draw - start_draw) * 1000
        print(f"[Timing] Display draw: {draw_time_ms:.2f} ms")
        # Timing for capture.read
        capture_time_ms = (t1 - t0) * 1000
        print(f"[Timing] Capture read: {capture_time_ms:.2f} ms")
        # Metrics
        attn_this_frame = sum(1 for label in attention_labels if label == 'attentive')
        total_this_frame = len(attention_labels)
        self.attn_stats.append((attn_this_frame, total_this_frame))
        # Update metrics every second
        if self.frame_count % self.fps == 0:
            # Use the last frame's stats for alert and metrics
            if self.attn_stats:
                last_attn, last_total = self.attn_stats[-1]
                attentive_ratio = last_attn / last_total if last_total > 0 else 0
                percent_val = int(attentive_ratio * 100)
                # Update metrics widgets
                self.attention_bar.setValue(percent_val)
                self.comprehension_bar.setValue(percent_val)  # Placeholder
                self.active_label.setText(f"Active Students: {last_attn}")
                # Alert logic
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
        # Buffer logic
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.buffer_size:
            # Pop and display the oldest frame
            frame_to_show = self.frame_buffer.pop(0)
            rgb_image = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # Always scale to current label size
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            print(f"[Buffering] Buffered {len(self.frame_buffer)}/{self.buffer_size} frames before visualization.")
        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000
        print(f"[Timing] Total frame: {total_time_ms:.2f} ms")
        fps = 1000 / total_time_ms if total_time_ms > 0 else 0
        print(f"[Timing] Computed FPS: {fps:.2f}")
        # Log timings for this frame
        self.timing_log.append({
            'frame': self.frame_count,
            'capture_read_ms': capture_time_ms,
            'detection_ms': det_time_ms,
            'attention_ms': attn_time_ms,
            'attention_labels_ms': labels_time_ms,
            'tracker_update_ms': tracker_time_ms,
            'display_draw_ms': draw_time_ms,
            'total_frame_ms': total_time_ms,
            'computed_fps': fps
        })
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())