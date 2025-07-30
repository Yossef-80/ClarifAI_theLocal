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
import cv2






color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

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
        self.fps_label = QLabel("FPS: 0.00")
        self.attention_bar.setFormat("Attention: %p%")
        self.comprehension_bar.setFormat("Comprehension: %p%")
        self.attention_bar.setMaximum(100)
        self.comprehension_bar.setMaximum(100)
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(self.attention_bar)
        metrics_layout.addWidget(self.comprehension_bar)
        metrics_layout.addWidget(self.active_label)
        metrics_layout.addWidget(self.fps_label)
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
        self.timer.timeout.connect(self.update_gui_frame)
        self.running = False
        
        # Add a variable to store the latest processed frame
        self.latest_frame = None
        
        # FPS tracking variables
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0

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

        # === Threading Setup ===
        self.shared_frame = None
        self.shared_boxes = []
        self.shared_detections = None

        self.attention_labels = []
        self.recognized_names = []
        self.tracked_faces = []

        self.frame_ready = threading.Event()
        self.face_done = threading.Event()
        self.detection_done = threading.Event()

        self.attention_done = threading.Event()
        self.recog_done = threading.Event()
        self.stop_event = threading.Event()
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.next_frame)


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
        self.stop_event.clear()
        
        # Initialize FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        # Start threads
        threading.Thread(target=self.run_capture, daemon=True).start()
        threading.Thread(target=self.run_detection, daemon=True).start()
        threading.Thread(target=self.run_attention, daemon=True).start()
        threading.Thread(target=self.run_tracker, daemon=True).start()
        # threading.Thread(target=self.run_recognizer, daemon=True).start()  # Commented out - recognition handled by tracker

        # Start timer for GUI updates
        self.timer.start(33)  # ~30 FPS

    def stop_video(self):
        self.running = False
        self.stop_event.set()
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
    def update_gui_frame(self):
        """Update GUI with latest processed frame"""
        if self.latest_frame is not None:
            try:
                rgb_image = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.width(), 
                    self.video_label.height(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
                
                # Update FPS display
                self.fps_label.setText(f"FPS: {self.current_fps:.2f}")
                
            except Exception as e:
                print(f"[GUI] Error updating display: {e}")

    def run_capture(self):
        print("[CAPTURE] Thread started")
        while self.running and not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                print("[CAPTURE] Frame not read, retrying...")
                continue

            print("[CAPTURE] Frame captured")
            self.shared_frame = frame.copy()

            self.frame_ready.set()  # Let others start
            print("[CAPTURE] frame_ready set")
            time.sleep(0.1)  # 100ms

    def run_detection(self):
        print("[DETECTION] Thread started")
        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            if self.shared_frame is None:
                continue

            print("[DETECTION] Detecting faces")
            frame = self.shared_frame.copy()

            self.shared_detections = self.detector.detect(frame)

            self.detection_done.set()
            # self.face_done.set()  # Set face_done for recognizer - commented out since recognition thread is disabled
            print("[DETECTION] Detection complete")

    def run_attention(self):
        print("[ATTENTION] Thread started")
        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            if self.shared_frame is None:
                continue

            print("[ATTENTION] Classifying attention")
            frame = self.shared_frame.copy()

            # Get attention detections
            attn_detections = self.attention.detect(frame)
            
            # Get attention labels for the detected faces
            face_boxes = [d['box'] for d in self.shared_detections] if self.shared_detections else []
            self.attention_labels = self.attention.get_attention_labels(face_boxes, frame, attn_detections)
            print(f"[ATTENTION] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")

            self.attention_done.set()
            print("[ATTENTION] Attention complete, attention_done set")

    def run_recognizer(self):
        print("[RECOGNITION] Thread started")
        while self.running and not self.stop_event.is_set():
            print("[RECOGNITION] Waiting for face_done...")
            self.face_done.wait()
            if self.shared_frame is None:
                continue
                
            print("[RECOGNITION] Recognizing faces")
            frame = self.shared_frame.copy()

            # Extract boxes from detections and recognize each face
            self.recognized_names = []
            if self.shared_detections:
                for detection in self.shared_detections:
                    x1, y1, x2, y2 = detection['box']
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        name, conf = self.recognizer.recognize(face_crop)
                        self.recognized_names.append((name, conf))
                    else:
                        self.recognized_names.append(("Unknown", 1.0))
            
            self.recog_done.set()
            self.recog_done.clear()
            
            print("[RECOGNITION] Recognition complete, clearing face_done")
            # Clear face_done to wait for next detection
            self.face_done.clear()

    def run_tracker(self):
        print("[TRACKER] Thread started")
        while self.running and not self.stop_event.is_set():
            print("[TRACKER] Waiting for detection and attention...")
            self.detection_done.wait()
            print("[TRACKER] Detection done received")
            self.attention_done.wait()
            print("[TRACKER] Attention done received")

            print("[TRACKER] Updating tracked faces")

            # Copy and process frame
            frame = self.shared_frame.copy()

            converted_boxes = []
            for d in self.shared_detections:
                x1, y1, x2, y2 = d['box']
                conf = d['conf']
                centroid = d['centroid']
                converted_boxes.append((x1, y1, x2, y2, conf, centroid))

            tracked_faces = self.tracker.update(
                converted_boxes,
                self.frame_count,
                self.recognizer,
                frame
            )

            self.frame_count += 1
            
            # Update FPS calculation
            self.fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                self.current_fps = self.fps_frame_count / elapsed_time
                self.fps_frame_count = 0
                self.fps_start_time = current_time
                print(f"[FPS] Current FPS: {self.current_fps:.2f}")

            print(f"[TRACKER] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")
            processed_frame = self.display.draw(
                frame,
                tracked_faces,
                self.attention_labels,
                self.attention.names
            )

            # Store the processed frame for GUI update
            self.latest_frame = processed_frame.copy()
            print("[TRACKER] Frame stored for GUI update")
            
            print("[TRACKER] Processing complete, clearing events")
            
            # Clear frame_ready to allow capture to set it again for next cycle
            self.frame_ready.clear()
            
            # Clear events to prep for next cycle
            self.detection_done.clear()
            self.attention_done.clear()

    def update_display(self, frame):
        print("[DISPLAY] DISPLAYING THE IMAGES")
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Use moveToThread to ensure this runs on the main thread
            QTimer.singleShot(0, lambda: self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.width(), 
                    self.video_label.height(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            ))
            print("[DISPLAY] Frame sent to GUI")
        except Exception as e:
            print(f"[DISPLAY] Error updating display: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())