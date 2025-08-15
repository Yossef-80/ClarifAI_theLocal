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
from collections import Counter


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.shared_attn_detections = []
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
        self.timing_log = []  # Store frame-by-frame and total timing
        self.log_buffer = []  # store log_entry each second

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui_frame)
        self.running = False
        
        # Add a variable to store the latest processed frame
        self.latest_frame = None
        
        # FPS tracking variables
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0
        self.engagements = []
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
        self.per_student_attention_frames = {}  # {student_id: [label, label, ...]}
        self.name_to_student_id = {}
        self.next_student_id = 1
        # === Threading Setup ===
        self.shared_frame = None
        self.shared_boxes = []
        self.shared_detections = None

        self.attention_labels = []
        self.recognized_names = []
        self.tracked_faces = []
        self.frame_ready = threading.Event()
        self.frame_ready1 = threading.Event()
        self.face_done = threading.Event()
        self.detection_done = threading.Event()

        self.attention_done = threading.Event()
        self.attention_done1 = threading.Event()
        self.recog_done = threading.Event()
        self.tracker_done_For_Det = threading.Event()
        self.tracker_done_For_Att = threading.Event()
        self.tracker_done_For_Cap = threading.Event()
        self.tracker_done1 = threading.Event()
        self.stop_event = threading.Event()
        self.log_event = threading.Event()

        self.latest_attention = 0
        self.latest_comprehension = 0
        self.latest_active = 0
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.next_frame)
        self.alert_timer = QTimer()
        self.alert_timer.timeout.connect(self.update_attention_alert)

        self.attention_logs=[]
        self.alert_timer.timeout.connect(self.update_metrics_display)
        #self.alert_timer.timeout.connect(self.log_attention_per_second)
        self.alert_timer.start(1000)  # every 1000 ms = 1 second

        #TIME  LOGGING CONTROLLER
        self.enable_time_log = False

        #self.engagements_timer = QTimer()
        #self.engagements_timer.timeout.connect(self.collect_engagement_data)
        #self.engagements_timer.start(7000)  # every 7000 ms = 7 second

    def log_attention_per_second(self):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")

        for student_id, frames in self.per_student_attention_frames.items():
            if not frames:
                continue

            # Separate labels and confidences
            labels = [lbl for lbl, _ in frames]
            confidences = [conf for _, conf in frames]

            # Majority vote
            label_counter = Counter(labels)
            most_common_label, count = label_counter.most_common(1)[0]

            # Latest frame
            latest_label, latest_confidence = frames[-1]

            # Reverse lookup student name
            name = next((n for n, sid in self.name_to_student_id.items() if sid == student_id), "Unknown")

            log_entry = {
                "timestamp": timestamp,
                "student_id": student_id,
                "name": name,
                "attention": most_common_label,
                "confidence": round(count / len(frames), 2),  # agreement percentage
                "latest_label": latest_label,
                "latest_confidence": round(latest_confidence, 2)
            }

            self.attention_logs.append(log_entry)
            self.log_buffer.append(log_entry)
            print(f"[AGG_LOG] {log_entry}")

        self.per_student_attention_frames.clear()

    def compute_engagements_from_log_buffer(self):
        """
        Processes log buffer into engagement metrics:
          - Aggregates per 7-second windows
          - Computes average attention per student & topic
          - Saves report to Excel with two sheets
        """
        import pandas as pd
        import time

        print("[STEP 1] Checking log buffer...")
        if not self.log_buffer:
            print("⚠ No data in log buffer.")
            return

        # -----------------------
        # 2. Configurable constants
        # -----------------------
        print("[STEP 2] Setting constants...")
        CLASS_ID = 2
        COURSE_ID = 1
        TIMETABLE_ID = 4
        TRANSCRIPT_ID = 1
        TOPIC_ID = 1

        # -----------------------
        # 3. Prepare DataFrame
        # -----------------------
        print("[STEP 3] Creating DataFrame from log buffer...")
        df = pd.DataFrame(self.log_buffer)

        print("[STEP 3.1] Parsing timestamps...")
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        start_time = df["timestamp_dt"].min()

        self.engagements = []  # Reset engagements list

        # -----------------------
        # 4. Process data in 7-second windows
        # -----------------------
        print(f"[STEP 4] Processing windows... DataFrame columns: {df.columns.tolist()}")

        topic_counter = 1  # Start topic ID at 1

        while not df.empty:
            window_end = start_time + pd.Timedelta(seconds=10)
            window_df = df[(df["timestamp_dt"] >= start_time) & (df["timestamp_dt"] < window_end)]

            print(f"   Processing window: {start_time} to {window_end} | {len(window_df)} rows")

            try:
                for student_id, group in window_df.groupby("student_id"):
                    print(f"      -> Student ID: {student_id}, Rows: {len(group)}")

                    if "name" not in group.columns:
                        print("      ❌ Missing 'student_name' column in group!")
                        continue

                    total = len(group)
                    attentive_count = sum(str(attn).lower() == "attentive" for attn in group["attention"])
                    attention_percent = round(100 * attentive_count / total, 2)

                    self.engagements.append({
                        "timestamp": int(window_end.timestamp()),
                        "studentId": student_id,
                        "studentName": group["name"].iloc[0],
                        "classId": CLASS_ID,
                        "courseId": COURSE_ID,
                        "timetableId": TIMETABLE_ID,
                        "attentionPercentage": float(attention_percent),
                        "understandingPercentage": float(attention_percent),
                        "transcriptId": TRANSCRIPT_ID,
                        "topicId": topic_counter
                    })

            except Exception as e:
                print(f"   ❌ Error processing window {start_time} - {window_end}: {e}")

            topic_counter += 1  # Increment topic ID for next window
            start_time = window_end
            df = df[df["timestamp_dt"] >= start_time]
        # -----------------------
        # 5. Aggregate averages
        # -----------------------
        print("[STEP 5] Aggregating averages...")
        if not self.engagements:
            print("⚠ No engagements computed.")
            self.avg_attention_per_topic = []
            return

        engagements_df = pd.DataFrame(self.engagements)

        print("[STEP 5.1] Grouping by student & topic...")
        avg_by_topic_student = (
            engagements_df
            .groupby(["studentId", "studentName", "topicId"], as_index=False)["attentionPercentage"]
            .mean()
            .rename(columns={"attentionPercentage": "avgAttentionPercentage"})
        )

        self.avg_attention_per_topic = avg_by_topic_student.to_dict(orient="records")
        print(f"[AVG_ATTENTION] {self.avg_attention_per_topic}")

        # -----------------------
        # 6. Save results to Excel
        # -----------------------
        print("[STEP 6] Saving results to Excel...")
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"engagement_report_{timestamp_str}.xlsx"

        try:
            print("   Opening ExcelWriter...")
            writer = pd.ExcelWriter(filename, engine="openpyxl")

            print("   Writing Per Window Data...")
            engagements_df.to_excel(writer, sheet_name="Per Window Data", index=False)

            print("   Writing Average Per Topic...")
            avg_by_topic_student.to_excel(writer, sheet_name="Average Per Topic", index=False)

            print("   Closing writer...")
            writer.close()

            print(f"✅ Engagement report saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving engagement report Excel: {e}")

    def update_metrics_display(self):
        self.attention_bar.setValue(self.latest_attention)
        self.comprehension_bar.setValue(self.latest_comprehension)
        self.active_label.setText(f"Active Students: {self.latest_active}")
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

    def collect_engagement_data(self):
        import time
        timestamp = int(time.time())
        class_id = 2
        course_id = 1
        timetable_id = 4
        transcript_id = 1
        topic_id = 1

        for student_id, labels in self.per_student_attention_frames.items():
            total = len(labels)
            if total == 0:
                continue

            attentive_count = sum(1 for label in labels if label == "attentive")

            try:
                attention_percent = round(100 * attentive_count / total, 2)
                understanding_percent = attention_percent  # Replace if needed

                if not (0 <= attention_percent <= 100):
                    raise ValueError("Invalid percentage range")

                engagement = {
                    "timestamp": timestamp,
                    "studentId": student_id,
                    "classId": class_id,
                    "courseId": course_id,
                    "timetableId": timetable_id,
                    "attentionPercentage": float(attention_percent),
                    "understandingPercentage": float(understanding_percent),
                    "transcriptId": transcript_id,
                    "topicId": topic_id
                }

                self.engagements.append(engagement)
            except Exception as e:
                print(f"[ERROR] Skipping invalid engagement for student {student_id}: {e}")

    def login_and_get_token(self, email, password):
        """
        Logs in to the backend and returns the access token.
        """
        import requests
        url = "http://localhost:3001/api/auth/login"  # Replace with your actual login endpoint
        headers = {'Content-Type': 'application/json'}
        payload = {"email": email, "password": password}
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code in (200, 201):
                data = response.json()
                return data.get("access_token")
            else:
                print(f"Login failed: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Login error: {e}")
            return None

    def send_engagements_with_login(self):
        """
        Logs in, gets the token, and sends self.engagements to the backend with the token.
        """
        import requests
        import json
        email = "john.smith@school.edu"
        password = "teacher123"
        token = self.login_and_get_token(email, password)
        class_id = 2
        if not token:
            print("Could not obtain access token. Aborting send.")
            return

        # Remove studentName field from each engagement
        clean_engagements = []
        for eng in self.engagements:
            eng_copy = {k: v for k, v in eng.items() if k != "studentName"}
            clean_engagements.append(eng_copy)

        url = f"http://localhost:3001/api/student-engagement/bulk/{class_id}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        payload = {"engagements": clean_engagements}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                print("Engagements sent successfully!")
            else:
                print(f"Failed to send engagements: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error sending engagements: {e}")

    def start_video(self):
        self.capture = VideoCaptureHandler(self.video_path)
        self.fps = self.capture.get_fps()
        self.detector = FaceDetector(self.face_model_path, self.device)
        self.attention = AttentionDetector(self.attention_model_path, self.device)
        self.recognizer = FaceRecognizer(self.face_db_path, self.device)
        self.tracker = FaceTracker()
        self.display = DisplayManager(color_map)
        self.per_student_attention_frames = {}  # {student_id: [label, label, ...]}

        self.frame_count = 0
        self.attn_stats = []
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_event.clear()
        
        # Initialize FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        self.tracker_done_For_Cap.set()
        #self.tracker_done_For_Det.set()
        #self.tracker_done_For_Att.set()

        # Start threads
        threading.Thread(target=self.run_capture, daemon=True).start()
        threading.Thread(target=self.run_detection, daemon=True).start()
        threading.Thread(target=self.run_attention, daemon=True).start()
        threading.Thread(target=self.run_tracker, daemon=True).start()
        threading.Thread(target=self.update_display, daemon=True).start()
        threading.Thread(target=self.logging_thread, daemon=True).start()

        # threading.Thread(target=self.run_recognizer, daemon=True).start()  # Commented out - recognition handled by tracker

        # Start timer for GUI updates
        self.timer.start(33)  # ~30 FPS

    def logging_thread(self):
        while self.running and not self.stop_event.is_set():
            self.log_event.wait()  # Wait until tracker signals
            self.log_event.clear()  # Reset for next trigger
            self.log_attention_per_second()

    def update_attention_alert(self):
        attention = self.latest_attention
        if attention < 50:
            phrase = f"Most students are inattentive ({attention}%)"
            alert_type = "danger"
        elif attention < 75:
            phrase = f"Most students are medium attentive ({attention}%)"
            alert_type = "warning"
        else:
            phrase = f"Most students are attentive ({attention}%)"
            alert_type = "success"

        time_str = QDateTime.currentDateTime().toString("hh:mm:ss")
        item = QListWidgetItem(f"[{time_str}] {phrase}")
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
    def stop_video(self):
        self.running = False
        self.stop_event.set()
        if self.capture:
            self.capture.release()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.attention_logs:
            df_log = pd.DataFrame(self.attention_logs)
            try:
                df_log.to_csv('aggregated_attention_log.csv', index=False)
                print("Aggregated attention log saved.")
            except Exception as e:
                print(f"Error saving attention log: {e}")

        self.compute_engagements_from_log_buffer()

        self.send_engagements_with_login()

        # Write timing log to Excel
        '''
        if self.enable_time_log:
            if self.timing_log:
                df_timing = pd.DataFrame(self.timing_log)
                try:
                    excel_path = 'pyqt_thread_timings_real_kids.xlsx'
                    df_timing.to_excel(excel_path, index=False)
                    print(f"Timing log saved to {excel_path}")
                except Exception as e:
                    print(f"Excel logging error: {e}")
                    
        '''
    def update_gui_frame(self):
        """Update GUI with latest processed frame"""

    def update_display(self):
        """Update GUI with latest processed frame"""
        while self.running and not self.stop_event.is_set():
            self.tracker_done1.wait()
            self.tracker_done1.clear()
           
            print("Display_started\r\n")
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
            print("Display_Ended\r\n")

    def run_capture(self):
        while self.running and not self.stop_event.is_set():
            self.tracker_done_For_Cap.wait()
            self.tracker_done_For_Cap.clear()
            print("CAPTURE_started\r\n")
            if self.enable_time_log:
                thread_start = time.perf_counter()
            ret, frame = self.capture.read()
            if not ret:
                #print("[CAPTURE] Frame not read, retrying...")
                continue

            #print("[CAPTURE] Frame captured")
            self.shared_frame = frame.copy()
            if self.enable_time_log:
                total_duration = time.perf_counter() - thread_start

                if hasattr(self, 'timing_log'):
                    self.timing_log.append({
                        'thread': 'Capture',
                        'frame': 'ALL',
                        'time': total_duration
                    })
            self.frame_ready.set()  # Let others start
            self.frame_ready1.set()  # Let others start
            print("CAPTURE_Ended\r\n")
            #print("[CAPTURE] frame_ready set")
            #time.sleep(0.033)  # 100ms

    def run_detection(self):

        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            # Clear frame_ready to allow capture to set it again for next cycle
            self.frame_ready.clear()
            #self.tracker_done_For_Det.wait()
            #self.tracker_done_For_Det.clear()

            if self.shared_frame is None:
                continue

            print("DETECTION_started\r\n")
            if self.enable_time_log:
                thread_start = time.perf_counter()

            #print("[DETECTION] Detecting faces")
            frame = self.shared_frame.copy()

            self.shared_detections = self.detector.detect(frame)
            if self.enable_time_log:
                total_duration = time.perf_counter()-thread_start

                if hasattr(self, 'timing_log'):
                    self.timing_log.append({
                        'thread': 'DETECTION_TOTAL',
                        'frame': 'ALL',
                        'time': total_duration
                    })
            self.detection_done.set()
            print("DETECTION_Ended\r\n")
            # self.face_done.set()  # Set face_done for recognizer - commented out since recognition thread is disabled
            #print("[DETECTION] Detection complete")

    def run_attention(self):

        while self.running and not self.stop_event.is_set():
            self.frame_ready1.wait()
            # Clear frame_ready to allow capture to set it again for next cycle
            self.frame_ready1.clear()
            #self.tracker_done_For_Att.wait()
            #self.tracker_done_For_Att.clear()

            if self.shared_frame is None:
                continue

            print("ATTENTION_started\r\n")
            if self.enable_time_log:
                thread_start = time.perf_counter()

            print("[ATTENTION] Classifying attention")
            frame = self.shared_frame.copy()

            # Get attention detections
            attn_detections = self.attention.detect(frame)
            self.shared_attn_detections = attn_detections

            # Get attention labels for the detected faces
            # face_boxes = [d['box'] for d in self.shared_detections] if self.shared_detections else []
            # self.attention_labels, self.attention_confidences = self.attention.get_attention_labels(face_boxes, frame,
            #                                                                                         attn_detections)
            #print(f"[ATTENTION] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")
            if self.enable_time_log:
                total_duration = time.perf_counter() - thread_start

                if hasattr(self, 'timing_log'):
                    self.timing_log.append({
                        'thread': 'ATTENTION_TOTAL',
                        'frame': 'ALL',
                        'time': total_duration
                    })
            self.attention_done.set()
            #self.attention_done1.set()
            print("ATTENTION_Ended\r\n")
            #print("[ATTENTION] Attention complete, attention_done set")

    def run_recognizer(self):

        while self.running and not self.stop_event.is_set():
            print("[RECOGNITION] Waiting for face_done...")
            self.face_done.wait()
            if self.shared_frame is None:
                continue

            print("RECOGNITION_started")

            #print("[RECOGNITION] Recognizing faces")
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

            print("RECOGNITION_Ended")
            #print("[RECOGNITION] Recognition complete, clearing face_done")
            # Clear face_done to wait for next detection
            self.face_done.clear()

    def run_tracker(self):

        while self.running and not self.stop_event.is_set():
            #print("[TRACKER] Waiting for detection and attention...")
            self.detection_done.wait()
            self.detection_done.clear()
            #print("[TRACKER] Detection done received")
            self.attention_done.wait()
            self.attention_done.clear()

            print("TRACKER_started\r\n")
            #print("[TRACKER] Updating tracked faces")
            if self.enable_time_log:
                thread_start = time.perf_counter()


            # Copy and process frame
            frame = self.shared_frame.copy()
            face_boxes = [d['box'] for d in self.shared_detections] if self.shared_detections else []
            self.attention_labels, self.attention_confidences = self.attention.get_attention_labels(
                face_boxes,
                frame,
                self.shared_attn_detections
            )
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
            # In run_tracker, after you get tracked_faces and attention_labels:
            for i, face in enumerate(tracked_faces):
                matched_id, x1, y1, x2, y2, name, conf = face

                label = self.attention_labels[i] if i < len(self.attention_labels) else "Unknown"
                attention_conf = self.attention_confidences[i] if hasattr(self, 'attention_confidences') and i < len(
                    self.attention_confidences) else 0.0

                # Dynamic student ID assignment
                if name not in self.name_to_student_id and name != "Unknown":
                    self.name_to_student_id[name] = self.next_student_id
                    self.next_student_id += 1

                student_id = self.name_to_student_id.get(name, None)
                if student_id is None:
                    continue

                # Store (label, confidence) for this frame
                if student_id not in self.per_student_attention_frames:
                    self.per_student_attention_frames[student_id] = []

                att_conf = self.attention_confidences[i] if i < len(self.attention_confidences) else 0.0
                self.per_student_attention_frames[student_id].append((label, att_conf))
                self.frame_count += 1
            
            # Update FPS calculation
            self.fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                self.current_fps = self.fps_frame_count / elapsed_time
                self.fps_frame_count = 0
                self.fps_start_time = current_time
                self.log_event.set()  # Tell logging thread to log now

                #print(f"[FPS] Current FPS: {self.current_fps:.2f}")

            #print(f"[TRACKER] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")
            processed_frame = self.display.draw(
                frame,
                tracked_faces,
                self.attention_labels,
                self.attention.names
            )
            if self.attention_labels:
                attentive_count = sum(1 for label in self.attention_labels if label == "attentive")
                inattentive_count = sum(1 for label in self.attention_labels if label == "inattentive")
                total = len(self.attention_labels)
                if total > 0:
                    attention_percent = int(100 * attentive_count / total)
                    comprehension_percent = int(100 * attentive_count / total)  # Replace with real logic if needed
                    active_students = total
                else:
                    attention_percent = 0
                    comprehension_percent = 0
                    active_students = 0
            else:
                attention_percent = 0
                comprehension_percent = 0
                active_students = 0

            # Store for GUI update
            self.latest_attention = attention_percent
            self.latest_comprehension = comprehension_percent
            self.latest_active = active_students
            # Store the processed frame for GUI update
            self.latest_frame = processed_frame.copy()
            #print("[TRACKER] Frame stored for GUI update")
            
            #print("[TRACKER] Processing complete, clearing events")
            if self.enable_time_log:
                total_duration = time.perf_counter() - thread_start
                if hasattr(self, 'timing_log'):
                    self.timing_log.append({
                        'thread': 'Tracker_TOTAL',
                        'frame': 'ALL',
                        'time': total_duration
                    })
                if hasattr(self, 'timing_log'):
                    self.timing_log.append({
                        'thread': 'FPS',
                        'frame': 'ALL',
                        'time': self.current_fps
                    })

            self.tracker_done1.set()
            #self.tracker_done_For_Det.set()
            #self.tracker_done_For_Att.set()
            self.tracker_done_For_Cap.set()

            print("TRACKER_Ended\r\n")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())