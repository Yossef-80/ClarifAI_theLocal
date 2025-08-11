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

# Add timing analysis imports
from collections import deque
import statistics






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
        
        # Add real-time timing display
        self.timing_display = QTextEdit()
        self.timing_display.setReadOnly(True)
        self.timing_display.setMaximumHeight(120)
        self.timing_display.setStyleSheet("background: #fff; color: #222; font-size: 12px; border-radius: 8px;")
        self.timing_display.setPlaceholderText("Thread timing will appear here...")
        
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(self.attention_bar)
        metrics_layout.addWidget(self.comprehension_bar)
        metrics_layout.addWidget(self.active_label)
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(self.timing_display)
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
        
        # Add a timer for alerts and metrics updates every second
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics_alerts)
        
        # Add periodic timing data saving
        self.periodic_save_counter = 0
        self.periodic_save_interval = 30  # Save every 30 seconds
        
        # Add a variable to store the latest processed frame
        self.latest_frame = None
        
        # FPS tracking variables
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # More precise FPS tracking
        self.frame_times = []  # Store timestamps for rolling average
        self.max_frame_times = 30  # Keep last 30 frame times for rolling average
        
        # Frame skipping for 45 FPS target
        self.frame_skip_counter = 0
        self.frame_skip_interval = 1  # Start with processing every frame
        
        # Metrics tracking variables (from original file)
        self.attn_stats = []
        self.buffer_size = 10  # Number of frames to buffer
        self.frame_buffer = []  # Rolling buffer for processed frames
        self.last_alert_level = None  # Track last alert type
        self.timing_log = []  # For per-frame timing

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
        self.frame_count = 0

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
        
        # === Threading Timing Analysis ===
        self.thread_timings = {
            'capture': deque(maxlen=1000),  # Increased buffer size for more data
            'detection': deque(maxlen=1000),
            'attention': deque(maxlen=1000),
            'tracker': deque(maxlen=1000),
            'gui_update': deque(maxlen=1000)
        }
        self.frame_start_time = None
        self.thread_start_times = {}
        self.timing_lock = threading.Lock()
        self.timing_data = []  # Store detailed timing data for each frame
        self.frame_counter = 0
        
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.next_frame)

    def record_thread_timing(self, thread_name, start_time, end_time):
        """Record timing for a specific thread with high precision"""
        with self.timing_lock:
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            self.thread_timings[thread_name].append(duration)
            
            # Store detailed timing data for each frame
            self.timing_data.append({
                'frame_number': self.frame_counter,
                'thread': thread_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': duration,
                'timestamp': time.perf_counter()
            })
    
    def record_frame_timing(self, frame_start_time, frame_end_time):
        """Record overall frame timing"""
        with self.timing_lock:
            frame_duration = (frame_end_time - frame_start_time) * 1000
            self.timing_data.append({
                'frame_number': self.frame_counter,
                'thread': 'total_frame',
                'start_time': frame_start_time,
                'end_time': frame_end_time,
                'duration_ms': frame_duration,
                'timestamp': time.perf_counter()
            })
            self.frame_counter += 1
    
    def get_thread_stats(self, thread_name):
        """Get statistics for a specific thread"""
        with self.timing_lock:
            timings = list(self.thread_timings[thread_name])
            if not timings:
                return None
            return {
                'count': len(timings),
                'avg_ms': statistics.mean(timings),
                'min_ms': min(timings),
                'max_ms': max(timings),
                'median_ms': statistics.median(timings),
                'std_ms': statistics.stdev(timings) if len(timings) > 1 else 0
            }
    
    def print_thread_timing_report(self):
        """Print comprehensive timing report for all threads"""
        print("\n" + "="*60)
        print("THREAD TIMING ANALYSIS REPORT")
        print("="*60)
        
        total_frames = 0
        for thread_name in self.thread_timings.keys():
            stats = self.get_thread_stats(thread_name)
            if stats:
                total_frames = max(total_frames, stats['count'])
                print(f"\n{thread_name.upper()} THREAD:")
                print(f"  Frames processed: {stats['count']}")
                print(f"  Average time: {stats['avg_ms']:.2f} ms")
                print(f"  Min time: {stats['min_ms']:.2f} ms")
                print(f"  Max time: {stats['max_ms']:.2f} ms")
                print(f"  Median time: {stats['median_ms']:.2f} ms")
                print(f"  Std deviation: {stats['std_ms']:.2f} ms")
                print(f"  FPS potential: {1000/stats['avg_ms']:.1f} FPS")
        
        print(f"\n{'='*60}")
        print(f"TOTAL FRAMES ANALYZED: {total_frames}")
        print("="*60)
    
    def save_timing_data_periodic(self):
        """Save timing data periodically to prevent data loss"""
        try:
            if self.timing_data:
                # Save current timing data
                df_current = pd.DataFrame(self.timing_data)
                periodic_filename = f'timing_data_periodic_{int(time.perf_counter())}.xlsx'
                df_current.to_excel(periodic_filename, index=False)
                print(f"Periodic timing data saved to: {periodic_filename}")
                
                # Also save a backup CSV file (faster to write)
                csv_filename = f'timing_data_backup_{int(time.perf_counter())}.csv'
                df_current.to_csv(csv_filename, index=False)
                
        except Exception as e:
            print(f"Error saving periodic timing data: {e}")

    def save_timing_report_to_excel(self):
        """Save detailed timing report to Excel file with high accuracy"""
        try:
            # Save detailed timing data
            if self.timing_data:
                df_detailed = pd.DataFrame(self.timing_data)
                detailed_filename = f'thread_timing_detailed_{int(time.perf_counter())}.xlsx'
                df_detailed.to_excel(detailed_filename, index=False)
                print(f"Detailed timing data saved to: {detailed_filename}")
            
            # Save per-frame summary
            frame_summary = []
            for frame_num in range(self.frame_counter):
                frame_data = [d for d in self.timing_data if d['frame_number'] == frame_num]
                if frame_data:
                    frame_summary.append({
                        'Frame_Number': frame_num,
                        'Total_Frame_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'total_frame'), 0),
                        'Capture_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'capture'), 0),
                        'Detection_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'detection'), 0),
                        'Attention_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'attention'), 0),
                        'Tracker_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'tracker'), 0),
                        'GUI_Update_Time_ms': next((d['duration_ms'] for d in frame_data if d['thread'] == 'gui_update'), 0),
                        'Timestamp': frame_data[0]['timestamp']
                    })
            
            if frame_summary:
                df_frame_summary = pd.DataFrame(frame_summary)
                frame_summary_filename = f'frame_timing_summary_{int(time.perf_counter())}.xlsx'
                df_frame_summary.to_excel(frame_summary_filename, index=False)
                print(f"Frame-by-frame summary saved to: {frame_summary_filename}")
            
            # Save thread statistics
            summary_data = []
            for thread_name in self.thread_timings.keys():
                stats = self.get_thread_stats(thread_name)
                if stats:
                    summary_data.append({
                        'Thread': thread_name,
                        'Frames_Processed': stats['count'],
                        'Avg_Time_ms': stats['avg_ms'],
                        'Min_Time_ms': stats['min_ms'],
                        'Max_Time_ms': stats['max_ms'],
                        'Median_Time_ms': stats['median_ms'],
                        'Std_Dev_ms': stats['std_ms'],
                        'Potential_FPS': 1000/stats['avg_ms'] if stats['avg_ms'] > 0 else 0
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                summary_filename = f'thread_timing_summary_{int(time.perf_counter())}.xlsx'
                df_summary.to_excel(summary_filename, index=False)
                print(f"Thread statistics saved to: {summary_filename}")
                
                # Print summary to console
                print(f"\n{'='*60}")
                print("TIMING SUMMARY")
                print(f"{'='*60}")
                for row in summary_data:
                    print(f"{row['Thread']:12}: {row['Avg_Time_ms']:6.2f}ms avg, {row['Min_Time_ms']:6.2f}ms min, {row['Max_Time_ms']:6.2f}ms max")
                print(f"{'='*60}")
                    
        except Exception as e:
            print(f"Error saving timing report: {e}")
            import traceback
            traceback.print_exc()


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
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0

        # Start threads
        threading.Thread(target=self.run_capture, daemon=True).start()
        threading.Thread(target=self.run_detection, daemon=True).start()
        threading.Thread(target=self.run_attention, daemon=True).start()
        threading.Thread(target=self.run_tracker, daemon=True).start()
        # threading.Thread(target=self.run_recognizer, daemon=True).start()  # Commented out - recognition handled by tracker

        # Start timer for GUI updates
        self.timer.start(22)  # ~45 FPS (1000ms/22ms ≈ 45.45 FPS)
        
        # Start timer for metrics and alerts updates every second
        self.metrics_timer.start(1000)  # 1000ms = 1 second

    def stop_video(self):
        self.running = False
        self.stop_event.set()
        if self.capture:
            self.capture.release()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Stop the metrics timer
        self.metrics_timer.stop()
        
        # Generate and save timing reports
        print("\nGenerating thread timing analysis...")
        self.print_thread_timing_report()
        self.save_timing_report_to_excel()
        
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
            start_time = time.perf_counter()
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
                
                end_time = time.perf_counter()
                self.record_thread_timing('gui_update', start_time, end_time)
                
            except Exception as e:
                print(f"[GUI] Error updating display: {e}")

    def update_timing_display(self):
        """Update the real-time timing display in the GUI"""
        try:
            timing_text = "THREAD TIMING (ms):\n"
            timing_text += "-" * 30 + "\n"
            
            for thread_name in ['capture', 'detection', 'attention', 'tracker', 'gui_update']:
                stats = self.get_thread_stats(thread_name)
                if stats and stats['count'] > 0:
                    timing_text += f"{thread_name.upper()[:8]}: {stats['avg_ms']:.1f}ms\n"
                else:
                    timing_text += f"{thread_name.upper()[:8]}: --\n"
            
            # Add bottleneck analysis
            bottlenecks = []
            for thread_name in ['capture', 'detection', 'attention', 'tracker']:
                stats = self.get_thread_stats(thread_name)
                if stats and stats['count'] > 0:
                    bottlenecks.append((thread_name, stats['avg_ms']))
            
            if bottlenecks:
                bottlenecks.sort(key=lambda x: x[1], reverse=True)
                slowest = bottlenecks[0]
                timing_text += f"\nSLOWEST: {slowest[0].upper()} ({slowest[1]:.1f}ms)"
            
            self.timing_display.setText(timing_text)
            
        except Exception as e:
            print(f"Error updating timing display: {e}")

    def update_metrics_alerts(self):
        """Update metrics and alerts every second"""
        if not self.running:
            return
            
        try:
            # Update timing display
            self.update_timing_display()
            
            # Periodic timing data saving
            self.periodic_save_counter += 1
            if self.periodic_save_counter >= self.periodic_save_interval:
                self.save_timing_data_periodic()
                self.periodic_save_counter = 0
            
            # Use the latest available data for metrics
            if hasattr(self, 'attention_labels') and self.attention_labels:
                attn_this_frame = sum(1 for label in self.attention_labels if label == 'attentive')
                total_this_frame = len(self.attention_labels)
                attentive_ratio = attn_this_frame / total_this_frame if total_this_frame > 0 else 0
                percent_val = int(attentive_ratio * 100)
                
                # Update metrics widgets
                self.attention_bar.setValue(percent_val)
                self.comprehension_bar.setValue(percent_val)  # Placeholder
                self.active_label.setText(f"Active Students: {attn_this_frame}")
                
                # Alert logic
                if attentive_ratio <= 0.5:
                    alert_type = "danger"
                    alert_msg = f"ALERT: Only {attn_this_frame} of {total_this_frame} students are attentive! ({percent_val}%)"
                elif attentive_ratio <= 0.7:
                    alert_type = "warning"
                    alert_msg = f"Warning: {attn_this_frame} of {total_this_frame} students are attentive. ({percent_val}%)"
                else:
                    alert_type = "success"
                    alert_msg = f"Good: {attn_this_frame} of {total_this_frame} students are attentive. ({percent_val}%)"
                
                # Only add alert if it's different from the last one
                if not hasattr(self, 'last_alert_level') or self.last_alert_level != alert_type:
                    self.add_alert(alert_msg, alert_type)
                    self.last_alert_level = alert_type
                    
            else:
                # No faces detected
                self.attention_bar.setValue(0)
                self.comprehension_bar.setValue(0)
                self.active_label.setText("Active Students: 0")
                
                # Add alert for no faces detected
                if not hasattr(self, 'last_alert_level') or self.last_alert_level != "no_faces":
                    self.add_alert("No students detected in frame", "warning")
                    self.last_alert_level = "no_faces"
            
            # Keep only last 10 alerts to prevent overflow            while self.alerts_widget.count() > 10:
                self.alerts_widget.takeItem(0)
                
        except Exception as e:
            print(f"[METRICS_ALERTS] Error updating metrics and alerts: {e}")

    def run_capture(self):
        print("[CAPTURE] Thread started")
        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            if self.shared_frame is None:
                continue

            frame_start_time = time.perf_counter()
            
            ret, frame = self.capture.read()
            if not ret:
                print("[CAPTURE] Frame not read, retrying...")
                continue

            print("[CAPTURE] Frame captured")
            self.shared_frame = frame.copy()
            
            # Store frame start time for complete frame timing
            with self.timing_lock:
                self.frame_start_time = frame_start_time

            # Frame skipping for 45 FPS target
            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.frame_skip_interval == 0:
                self.frame_ready.set()  # Let others start
                print("[CAPTURE] frame_ready set")
            
            capture_end_time = time.perf_counter()
            self.record_thread_timing('capture', frame_start_time, capture_end_time)
            # time.sleep(0.010)  # REMOVED: This limits FPS to ~100 FPS max

    def run_detection(self):
        print("[DETECTION] Thread started")
        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            if self.shared_frame is None:
                continue

            start_time = time.perf_counter()
            print("[DETECTION] Detecting faces")
            frame = self.shared_frame.copy()

            self.shared_detections = self.detector.detect(frame)

            end_time = time.perf_counter()
            self.record_thread_timing('detection', start_time, end_time)
            
            self.detection_done.set()
            # self.face_done.set()  # Set face_done for recognizer - commented out since recognition thread is disabled
            # print("[DETECTION] Detection complete")  # Commented out to reduce overhead

    def run_attention(self):
        print("[ATTENTION] Thread started")
        while self.running and not self.stop_event.is_set():
            self.frame_ready.wait()
            if self.shared_frame is None:
                continue

            start_time = time.perf_counter()
            print("[ATTENTION] Classifying attention")
            frame = self.shared_frame.copy()

            # Get attention detections
            attn_detections = self.attention.detect(frame)
            
            # Get attention labels for the detected faces
            face_boxes = [d['box'] for d in self.shared_detections] if self.shared_detections else []
            self.attention_labels = self.attention.get_attention_labels(face_boxes, frame, attn_detections)
            # print(f"[ATTENTION] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")  # Commented out to reduce overhead

            end_time = time.perf_counter()
            self.record_thread_timing('attention', start_time, end_time)
            
            self.attention_done.set()
            # print("[ATTENTION] Attention complete, attention_done set")  # Commented out to reduce overhead

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
            # print("[TRACKER] Waiting for detection and attention...")  # Commented out to reduce overhead
            self.detection_done.wait()
            # print("[TRACKER] Detection done received")  # Commented out to reduce overhead
            self.attention_done.wait()
            # print("[TRACKER] Attention done received")  # Commented out to reduce overhead

            start_time = time.perf_counter()
            # print("[TRACKER] Updating tracked faces")  # Commented out to reduce overhead

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

            # Store tracked faces for metrics access
            self.tracked_faces = tracked_faces

            self.frame_count += 1
            

            
            # Update FPS calculation
            self.fps_frame_count += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                self.current_fps = self.fps_frame_count / elapsed_time
                self.fps_frame_count = 0
                self.fps_start_time = current_time
                print(f"[FPS] Current FPS: {self.current_fps:.2f}")
                
                # Auto-adjust frame skip for 45 FPS target
                if self.current_fps < 40 and self.frame_skip_interval == 1:
                    self.frame_skip_interval = 2  # Skip every other frame
                    print(f"[PERFORMANCE] Low FPS detected, skipping every other frame")
                elif self.current_fps > 50 and self.frame_skip_interval == 2:
                    self.frame_skip_interval = 1  # Process every frame
                    print(f"[PERFORMANCE] High FPS detected, processing every frame")

            # print(f"[TRACKER] attention_labels type: {type(self.attention_labels)}, value: {self.attention_labels}")  # Commented out to reduce overhead
            processed_frame = self.display.draw(
                frame,
                tracked_faces,
                self.attention_labels,
                self.attention.names
            )

            # Store the processed frame for GUI update
            self.latest_frame = processed_frame.copy()
            print("[TRACKER] Frame stored for GUI update")
            
            end_time = time.perf_counter()
            self.record_thread_timing('tracker', start_time, end_time)
            
            # Record complete frame timing (from capture to tracker completion)
            if hasattr(self, 'frame_start_time') and self.frame_start_time:
                self.record_frame_timing(self.frame_start_time, end_time)
            
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