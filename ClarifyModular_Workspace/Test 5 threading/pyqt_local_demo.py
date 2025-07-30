import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QPushButton, QGridLayout, QTextEdit, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal, QMutex, QWaitCondition, QThreadPool, QRunnable, QObject, QEventLoop
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager
import time
import pandas as pd
import queue
import threading
from collections import deque
import weakref

# Import configuration only once
try:
    from threading_config import *
except ImportError:
    # Fallback configuration if threading_config.py is not available
    QUEUE_SIZES = {
        'capture': 3,
        'detection': 2,
        'recognition': 2,
        'attention': 2,
        'display': 1,
    }
    TIMING = {
        'capture_interval': 40,
        'display_fps': 25,
        'queue_timeout': 0.1,
        'performance_log_interval': 1000,
    }
    PERFORMANCE = {
        'max_timing_history': 100,
        'log_performance_stats': True,
        'save_performance_logs': True,
    }
    ERROR_HANDLING = {
        'max_retries': 3,
        'retry_delay': 0.1,
        'continue_on_error': True,
    }
    MEMORY = {
        'max_pending_frames': 10,
        'cleanup_interval': 100,
        'drop_old_frames': True,
    }
    MODELS = {
        'device': 'auto',
        'detection_confidence': 0.5,
        'recognition_threshold': 0.65,
        'attention_iou_threshold': 0.08,
    }
    DISPLAY = {
        'show_fps': True,
        'show_performance_stats': True,
        'update_metrics_interval': 30,
    }
    LOGGING = {
        'log_level': 'INFO',
        'log_to_file': True,
        'log_file': 'threading_system.log',
        'max_log_size': 10 * 1024 * 1024,
    }
    THREAD_PRIORITIES = {
        'capture': 'normal',
        'detection': 'high',
        'recognition': 'normal',
        'attention': 'normal',
        'display': 'low',
    }
    ADVANCED = {
        'use_thread_pool': False,
        'max_thread_pool_size': 4,
        'enable_work_stealing': True,
        'adaptive_queue_sizes': True,
    }
    
    def get_device():
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def validate_config():
        return True

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

class SimpleQueue:
    """Simplified thread-safe queue for better performance"""
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        
    def put(self, item, block=False):
        """Put item without blocking"""
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False
            
    def get(self, block=False):
        """Get item without blocking"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
            
    def empty(self):
        return self.queue.empty()
        
    def size(self):
        return self.queue.qsize()
        
    def clear(self):
        """Clear all items from queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

class FrameData:
    """Container for frame data with metadata"""
    def __init__(self, frame, frame_count, timestamp=None):
        self.frame = frame
        self.frame_count = frame_count
        self.timestamp = timestamp or time.time()
        self.detections = None
        self.face_crops = None
        self.recognition_results = None
        self.attention_results = None
        
    def is_complete(self):
        """Check if all processing is complete"""
        return (self.detections is not None and 
                self.face_crops is not None and 
                self.recognition_results is not None and 
                self.attention_results is not None)

class CaptureThread(QThread):
    """Thread 1: Frame capture - optimized for performance"""
    frame_captured = pyqtSignal(object)  # FrameData object
    
    def __init__(self, capture_handler, interval=None):
        super().__init__()
        self.capture_handler = capture_handler
        self.interval = interval or TIMING['capture_interval']
        self.running = True
        self.frame_count = 0
        
    def run(self):
        while self.running:
            ret, frame = self.capture_handler.read()
            if ret:
                self.frame_count += 1
                # Don't copy frame - use reference for better performance
                frame_data = FrameData(frame, self.frame_count)
                self.frame_captured.emit(frame_data)
            else:
                print("[Capture] Video ended")
                break
                
            # Simple sleep without complex timing calculations
            self.msleep(self.interval)
            
    def stop(self):
        self.running = False

class DetectionThread(QThread):
    """Thread 2: Face detection - optimized for performance"""
    detection_complete = pyqtSignal(object)  # FrameData object
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True
        self.frame_queue = SimpleQueue(maxsize=QUEUE_SIZES['detection'])
        
    def add_frame(self, frame_data):
        """Add frame data for processing"""
        self.frame_queue.put(frame_data)
        
    def run(self):
        while self.running:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                self.msleep(10)  # Short sleep when no work
                continue
                
            try:
                # Perform face detection
                detections = self.detector.detect(frame_data.frame)
                
                # Extract face crops for recognition and attention
                face_crops = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    face_crop = frame_data.frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crops.append(face_crop)
                    else:
                        face_crops.append(None)
                
                # Update frame data
                frame_data.detections = detections
                frame_data.face_crops = face_crops
                
                self.detection_complete.emit(frame_data)
                
            except Exception as e:
                print(f"[Detection] Error: {e}")
                # Create empty results for error case
                frame_data.detections = []
                frame_data.face_crops = []
                self.detection_complete.emit(frame_data)
                
    def stop(self):
        self.running = False
        self.frame_queue.clear()

class RecognitionThread(QThread):
    """Thread 3: Face recognition - optimized for performance"""
    recognition_complete = pyqtSignal(object)  # FrameData object
    
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.running = True
        self.frame_queue = SimpleQueue(maxsize=QUEUE_SIZES['recognition'])
        
    def add_frame(self, frame_data):
        """Add frame data for recognition processing"""
        self.frame_queue.put(frame_data)
        
    def run(self):
        while self.running:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                self.msleep(10)  # Short sleep when no work
                continue
                
            try:
                recognition_results = []
                for face_crop in frame_data.face_crops:
                    if face_crop is not None and face_crop.size > 0:
                        name, score = self.recognizer.recognize(face_crop)
                        recognition_results.append({'name': name, 'score': score})
                    else:
                        recognition_results.append({'name': 'Unknown', 'score': 0.0})
                
                frame_data.recognition_results = recognition_results
                self.recognition_complete.emit(frame_data)
                
            except Exception as e:
                print(f"[Recognition] Error: {e}")
                # Create default results for error case
                frame_data.recognition_results = [{'name': 'Unknown', 'score': 0.0}] * len(frame_data.face_crops)
                self.recognition_complete.emit(frame_data)
                
    def stop(self):
        self.running = False
        self.frame_queue.clear()

class AttentionThread(QThread):
    """Thread 4: Attention detection - optimized for performance"""
    attention_complete = pyqtSignal(object)  # FrameData object
    
    def __init__(self, attention_detector):
        super().__init__()
        self.attention_detector = attention_detector
        self.running = True
        self.frame_queue = SimpleQueue(maxsize=QUEUE_SIZES['attention'])
        
    def add_frame(self, frame_data):
        """Add frame data for attention processing"""
        self.frame_queue.put(frame_data)
        
    def run(self):
        while self.running:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                self.msleep(10)  # Short sleep when no work
                continue
                
            try:
                # Extract face boxes for attention detection
                face_boxes = [det['box'] for det in frame_data.detections]
                
                # Use the attention detector to get attention labels
                attention_labels = self.attention_detector.get_attention_labels(face_boxes, frame_data.frame)
                
                frame_data.attention_results = attention_labels
                self.attention_complete.emit(frame_data)
                
            except Exception as e:
                print(f"[Attention] Error: {e}")
                # Return default labels in case of error
                frame_data.attention_results = ['Unknown'] * len(frame_data.detections)
                self.attention_complete.emit(frame_data)
                
    def stop(self):
        self.running = False
        self.frame_queue.clear()

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClarifAI Modular - Optimized Threaded Video Processing")
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
        self.fps_label = QLabel("FPS: 0")
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

        # Video/model setup
        self.device = get_device()
        self.video_path = "../../Source Video/combined videos.mp4"
        self.face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
        self.attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
        self.face_db_path = '../../AI_VID_face_db.pkl'
        
        # Thread management
        self.capture_thread = None
        self.detection_thread = None
        self.recognition_thread = None
        self.attention_thread = None
        
        # Data storage
        self.pending_frames = {}  # frame_count -> FrameData
        self.frame_count = 0
        self.attn_stats = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Display manager
        self.display_manager = DisplayManager(color_map)
        
        # FPS timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS every second

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
        try:
            # Initialize components
            capture_handler = VideoCaptureHandler(self.video_path)
            detector = FaceDetector(self.face_model_path, self.device)
            recognizer = FaceRecognizer(self.face_db_path, self.device)
            attention_detector = AttentionDetector(self.attention_model_path, self.device)
            
            # Create threads
            self.capture_thread = CaptureThread(capture_handler)
            self.detection_thread = DetectionThread(detector)
            self.recognition_thread = RecognitionThread(recognizer)
            self.attention_thread = AttentionThread(attention_detector)
            
            # Connect signals
            self.capture_thread.frame_captured.connect(self.detection_thread.add_frame)
            self.detection_thread.detection_complete.connect(self.on_detection_complete)
            self.recognition_thread.recognition_complete.connect(self.on_recognition_complete)
            self.attention_thread.attention_complete.connect(self.on_attention_complete)
            
            # Start threads
            self.capture_thread.start()
            self.detection_thread.start()
            self.recognition_thread.start()
            self.attention_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_alert("Video processing started successfully", "success")
            
        except Exception as e:
            self.add_alert(f"Error starting video: {e}", "danger")
            print(f"Error starting video: {e}")

    def stop_video(self):
        try:
            # Stop all threads
            threads = [
                self.capture_thread,
                self.detection_thread,
                self.recognition_thread,
                self.attention_thread
            ]
            
            for thread in threads:
                if thread:
                    thread.stop()
                    thread.wait()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.add_alert("Video processing stopped", "warning")
            
        except Exception as e:
            self.add_alert(f"Error stopping video: {e}", "danger")
            print(f"Error stopping video: {e}")

    def on_detection_complete(self, frame_data):
        """Called when detection thread completes processing"""
        # Store frame data
        self.pending_frames[frame_data.frame_count] = frame_data
        
        # Send to recognition and attention threads
        self.recognition_thread.add_frame(frame_data)
        self.attention_thread.add_frame(frame_data)
        
        # Check if we can display this frame
        self.try_display_frame(frame_data.frame_count)

    def on_recognition_complete(self, frame_data):
        """Called when recognition thread completes processing"""
        if frame_data.frame_count in self.pending_frames:
            self.pending_frames[frame_data.frame_count].recognition_results = frame_data.recognition_results
            self.try_display_frame(frame_data.frame_count)

    def on_attention_complete(self, frame_data):
        """Called when attention thread completes processing"""
        if frame_data.frame_count in self.pending_frames:
            self.pending_frames[frame_data.frame_count].attention_results = frame_data.attention_results
            self.try_display_frame(frame_data.frame_count)

    def try_display_frame(self, frame_count):
        """Check if all data is available for a frame and display it"""
        if frame_count not in self.pending_frames:
            return
            
        frame_data = self.pending_frames[frame_count]
        
        if frame_data.is_complete():
            # Display the frame directly in main thread for better performance
            self.display_frame(frame_data)
            
            # Update metrics
            self.update_metrics(frame_data.attention_results, frame_count)
            
            # Clean up
            del self.pending_frames[frame_count]

    def display_frame(self, frame_data):
        """Display the processed frame"""
        try:
            # Format data for DisplayManager
            tracked_faces = []
            for i, det in enumerate(frame_data.detections):
                if (i < len(frame_data.recognition_results) and 
                    i < len(frame_data.attention_results)):
                    x1, y1, x2, y2 = det['box']
                    name = frame_data.recognition_results[i]['name']
                    conf = frame_data.recognition_results[i]['score']
                    tracked_faces.append((i, x1, y1, x2, y2, name, conf))
            
            # Draw the frame
            processed_frame = self.display_manager.draw(
                frame_data.frame, 
                tracked_faces, 
                frame_data.attention_results, 
                ['attentive', 'inattentive']
            )
            
            # Update display
            if processed_frame is not None:
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.width(), self.video_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
                # Update FPS counter
                self.fps_counter += 1
                
        except Exception as e:
            print(f"[Display] Error: {e}")

    def update_fps(self):
        """Update FPS display"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time

    def update_metrics(self, attention_results, frame_count):
        """Update metrics and alerts based on attention results"""
        if not attention_results:
            return
            
        attn_this_frame = sum(1 for state in attention_results if state == 'attentive')
        total_this_frame = len(attention_results)
        self.attn_stats.append((attn_this_frame, total_this_frame))
        
        # Update metrics based on configuration
        if frame_count % DISPLAY['update_metrics_interval'] == 0:
            if self.attn_stats:
                last_attn, last_total = self.attn_stats[-1]
                attentive_ratio = last_attn / last_total if last_total > 0 else 0
                percent_val = int(attentive_ratio * 100)
                
                self.attention_bar.setValue(percent_val)
                self.comprehension_bar.setValue(percent_val)
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
            self.attn_stats = []

# Module guard to prevent multiple executions
if __name__ == "__main__":
    # Suppress PyQt5 deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyQt5")
    
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_()) 