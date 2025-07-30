import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QPushButton, QGridLayout, QTextEdit, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal, QObject
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager
import time
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio

# Color map for display
color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

# Event classes
@dataclass
class CaptureEvent:
    frame: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class DetectionEvent:
    detected_faces: List[Dict]
    frame_id: int
    timestamp: float

@dataclass
class TrackingEvent:
    tracked_faces: List[Dict]
    frame_id: int
    timestamp: float

@dataclass
class AttentionEvent:
    attention_results: List[Dict]
    frame_id: int
    timestamp: float

@dataclass
class FusionEvent:
    fused_results: List[Dict]
    frame_id: int
    timestamp: float

@dataclass
class DisplayEvent:
    processed_frame: np.ndarray
    frame_id: int
    timestamp: float

# Event Manager
class EventManager(QObject):
    capture_ready = pyqtSignal(object)  # CaptureEvent
    detection_ready = pyqtSignal(object)  # DetectionEvent
    tracking_ready = pyqtSignal(object)  # TrackingEvent
    attention_ready = pyqtSignal(object)  # AttentionEvent
    fusion_ready = pyqtSignal(object)  # FusionEvent
    display_ready = pyqtSignal(object)  # DisplayEvent

# Component Classes
class CaptureComponent(QThread):
    def __init__(self, video_path: str, event_manager: EventManager):
        super().__init__()
        self.video_path = video_path
        self.event_manager = event_manager
        self.running = False
        self.capture = None
        self.frame_id = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        
    def run(self):
        self.capture = VideoCaptureHandler(self.video_path)
        self.running = True
        self.timer.start(200)  # 200ms periodicity
        self.exec_()
        
    def capture_frame(self):
        if not self.running or not self.capture:
            return
            
        ret, frame = self.capture.read()
        if ret:
            self.frame_id += 1
            event = CaptureEvent(
                frame=frame,
                timestamp=time.time(),
                frame_id=self.frame_id
            )
            self.event_manager.capture_ready.emit(event)
        else:
            self.running = False
            
    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()

class DetectionComponent(QThread):
    def __init__(self, model_path: str, device: str, event_manager: EventManager):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.event_manager = event_manager
        self.detector = None
        self.running = False
        
    def run(self):
        self.detector = FaceDetector(self.model_path, self.device)
        self.running = True
        self.exec_()
        
    def process_capture(self, capture_event: CaptureEvent):
        if not self.running or not self.detector:
            return
            
        detected = self.detector.detect(capture_event.frame)
        event = DetectionEvent(
            detected_faces=detected,
            frame_id=capture_event.frame_id,
            timestamp=time.time()
        )
        self.event_manager.detection_ready.emit(event)

class FaceRecognitionComponent(QThread):
    def __init__(self, db_path: str, device: str, event_manager: EventManager):
        super().__init__()
        self.db_path = db_path
        self.device = device
        self.event_manager = event_manager
        self.recognizer = None
        self.running = False
        
    def run(self):
        self.recognizer = FaceRecognizer(self.db_path, self.device)
        self.running = True
        self.exec_()
        
    def process_detection(self, detection_event: DetectionEvent):
        if not self.running or not self.recognizer:
            return
            
        # Convert detection results for tracker
        detected_for_tracker = [
            (d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], d['centroid']) 
            for d in detection_event.detected_faces
        ]
        
        # Create tracker and update
        tracker = FaceTracker()
        tracked_faces = tracker.update(detected_for_tracker, detection_event.frame_id, self.recognizer, None)
        
        event = TrackingEvent(
            tracked_faces=tracked_faces,
            frame_id=detection_event.frame_id,
            timestamp=time.time()
        )
        self.event_manager.tracking_ready.emit(event)

class AttentionComponent(QThread):
    def __init__(self, model_path: str, device: str, event_manager: EventManager):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.event_manager = event_manager
        self.attention = None
        self.running = False
        
    def run(self):
        self.attention = AttentionDetector(self.model_path, self.device)
        self.running = True
        self.exec_()
        
    def process_capture(self, capture_event: CaptureEvent):
        if not self.running or not self.attention:
            return
            
        attn_detections = self.attention.detect(capture_event.frame)
        event = AttentionEvent(
            attention_results=attn_detections,
            frame_id=capture_event.frame_id,
            timestamp=time.time()
        )
        self.event_manager.attention_ready.emit(event)

class FusionComponent(QThread):
    def __init__(self, event_manager: EventManager):
        super().__init__()
        self.event_manager = event_manager
        self.running = False
        self.pending_detection = {}
        self.pending_tracking = {}
        self.pending_attention = {}
        
    def run(self):
        self.running = True
        self.exec_()
        
    def process_detection(self, detection_event: DetectionEvent):
        if not self.running:
            return
        self.pending_detection[detection_event.frame_id] = detection_event
        self.try_fusion(detection_event.frame_id)
        
    def process_tracking(self, tracking_event: TrackingEvent):
        if not self.running:
            return
        self.pending_tracking[tracking_event.frame_id] = tracking_event
        self.try_fusion(tracking_event.frame_id)
        
    def process_attention(self, attention_event: AttentionEvent):
        if not self.running:
            return
        self.pending_attention[attention_event.frame_id] = attention_event
        self.try_fusion(attention_event.frame_id)
        
    def try_fusion(self, frame_id: int):
        if (frame_id in self.pending_detection and 
            frame_id in self.pending_tracking and 
            frame_id in self.pending_attention):
            
            detection_event = self.pending_detection[frame_id]
            tracking_event = self.pending_tracking[frame_id]
            attention_event = self.pending_attention[frame_id]
            
            # Combine all results
            fused_results = {
                'detection': detection_event.detected_faces,
                'tracking': tracking_event.tracked_faces,
                'attention': attention_event.attention_results,
                'frame_id': frame_id
            }
            
            event = FusionEvent(
                fused_results=fused_results,
                frame_id=frame_id,
                timestamp=time.time()
            )
            self.event_manager.fusion_ready.emit(event)
            
            # Clean up
            del self.pending_detection[frame_id]
            del self.pending_tracking[frame_id]
            del self.pending_attention[frame_id]

class DisplayComponent(QThread):
    def __init__(self, color_map: Dict, event_manager: EventManager):
        super().__init__()
        self.color_map = color_map
        self.event_manager = event_manager
        self.display = None
        self.running = False
        
    def run(self):
        self.display = DisplayManager(self.color_map)
        self.running = True
        self.exec_()
        
    def process_fusion(self, fusion_event: FusionEvent):
        if not self.running or not self.display:
            return
            
        # Extract data from fusion event
        detected_faces = fusion_event.fused_results['detection']
        tracked_faces = fusion_event.fused_results['tracking']
        attention_results = fusion_event.fused_results['attention']
        
        # Create a dummy frame (in real implementation, you'd need to store the original frame)
        # For now, we'll create a placeholder
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get attention labels
        attention = AttentionDetector("", "cpu")  # Dummy for now
        boxes = [d['box'] for d in detected_faces]
        attention_labels = attention.get_attention_labels(boxes, frame, attention_results)
        
        # Draw the frame
        processed_frame = self.display.draw(frame, tracked_faces, attention_labels, attention.names)
        
        event = DisplayEvent(
            processed_frame=processed_frame,
            frame_id=fusion_event.frame_id,
            timestamp=time.time()
        )
        self.event_manager.display_ready.emit(event)

# Main Video Window with Event-Driven Architecture
class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClarifAI Modular - Event-Driven Architecture")
        self.setGeometry(100, 100, 1280, 800)
        
        # Setup UI (same as before)
        self.setup_ui()
        
        # Event-driven architecture setup
        self.event_manager = EventManager()
        
        # Component initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.video_path = "../../Source Video/combined videos.mp4"
        self.face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
        self.attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
        self.face_db_path = '../../AI_VID_face_db.pkl'
        
        # Initialize components
        self.capture_component = CaptureComponent(self.video_path, self.event_manager)
        self.detection_component = DetectionComponent(self.face_model_path, self.device, self.event_manager)
        self.recognition_component = FaceRecognitionComponent(self.face_db_path, self.device, self.event_manager)
        self.attention_component = AttentionComponent(self.attention_model_path, self.device, self.event_manager)
        self.fusion_component = FusionComponent(self.event_manager)
        self.display_component = DisplayComponent(color_map, self.event_manager)
        
        # Setup event connections AFTER components are initialized
        self.setup_event_connections()
        
        # Control flags
        self.running = False
        self.frame_count = 0
        self.attn_stats = []
        self.last_alert_level = None
        self.timing_log = []

    def setup_ui(self):
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

    def setup_event_connections(self):
        # Connect capture events
        self.event_manager.capture_ready.connect(self.detection_component.process_capture)
        self.event_manager.capture_ready.connect(self.attention_component.process_capture)
        
        # Connect detection events
        self.event_manager.detection_ready.connect(self.recognition_component.process_detection)
        self.event_manager.detection_ready.connect(self.fusion_component.process_detection)
        
        # Connect tracking events
        self.event_manager.tracking_ready.connect(self.fusion_component.process_tracking)
        
        # Connect attention events
        self.event_manager.attention_ready.connect(self.fusion_component.process_attention)
        
        # Connect fusion events
        self.event_manager.fusion_ready.connect(self.display_component.process_fusion)
        
        # Connect display events to UI updates
        self.event_manager.display_ready.connect(self.update_ui)

    def start_video(self):
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start all components
        self.capture_component.start()
        self.detection_component.start()
        self.recognition_component.start()
        self.attention_component.start()
        self.fusion_component.start()
        self.display_component.start()

    def stop_video(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Stop all components
        self.capture_component.stop()
        self.capture_component.quit()
        self.detection_component.quit()
        self.recognition_component.quit()
        self.attention_component.quit()
        self.fusion_component.quit()
        self.display_component.quit()
        
        # Wait for threads to finish
        self.capture_component.wait()
        self.detection_component.wait()
        self.recognition_component.wait()
        self.attention_component.wait()
        self.fusion_component.wait()
        self.display_component.wait()

    def update_ui(self, display_event: DisplayEvent):
        # Update video display
        rgb_image = cv2.cvtColor(display_event.processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update metrics (simplified)
        self.frame_count += 1
        print(f"Frame {self.frame_count} displayed at {display_event.timestamp}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_()) 