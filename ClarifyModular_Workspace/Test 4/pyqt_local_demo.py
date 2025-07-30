import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QPushButton, QGridLayout, QTextEdit, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal, QMutex, QWaitCondition
from Capture import VideoCaptureHandler
from Detection import FaceDetector
from Attention import AttentionDetector
from Face_recognition import FaceRecognizer
from Fusion import FaceTracker
from Display import DisplayManager
import time
import pandas as pd
import queue

color_map = {
    "attentive": (72, 219, 112),
    "inattentive": (66, 135, 245),
    "unattentive": (66, 135, 245),
    "on phone": (0, 165, 255),
    "Unknown": (255, 255, 255),
    "Error": (0, 0, 0)
}

class CaptureThread(QThread):
    """Thread 1: Frame capture - feeds raw frames to detection thread"""
    frame_captured = pyqtSignal(object, int)  # frame, frame_count
    
    def __init__(self, capture_handler, interval=100):
        super().__init__()
        self.capture_handler = capture_handler
        self.interval = interval
        self.running = True
        self.frame_count = 0
        
    def run(self):
        while self.running:
            ret, frame = self.capture_handler.read()
            if ret:
                self.frame_count += 1
                print(f"[Capture] Frame {self.frame_count} captured")
                self.frame_captured.emit(frame.copy(), self.frame_count)
            else:
                print("[Capture] Video ended")
                break
            self.msleep(self.interval)
            
    def stop(self):
        self.running = False

class DetectionThread(QThread):
    """Thread 2: Face detection - gets frames from capture, passes results to recognition and attention"""
    detection_complete = pyqtSignal(object, list, list, int)  # frame, detections, face_crops, frame_count
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True
        self.frame_queue = queue.Queue()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        
    def add_frame(self, frame, frame_count):
        """Called by capture thread to add a new frame for processing"""
        self.mutex.lock()
        self.frame_queue.put((frame, frame_count))
        self.wait_condition.wakeOne()
        self.mutex.unlock()
        
    def run(self):
        while self.running:
            self.mutex.lock()
            if self.frame_queue.empty():
                self.wait_condition.wait(self.mutex)
            if not self.running:
                self.mutex.unlock()
                break
                
            frame, frame_count = self.frame_queue.get()
            self.mutex.unlock()
            
            try:
                # Perform face detection
                detections = self.detector.detect(frame)
                
                # Extract face crops for recognition and attention
                face_crops = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crops.append(face_crop)
                    else:
                        face_crops.append(None)
                
                print(f"[Detection] Frame {frame_count}: {len(detections)} faces detected")
                self.detection_complete.emit(frame, detections, face_crops, frame_count)
                
            except Exception as e:
                print(f"[Detection] Error processing frame {frame_count}: {e}")
                
    def stop(self):
        self.running = False
        self.wait_condition.wakeAll()

class RecognitionThread(QThread):
    """Thread 3: Face recognition - gets face crops from detection thread"""
    recognition_complete = pyqtSignal(list, int)  # recognition_results, frame_count
    
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.running = True
        self.face_queue = queue.Queue()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        
    def add_faces(self, face_crops, frame_count):
        """Called by detection thread to add face crops for recognition"""
        self.mutex.lock()
        self.face_queue.put((face_crops, frame_count))
        self.wait_condition.wakeOne()
        self.mutex.unlock()
        
    def run(self):
        while self.running:
            self.mutex.lock()
            if self.face_queue.empty():
                self.wait_condition.wait(self.mutex)
            if not self.running:
                self.mutex.unlock()
                break
                
            face_crops, frame_count = self.face_queue.get()
            self.mutex.unlock()
            
            try:
                recognition_results = []
                for face_crop in face_crops:
                    if face_crop is not None and face_crop.size > 0:
                        name, score = self.recognizer.recognize(face_crop)
                        recognition_results.append({'name': name, 'score': score})
                    else:
                        recognition_results.append({'name': 'Unknown', 'score': 0.0})
                
                print(f"[Recognition] Frame {frame_count}: {len(recognition_results)} faces recognized")
                self.recognition_complete.emit(recognition_results, frame_count)
                
            except Exception as e:
                print(f"[Recognition] Error processing frame {frame_count}: {e}")
                
    def stop(self):
        self.running = False
        self.wait_condition.wakeAll()

class AttentionThread(QThread):
    """Thread 4: Attention detection - gets face crops from detection thread"""
    attention_complete = pyqtSignal(list, int)  # attention_results, frame_count
    
    def __init__(self, attention_detector):
        super().__init__()
        self.attention_detector = attention_detector
        self.running = True
        self.frame_queue = queue.Queue()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        
    def add_frame_data(self, frame, face_boxes, frame_count):
        """Called by detection thread to add frame and face boxes for attention detection"""
        self.mutex.lock()
        self.frame_queue.put((frame, face_boxes, frame_count))
        self.wait_condition.wakeOne()
        self.mutex.unlock()
        
    def run(self):
        while self.running:
            self.mutex.lock()
            if self.frame_queue.empty():
                self.wait_condition.wait(self.mutex)
            if not self.running:
                self.mutex.unlock()
                break
                
            frame, face_boxes, frame_count = self.frame_queue.get()
            self.mutex.unlock()
            
            try:
                # Use the attention detector to get attention labels for the face boxes
                attention_labels = self.attention_detector.get_attention_labels(face_boxes, frame)
                
                print(f"[Attention] Frame {frame_count}: {len(attention_labels)} attention states detected")
                self.attention_complete.emit(attention_labels, frame_count)
                
            except Exception as e:
                print(f"[Attention] Error processing frame {frame_count}: {e}")
                # Return default labels in case of error
                attention_labels = ['Unknown'] * len(face_boxes)
                self.attention_complete.emit(attention_labels, frame_count)
                
    def stop(self):
        self.running = False
        self.wait_condition.wakeAll()

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClarifAI Modular - Threaded Video Processing Demo")
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

        # Video/model setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.video_path = "../../Source Video/combined videos.mp4"
        self.face_model_path = '../../yolo_detection_model/yolov11s-face.pt'
        self.attention_model_path = '../../yolo_attention_model/attention_weights_16july2025.pt'
        self.face_db_path = '../../AI_VID_face_db.pkl'
        
        # Thread management
        self.capture_thread = None
        self.detection_thread = None
        self.recognition_thread = None
        self.attention_thread = None
        
        # Data storage for combining results
        self.pending_frames = {}  # frame_count -> {frame, detections, recognition, attention}
        self.frame_count = 0
        self.attn_stats = []
        self.timing_log = []
        
        # Display manager
        self.display_manager = DisplayManager(color_map)

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
        # Initialize components
        capture_handler = VideoCaptureHandler(self.video_path)
        detector = FaceDetector(self.face_model_path, self.device)
        recognizer = FaceRecognizer(self.face_db_path, self.device)
        attention_detector = AttentionDetector(self.attention_model_path, self.device)
        
        # Create and start threads
        self.capture_thread = CaptureThread(capture_handler, interval=100)
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

    def stop_video(self):
        # Stop all threads
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.wait()
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
        if self.recognition_thread:
            self.recognition_thread.stop()
            self.recognition_thread.wait()
        if self.attention_thread:
            self.attention_thread.stop()
            self.attention_thread.wait()
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Save timing log
        if self.timing_log:
            df_timing = pd.DataFrame(self.timing_log)
            try:
                excel_path = 'pyqt_timings.xlsx'
                df_timing.to_excel(excel_path, index=False)
                print(f"Timing log saved to {excel_path}")
            except Exception as e:
                print(f"Excel logging error: {e}")

    def on_detection_complete(self, frame, detections, face_crops, frame_count):
        """Called when detection thread completes processing a frame"""
        print(f"[Main] Detection complete for frame {frame_count}")
        
        # Store detection results
        if frame_count not in self.pending_frames:
            self.pending_frames[frame_count] = {}
        self.pending_frames[frame_count].update({
            'frame': frame,
            'detections': detections,
            'face_crops': face_crops
        })
        
        # Extract face boxes for attention detection
        face_boxes = [det['box'] for det in detections]
        
        # Send face crops to recognition thread
        self.recognition_thread.add_faces(face_crops, frame_count)
        # Send frame and face boxes to attention thread
        self.attention_thread.add_frame_data(frame, face_boxes, frame_count)
        
        # Check if we can display this frame
        self.try_display_frame(frame_count)

    def on_recognition_complete(self, recognition_results, frame_count):
        """Called when recognition thread completes processing"""
        print(f"[Main] Recognition complete for frame {frame_count}")
        
        if frame_count not in self.pending_frames:
            self.pending_frames[frame_count] = {}
        self.pending_frames[frame_count]['recognition'] = recognition_results
        
        self.try_display_frame(frame_count)

    def on_attention_complete(self, attention_results, frame_count):
        """Called when attention thread completes processing"""
        print(f"[Main] Attention complete for frame {frame_count}")
        
        if frame_count not in self.pending_frames:
            self.pending_frames[frame_count] = {}
        self.pending_frames[frame_count]['attention'] = attention_results
        
        self.try_display_frame(frame_count)

    def try_display_frame(self, frame_count):
        """Check if all data is available for a frame and display it"""
        if frame_count not in self.pending_frames:
            return
            
        frame_data = self.pending_frames[frame_count]
        required_keys = ['frame', 'detections', 'recognition', 'attention']
        
        if not all(key in frame_data for key in required_keys):
            return  # Not all data available yet
            
        # All data available, display the frame
        frame = frame_data['frame']
        detections = frame_data['detections']
        recognition_results = frame_data['recognition']
        attention_results = frame_data['attention']
        
        # Format data for DisplayManager
        tracked_faces = []
        for i, det in enumerate(detections):
            if i < len(recognition_results) and i < len(attention_results):
                x1, y1, x2, y2 = det['box']
                name = recognition_results[i]['name']
                conf = recognition_results[i]['score']
                tracked_faces.append((i, x1, y1, x2, y2, name, conf))  # Use i as matched_id for now
        
        # Draw the frame
        processed_frame = self.display_manager.draw(frame, tracked_faces, attention_results, ['attentive', 'inattentive'])
        
        # Update GUI
        self.update_display(processed_frame, frame_count)
        
        # Update metrics
        self.update_metrics(attention_results, frame_count)
        
        # Clean up
        del self.pending_frames[frame_count]

    def update_display(self, processed_frame, frame_count):
        """Update the video display with the processed frame"""
        if processed_frame is None:
            return
            
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        print(f"[Display] Frame {frame_count} displayed")

    def update_metrics(self, attention_results, frame_count):
        """Update metrics and alerts based on attention results"""
        if not attention_results:
            return
            
        attn_this_frame = sum(1 for state in attention_results if state == 'attentive')
        total_this_frame = len(attention_results)
        self.attn_stats.append((attn_this_frame, total_this_frame))
        
        # Update metrics every 30 frames (assuming 30 FPS)
        if frame_count % 30 == 0:
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_()) 