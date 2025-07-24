import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QListWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QProgressBar, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import random
import time

# --- Video Processing Thread ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    metrics_signal = pyqtSignal(dict)
    alert_signal = pyqtSignal(dict)
    transcription_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self._run_flag = True
        self.video_path = video_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Resize for display
            frame = cv2.resize(frame, (480, 270))
            self.change_pixmap_signal.emit(frame)
            # Simulate metrics, alerts, transcription
            if frame_count % 30 == 0:
                metrics = {
                    'attention': random.randint(60, 100),
                    'comprehension': random.randint(50, 100),
                    'active': random.randint(10, 30)
                }
                self.metrics_signal.emit(metrics)
            if frame_count % 45 == 0:
                alert = {
                    'type': random.choice(['success', 'warning', 'danger']),
                    'time': time.strftime('%H:%M:%S'),
                    'message': random.choice([
                        'Student left seat',
                        'Phone detected',
                        'High attention',
                        'Low comprehension',
                        'Unknown face detected'
                    ])
                }
                self.alert_signal.emit(alert)
            if frame_count % 60 == 0:
                transcript = f"[{time.strftime('%H:%M:%S')}] Teacher: Please pay attention to the board."
                self.transcription_signal.emit(transcript)
            self.msleep(33)  # ~30 FPS
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- Metrics Widget ---
class MetricsWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.attn_label = QLabel('Attention: 0%')
        self.attn_bar = QProgressBar()
        self.attn_bar.setMaximum(100)
        self.comp_label = QLabel('Comprehension: 0%')
        self.comp_bar = QProgressBar()
        self.comp_bar.setMaximum(100)
        self.active_label = QLabel('Active Students: 0')
        layout.addWidget(QLabel('📊 Class Metrics'))
        layout.addWidget(self.attn_label)
        layout.addWidget(self.attn_bar)
        layout.addWidget(self.comp_label)
        layout.addWidget(self.comp_bar)
        layout.addWidget(self.active_label)
        self.setLayout(layout)

    def update_metrics(self, metrics):
        self.attn_label.setText(f"Attention: {metrics['attention']}%")
        self.attn_bar.setValue(metrics['attention'])
        self.comp_label.setText(f"Comprehension: {metrics['comprehension']}%")
        self.comp_bar.setValue(metrics['comprehension'])
        self.active_label.setText(f"Active Students: {metrics['active']}")

# --- Main GUI ---
class Dashboard(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle('Classroom Dashboard')
        self.setGeometry(100, 100, 1200, 700)
        grid = QGridLayout()
        grid.setSpacing(10)

        # Video (Top Left)
        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 270)
        self.video_label.setStyleSheet('background: #222;')
        video_box = QGroupBox('Video Stream')
        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        video_box.setLayout(vbox)
        grid.addWidget(video_box, 0, 0)

        # Alerts (Top Right)
        self.alerts_list = QListWidget()
        alerts_box = QGroupBox('🚨 Alerts')
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.alerts_list)
        alerts_box.setLayout(vbox2)
        grid.addWidget(alerts_box, 0, 1)

        # Transcription (Bottom Left)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        transcript_box = QGroupBox('📝 Transcription')
        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.transcript_edit)
        transcript_box.setLayout(vbox3)
        grid.addWidget(transcript_box, 1, 0)

        # Metrics (Bottom Right)
        self.metrics_widget = MetricsWidget()
        metrics_box = QGroupBox()
        vbox4 = QVBoxLayout()
        vbox4.addWidget(self.metrics_widget)
        metrics_box.setLayout(vbox4)
        grid.addWidget(metrics_box, 1, 1)

        self.setLayout(grid)

        # Video Thread
        self.thread = VideoThread(video_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.metrics_signal.connect(self.metrics_widget.update_metrics)
        self.thread.alert_signal.connect(self.add_alert)
        self.thread.transcription_signal.connect(self.add_transcription)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def add_alert(self, alert):
        color = {
            'success': '#d1fae5',
            'warning': '#fef9c3',
            'danger': '#fee2e2'
        }.get(alert['type'], '#fff')
        item_text = f"[{alert['time']}] {alert['message']}"
        self.alerts_list.addItem(item_text)
        self.alerts_list.item(self.alerts_list.count()-1).setBackground(Qt.white)
        self.alerts_list.scrollToBottom()

    def add_transcription(self, text):
        self.transcript_edit.append(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = "../../Source Video/combined videos.mp4"  # Adjust as needed
    dashboard = Dashboard(video_path)
    dashboard.show()
    sys.exit(app.exec_()) 