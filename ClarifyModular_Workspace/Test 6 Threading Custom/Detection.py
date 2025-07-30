from ultralytics import YOLO
import torch
from PyQt5.QtCore import QThread, pyqtSignal

class FaceDetector:
    def __init__(self, model_path, device='cuda'):
        self.model = YOLO(model_path)
        self.device = device
        # Move model to the specified device for faster inference
        # try:
        #     self.model.to(self.device)
        #     if self.device.startswith('cuda') and torch.cuda.is_available():
        #         # Use half precision for additional speed if supported
        #         self.model.model.half()
        # except Exception as e:
        #     print('Warning: could not move model to device or set half precision:', e)

    def detect(self, frame):
        results = self.model(frame, verbose=True, imgsz=1280, device=self.device)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append({'box': (x1, y1, x2, y2), 'conf': conf, 'centroid': centroid})
        return boxes

class DetectionWorker(QThread):
    detection_done = pyqtSignal(list)
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.frame = None
        self.running = True
        self.busy = False

    def set_frame(self, frame):
        if not self.busy:
            self.frame = frame

    def run(self):
        while self.running:
            if self.frame is not None:
                self.busy = True
                try:
                    detected = self.detector.detect(self.frame)
                    self.detection_done.emit(detected)
                except Exception as e:
                    print(f"DetectionWorker crashed: {e}")
                self.frame = None
                self.busy = False
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait()
