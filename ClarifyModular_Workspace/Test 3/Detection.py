from ultralytics import YOLO
import torch

class FaceDetector:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        results = self.model(frame, verbose=False, imgsz=1280)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append({'box': (x1, y1, x2, y2), 'conf': conf, 'centroid': centroid})
        return boxes
