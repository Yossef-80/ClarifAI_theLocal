from ultralytics import YOLO
import numpy as np

class YOLODetectionModel:
    """
    Loads a YOLO model and performs detection on frames, returning boxes, classes, and confidences.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, frame, verbose=False, imgsz=1280):
        results = self.model(frame, verbose=verbose, imgsz=imgsz)
        if not results or len(results) == 0:
            return None, None, None
        boxes = results[0].boxes
        classes = None
        confidences = None
        if boxes is not None and len(boxes) > 0:
            # Convert to numpy arrays for easier handling
            classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
        return boxes, classes, confidences
