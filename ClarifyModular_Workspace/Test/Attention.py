# Attention.py

from ultralytics import YOLO
import cv2



class YOLOAttentionModel:
    """
    Wrapper for Ultralytics YOLO model.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def predict(self, frame, verbose=False, imgsz=960):
        # Returns the boxes from the first result
        results = self.model(frame, verbose=verbose, imgsz=imgsz)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            # Extract class and confidence for each detection
            classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
            return boxes, classes, confidences
        else:
            return None, None, None


class Attention:
    """
    Attention model/controller that retrieves frames and passes them to YOLO.
    """
    def __init__(self, frame_provider, yolo_model):
        self.frame_provider = frame_provider
        self.yolo_model = yolo_model

    def process(self, verbose=False, imgsz=960):
        # Retrieve frame from the provider
        frame = self.frame_provider.get_frame()
        if frame is None:
            print("No frame retrieved.")
            return None

        # (Optional) Apply attention mechanism here

        # Pass frame to YOLO model
        yolo_output = self.yolo_model.predict(frame, verbose=verbose, imgsz=imgsz)
        return yolo_output
