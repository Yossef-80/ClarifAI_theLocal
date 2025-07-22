from ultralytics import YOLO
import torch

class FaceDetector:
    def __init__(self, model_path, device='cuda'):
        self.model = YOLO(model_path)
        self.device = device
        # Move model to the specified device for faster inference
        try:
            self.model.to(self.device)
            if self.device.startswith('cuda') and torch.cuda.is_available():
                # Use half precision for additional speed if supported
                self.model.model.half()
        except Exception as e:
            print('Warning: could not move model to device or set half precision:', e)

    def detect(self, frame):
        results = self.model(frame, verbose=False, imgsz=1280, device=self.device)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            boxes.append({'box': (x1, y1, x2, y2), 'conf': conf, 'centroid': centroid})
        return boxes
