from ultralytics import YOLO
import torch
import numpy as np

class AttentionDetector:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        self.device = device
        self.names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=True, imgsz=960,device=self.device)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            attn_class = int(box.cls[0])
            conf = float(box.conf[0])
            boxes.append({'box': (x1, y1, x2, y2), 'class': attn_class, 'conf': conf})
        return boxes

    def get_attention_labels(self, face_boxes, frame, attn_boxes=None):
        # face_boxes: list of (x1, y1, x2, y2)
        if attn_boxes is None:
            attn_boxes = self.detect(frame)
        labels = []
        for (x1, y1, x2, y2) in face_boxes:
            attention_label = "inattentive"
            best_iou = 0
            for attn in attn_boxes:
                ax1, ay1, ax2, ay2 = attn['box']
                cls = attn['class']
                iou = self.compute_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                if iou > best_iou and iou > 0.08:
                    best_iou = iou
                    attention_label = self.names[cls]
                    if attention_label == "unattentive":
                        attention_label = "inattentive"
            labels.append(attention_label)
        return labels

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
