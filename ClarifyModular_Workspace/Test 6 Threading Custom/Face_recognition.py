import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class FaceRecognizer:
    def __init__(self, db_path, device='cpu'):
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        with open(db_path, 'rb') as f:
            self.face_db = pickle.load(f)
        self.device = device

    def recognize(self, face_crop):
        if face_crop.size == 0:
            return "Unknown", 1.0
        try:
            face_resized = cv2.resize(face_crop, (160, 160))
            face_tensor = torch.tensor(np.array(face_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                face_emb = self.facenet(face_tensor).squeeze().cpu().numpy()
            best_match, best_score = "Unknown", 1.0
            for name, emb in self.face_db.items():
                dist = cosine(face_emb, emb)
                if dist < best_score:
                    best_score, best_match = dist, name
            if best_score < 0.65:
                return best_match, best_score
            else:
                return "Unknown", best_score
        except:
            return "Error", 1.0

class RecognitionWorker(QThread):
    recognition_done = pyqtSignal(list)
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.frame = None
        self.detections = None
        self.running = True

    def set_data(self, frame, detections):
        self.frame = frame
        self.detections = detections

    def run(self):
        while self.running:
            if self.frame is not None and self.detections is not None:
                results = []
                for det in self.detections:
                    x1, y1, x2, y2 = det['box']
                    face_crop = self.frame[y1:y2, x1:x2]
                    name, score = self.recognizer.recognize(face_crop)
                    results.append({'box': det['box'], 'name': name, 'score': score})
                self.recognition_done.emit(results)
                self.frame = None
                self.detections = None
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait()
