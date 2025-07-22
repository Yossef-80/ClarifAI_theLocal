import numpy as np
from collections import deque

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

class FaceTracker:
    def __init__(self, max_distance=40, max_history=20, max_missing_frames=10):
        self.face_tracks = {}
        self.last_seen = {}
        self.next_face_id = 0
        self.max_distance = max_distance
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        self.face_id_to_name = {}
        self.face_id_to_conf = {}

    def update(self, boxes, frame_count, recognizer, frame):
        used_ids = set()
        results = []
        for x1, y1, x2, y2, conf, centroid in boxes:
            matched_id = None
            min_distance = float('inf')
            for fid, history in self.face_tracks.items():
                if not history or fid in used_ids:
                    continue
                prev_centroid = history[-1]
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < min_distance and dist < self.max_distance:
                    min_distance = dist
                    matched_id = fid
            if matched_id is None:
                matched_id = self.next_face_id
                self.next_face_id += 1
            used_ids.add(matched_id)
            if matched_id not in self.face_tracks:
                self.face_tracks[matched_id] = deque(maxlen=self.max_history)
            self.face_tracks[matched_id].append(centroid)
            self.last_seen[matched_id] = frame_count
            face_crop = frame[y1:y2, x1:x2]
            should_recognize = matched_id not in self.face_id_to_name or self.face_id_to_name[matched_id] == "Unknown"
            if face_crop.size != 0 and should_recognize:
                name, conf_score = recognizer.recognize(face_crop)
                self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = name, conf_score
            results.append((matched_id, x1, y1, x2, y2, self.face_id_to_name.get(matched_id, "Unknown"), self.face_id_to_conf.get(matched_id, 1.0)))
        # Remove old IDs
        to_delete = [fid for fid in self.last_seen if frame_count - self.last_seen[fid] > self.max_missing_frames]
        for fid in to_delete:
            self.face_tracks.pop(fid, None)
            self.last_seen.pop(fid, None)
            self.face_id_to_name.pop(fid, None)
            self.face_id_to_conf.pop(fid, None)
        return results
