import cv2
import torch
import numpy as np
from collections import deque
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle




class FaceRecognitionSystem:
    """
    Comprehensive face recognition and tracking system using proven tracking logic.
    """
    def __init__(self, face_db_path, device='cuda', max_distance=40, max_history=20, max_missing_frames=10):
        # Setup device
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load facenet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Load face database
        with open(face_db_path, 'rb') as f:
            self.face_db = pickle.load(f)
        
        # Tracking parameters (exact same as enhanced file)
        self.max_distance = max_distance
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        
        # Recognition state
        self.face_id_to_name = {}
        self.face_id_to_conf = {}
        
        # Color map for visualization
        self.color_map = {
            "attentive": (72, 219, 112),
            "inattentive": (66, 135, 245),
            "unattentive": (66, 135, 245),
            "on phone": (0, 165, 255),
            "Unknown": (255, 255, 255),
            "Error": (0, 0, 0)
        }
        
        # Current frame results
        self._current_results = []
        self._current_frame = None
        self._current_annotated_frame = None

    def process_frame(self, frame, detection_boxes, frame_count):
        """
        Process a single frame with detection boxes using exact logic from enhanced file.
        Args:
            frame: Input frame (numpy array)
            detection_boxes: List of (x1, y1, x2, y2, conf, centroid) from detection model
            frame_count: Current frame number
        Returns:
            tuple: (annotated_frame, face_results)
        """
        self._current_frame = frame.copy()
        self._current_results = []
        
        # Initialize tracking state exactly like enhanced file
        face_tracks = getattr(self, '_face_tracks', {})
        last_seen = getattr(self, '_last_seen', {})
        next_face_id = getattr(self, '_next_face_id', 0)
        used_ids = set()
        
        # Process each detected face using exact logic from enhanced file
        for x1, y1, x2, y2, conf, centroid in detection_boxes:
            matched_id = None
            min_distance = float('inf')
            
            # Find closest existing track within distance threshold
            for fid, history in face_tracks.items():
                if not history or fid in used_ids:
                    continue
                prev_centroid = history[-1]
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < min_distance and dist < self.max_distance:
                    min_distance = dist
                    matched_id = fid
            
            # If no match found, create new track
            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1
            
            # Update tracking state (exact same as enhanced file)
            used_ids.add(matched_id)
            if matched_id not in face_tracks:
                face_tracks[matched_id] = deque(maxlen=self.max_history)
            face_tracks[matched_id].append(centroid)
            last_seen[matched_id] = frame_count
            
            # Recognize the face
            face_crop = frame[y1:y2, x1:x2]
            name, recog_conf = self._recognize_face(matched_id, face_crop)
            
            # Store result (without attention label - that's handled by FusionSystem)
            result = {
                'id': matched_id,
                'name': name,
                'confidence': recog_conf,
                'bbox': (x1, y1, x2, y2),
                'centroid': centroid,
                'detection_conf': conf
            }
            self._current_results.append(result)
        
        # Remove old IDs (exact same as enhanced file)
        to_delete = [fid for fid in last_seen if frame_count - last_seen[fid] > self.max_missing_frames]
        for fid in to_delete:
            face_tracks.pop(fid, None)
            last_seen.pop(fid, None)
            self.face_id_to_name.pop(fid, None)
            self.face_id_to_conf.pop(fid, None)
        
        # Save tracking state for next frame
        self._face_tracks = face_tracks
        self._last_seen = last_seen
        self._next_face_id = next_face_id
        
        # Create annotated frame
        self._current_annotated_frame = self._annotate_frame(frame.copy())
        
        return self._current_annotated_frame, self._current_results





    def _recognize_face(self, matched_id, face_crop):
        """Recognize a face using facenet embeddings."""
        should_recognize = matched_id not in self.face_id_to_name or self.face_id_to_name[matched_id] == "Unknown"
        
        if face_crop.size == 0 or not should_recognize:
            return self.face_id_to_name.get(matched_id, "Unknown"), self.face_id_to_conf.get(matched_id, 1.0)
        
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
                self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = best_match, best_score
            else:
                self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = "Unknown", best_score
                
        except Exception:
            self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id] = "Error", 1.0
        
        return self.face_id_to_name[matched_id], self.face_id_to_conf[matched_id]



    def _annotate_frame(self, frame):
        """Annotate frame with face recognition results."""
        for result in self._current_results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            matched_id = result['id']
            recog_conf = result['confidence']
            
            label = f'{name} ({matched_id}) {recog_conf:.2f}'
            color = self.color_map.get(name, (128, 128, 128))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

    # ========== GETTERS ==========
    
    @property
    def tracked_faces(self):
        """Get current face tracks."""
        return getattr(self, '_face_tracks', {})
    
    @property
    def id_to_name(self):
        """Get mapping of face IDs to names."""
        return self.face_id_to_name
    
    @property
    def id_to_conf(self):
        """Get mapping of face IDs to recognition confidence."""
        return self.face_id_to_conf
    
    @property
    def last_seen_ids(self):
        """Get mapping of face IDs to last seen frame."""
        return getattr(self, '_last_seen', {})
    
    @property
    def current_results(self):
        """Get current frame recognition results."""
        return self._current_results
    
    @property
    def current_annotated_frame(self):
        """Get current annotated frame."""
        return self._current_annotated_frame
    
    @property
    def facenet_model(self):
        """Get the facenet model."""
        return self.facenet
    
    @property
    def face_database(self):
        """Get the face database."""
        return self.face_db
    
    @property
    def color_mapping(self):
        """Get the color mapping for visualization."""
        return self.color_map
    
    def get_face_info(self, face_id):
        """Get comprehensive info for a specific face ID."""
        face_tracks = getattr(self, '_face_tracks', {})
        last_seen = getattr(self, '_last_seen', {})
        if face_id in face_tracks:
            history = face_tracks[face_id]
            current_pos = history[-1] if history else None
            return {
                'id': face_id,
                'name': self.face_id_to_name.get(face_id, "Unknown"),
                'confidence': self.face_id_to_conf.get(face_id, 1.0),
                'last_seen': last_seen.get(face_id, 0),
                'current_position': current_pos,
                'track_history_length': len(history) if history else 0
            }
        return None
    
    def get_all_face_info(self):
        """Get comprehensive info for all tracked faces."""
        face_info = {}
        face_tracks = getattr(self, '_face_tracks', {})
        for face_id in face_tracks.keys():
            face_info[face_id] = self.get_face_info(face_id)
        return face_info

    def get_tracking_stats(self):
        """Get tracking statistics."""
        face_tracks = getattr(self, '_face_tracks', {})
        next_face_id = getattr(self, '_next_face_id', 0)
        return {
            'total_tracks': len(face_tracks),
            'active_tracks': len(face_tracks),  # All tracks are active in this implementation
            'next_id': next_face_id,
            'track_histories': {fid: len(history) for fid, history in face_tracks.items()}
        }
