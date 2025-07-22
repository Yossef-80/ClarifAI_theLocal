import cv2
import numpy as np

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Args:
        boxA: (x1, y1, x2, y2)
        boxB: (x1, y1, x2, y2)
    Returns:
        IoU value (float)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

class FusionSystem:
    """
    Fusion system that combines attention and detection results using IoU.
    Extracts IoU logic from AllModels&Tracking Enhanced.py
    """
    def __init__(self, attention_model, iou_threshold=0.08):  # Lowered threshold
        self._attention_model = attention_model
        self._iou_threshold = iou_threshold
        
        # Current frame state
        self._current_attn_boxes = []
        self._current_detection_boxes = []
        self._current_fusion_results = []
        self._current_annotated_frame = None
        
        # Color map for attention labels
        self.color_map = {
            "attentive": (72, 219, 112),
            "inattentive": (66, 135, 245),
            "unattentive": (66, 135, 245),
            "on phone": (0, 165, 255),
            "Unknown": (255, 255, 255),
            "Error": (0, 0, 0)
        }

    def process_attention_frame(self, frame, verbose=False, imgsz=960):
        """
        Process frame with attention model and extract attention boxes.
        Args:
            frame: Input frame (numpy array)
            verbose: Verbose output for attention model
            imgsz: Image size for attention model
        """
        attn_results = self._attention_model(frame, verbose=verbose, imgsz=imgsz)[0].boxes
        self._current_attn_boxes = []
        
        if attn_results is not None and len(attn_results) > 0:
            for attn_box in attn_results:
                ax1, ay1, ax2, ay2 = map(int, attn_box.xyxy[0])
                attn_class = int(attn_box.cls[0])
                att_conf = float(attn_box.conf[0])
                self._current_attn_boxes.append(((ax1, ay1, ax2, ay2), attn_class, att_conf))
        
        return self._current_attn_boxes

    def fuse_detection_attention(self, detection_boxes, frame=None):
        """
        Fuse detection boxes with attention boxes using IoU.
        Args:
            detection_boxes: List of (x1, y1, x2, y2, conf, centroid) from detection
            frame: Optional frame for annotation
        Returns:
            List of detection boxes with assigned attention labels
        """
        self._current_detection_boxes = detection_boxes
        self._current_fusion_results = []
        
        for x1, y1, x2, y2, conf, centroid in detection_boxes:
            attention_label = "inattentive"
            best_iou = 0
            best_attn_class = None
            for (ax1, ay1, ax2, ay2), cls, _ in self._current_attn_boxes:
                iou = compute_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                if iou > best_iou and iou > self._iou_threshold:
                    best_iou = iou
                    best_attn_class = cls
            if best_attn_class is not None:
                names = self._attention_model.names
                if isinstance(names, dict):
                    attention_label = names.get(best_attn_class, str(best_attn_class))
                else:
                    attention_label = names[best_attn_class]
                if attention_label == "unattentive":
                    attention_label = "inattentive"
            # Store fusion result
            fusion_result = {
                'bbox': (x1, y1, x2, y2),
                'centroid': centroid,
                'detection_conf': conf,
                'attention_label': attention_label,
                'attention_class': best_attn_class,
                'attention_conf': 0, # Confidence is not used for label assignment
                'iou_score': best_iou
            }
            self._current_fusion_results.append(fusion_result)
        
        # Annotate frame if provided
        if frame is not None:
            self._current_annotated_frame = self._annotate_frame(frame.copy())
        
        return self._current_fusion_results

    def assign_attention_to_faces(self, face_results):
        """
        Assign attention labels to face recognition results.
        Args:
            face_results: List of face recognition results from FaceRecognitionSystem
        Returns:
            Updated face results with attention labels
        """
        updated_results = []
        
        for face_result in face_results:
            x1, y1, x2, y2 = face_result['bbox']
            
            # Find best matching attention box
            attention_label = "inattentive"
            best_iou = 0
            best_attn_class = None
            for (ax1, ay1, ax2, ay2), cls, _ in self._current_attn_boxes:
                iou = compute_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                if iou > best_iou and iou > self._iou_threshold:
                    best_iou = iou
                    best_attn_class = cls
            if best_attn_class is not None:
                names = self._attention_model.names
                if isinstance(names, dict):
                    attention_label = names.get(best_attn_class, str(best_attn_class))
                else:
                    attention_label = names[best_attn_class]
                if attention_label == "unattentive":
                    attention_label = "inattentive"
            
            # Update face result with attention info
            updated_result = face_result.copy()
            updated_result.update({
                'attention_label': attention_label,
                'attention_class': best_attn_class,
                'attention_conf': 0, # Confidence is not used for label assignment
                'iou_score': best_iou
            })
            updated_results.append(updated_result)
        
        return updated_results

    def _annotate_frame(self, frame):
        """Annotate frame with fusion results."""
        for result in self._current_fusion_results:
            x1, y1, x2, y2 = result['bbox']
            attention_label = result['attention_label']
            detection_conf = result['detection_conf']
            iou_score = result['iou_score']
            
            label = f'{attention_label} | Conf: {detection_conf:.2f} | IoU: {iou_score:.2f}'
            color = self.color_map.get(attention_label, (128, 128, 128))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

    def annotate_faces_with_attention(self, frame, face_results):
        """
        Annotate frame with face recognition results that include attention labels.
        Args:
            frame: Input frame
            face_results: Face recognition results with attention labels
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in face_results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            matched_id = result['id']
            recog_conf = result['confidence']
            attention_label = result.get('attention_label', 'Unknown')
            
            label = f'{name} ({matched_id}) {recog_conf:.2f} | {attention_label}'
            color = self.color_map.get(attention_label, (128, 128, 128))
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame

    # ========== GETTERS ==========
    
    @property
    def current_attention_boxes(self):
        """Get current attention boxes."""
        return self._current_attn_boxes
    
    @property
    def current_detection_boxes(self):
        """Get current detection boxes."""
        return self._current_detection_boxes
    
    @property
    def current_fusion_results(self):
        """Get current fusion results."""
        return self._current_fusion_results
    
    @property
    def current_annotated_frame(self):
        """Get current annotated frame."""
        return self._current_annotated_frame
    
    @property
    def attention_model(self):
        """Get the attention model."""
        return self._attention_model
    
    @property
    def color_mapping(self):
        """Get the color mapping for visualization."""
        return self.color_map
    
    @property
    def iou_threshold(self):
        """Get the IoU threshold."""
        return self._iou_threshold
    
    def get_attention_stats(self):
        """Get statistics about current attention detections."""
        if not self._current_attn_boxes:
            return {}
        
        attention_counts = {}
        for (_, _, _), cls, conf in self._current_attn_boxes:
            label = self._attention_model.names[cls] if hasattr(self._attention_model, 'names') else str(cls)
            if label not in attention_counts:
                attention_counts[label] = {'count': 0, 'avg_conf': 0, 'confs': []}
            attention_counts[label]['count'] += 1
            attention_counts[label]['confs'].append(conf)
        
        # Calculate average confidence
        for label in attention_counts:
            attention_counts[label]['avg_conf'] = np.mean(attention_counts[label]['confs'])
        
        return attention_counts
    
    def get_fusion_stats(self):
        """Get statistics about current fusion results."""
        if not self._current_fusion_results:
            return {}
        
        stats = {
            'total_detections': len(self._current_fusion_results),
            'attention_assignments': {},
            'avg_iou': 0,
            'iou_scores': []
        }
        
        attention_counts = {}
        for result in self._current_fusion_results:
            label = result['attention_label']
            if label not in attention_counts:
                attention_counts[label] = 0
            attention_counts[label] += 1
            stats['iou_scores'].append(result['iou_score'])
        
        stats['attention_assignments'] = attention_counts
        stats['avg_iou'] = np.mean(stats['iou_scores']) if stats['iou_scores'] else 0
        
        return stats

    @property
    def names(self):
        """Get the class names from the attention model."""
        return self.model.names if hasattr(self.model, 'names') else {}
