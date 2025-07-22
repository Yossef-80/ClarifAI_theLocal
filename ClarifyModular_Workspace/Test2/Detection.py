from ultralytics import YOLO
import numpy as np

class YOLODetectionModel:
    """
    Loads a YOLO model and performs detection on frames, returning boxes, classes, and confidences.
    Communicates with other modules through getters.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        
        # Internal state for communication
        self._current_boxes = []
        self._current_classes = []
        self._current_confidences = []
        self._current_detection_boxes = []  # Formatted for other modules
        self._last_frame = None
        self._last_frame_count = 0

    def detect(self, frame, verbose=False, imgsz=1280):
        """
        Perform detection on frame and store results internally.
        Args:
            frame: Input frame
            verbose: Verbose output
            imgsz: Image size
        """
        self._last_frame = frame.copy()
        results = self.model(frame, verbose=verbose, imgsz=imgsz)
        
        if not results or len(results) == 0:
            self._current_boxes = []
            self._current_classes = []
            self._current_confidences = []
            self._current_detection_boxes = []
            return None, None, None
            
        boxes = results[0].boxes
        self._current_boxes = boxes
        
        if boxes is not None and len(boxes) > 0:
            # Convert to numpy arrays for easier handling
            self._current_classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            self._current_confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            
            # Format detection boxes for other modules (x1, y1, x2, y2, conf, centroid)
            self._current_detection_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                self._current_detection_boxes.append((x1, y1, x2, y2, conf, centroid))
        else:
            self._current_classes = []
            self._current_confidences = []
            self._current_detection_boxes = []
        
        return self._current_boxes, self._current_classes, self._current_confidences

    # ========== GETTERS ==========
    
    @property
    def current_boxes(self):
        """Get current detection boxes (raw YOLO format)."""
        return self._current_boxes
    
    @property
    def current_classes(self):
        """Get current detection classes."""
        return self._current_classes
    
    @property
    def current_confidences(self):
        """Get current detection confidences."""
        return self._current_confidences
    
    @property
    def current_detection_boxes(self):
        """Get current detection boxes formatted for other modules (x1, y1, x2, y2, conf, centroid)."""
        return self._current_detection_boxes
    
    @property
    def last_frame(self):
        """Get the last processed frame."""
        return self._last_frame
    
    @property
    def detection_model(self):
        """Get the YOLO detection model."""
        return self.model
    
    def get_detection_stats(self):
        """Get statistics about current detections."""
        if not self._current_detection_boxes:
            return {
                'num_detections': 0,
                'avg_confidence': 0,
                'detection_areas': []
            }
        
        confidences = [box[4] for box in self._current_detection_boxes]
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in self._current_detection_boxes]
        
        return {
            'num_detections': len(self._current_detection_boxes),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'detection_areas': areas
        }
