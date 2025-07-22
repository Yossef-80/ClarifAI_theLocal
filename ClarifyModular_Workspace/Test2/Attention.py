# Attention.py

from ultralytics import YOLO
import cv2
import numpy as np

class YOLOAttentionModel:
    """
    Wrapper for Ultralytics YOLO model.
    Communicates with other modules through getters.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        
        # Internal state for communication
        self._current_boxes = []
        self._current_classes = []
        self._current_confidences = []
        self._current_attention_boxes = []  # Formatted for other modules
        self._last_frame = None

    def predict(self, frame, verbose=False, imgsz=960):
        """
        Perform attention prediction on frame and store results internally.
        Args:
            frame: Input frame
            verbose: Verbose output
            imgsz: Image size
        Returns:
            boxes, classes, confidences (for backward compatibility)
        """
        self._last_frame = frame.copy()
        results = self.model(frame, verbose=verbose, imgsz=imgsz)
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            self._current_boxes = boxes
            
            # Extract class and confidence for each detection
            self._current_classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            self._current_confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            
            # Format attention boxes for other modules ((x1, y1, x2, y2), class, conf)
            self._current_attention_boxes = []
            for i, box in enumerate(boxes):
                ax1, ay1, ax2, ay2 = map(int, box.xyxy[0])
                attn_class = int(box.cls[0]) if hasattr(box, 'cls') else 0
                att_conf = float(box.conf[0]) if hasattr(box, 'conf') else 0
                self._current_attention_boxes.append(((ax1, ay1, ax2, ay2), attn_class, att_conf))
        else:
            self._current_boxes = []
            self._current_classes = []
            self._current_confidences = []
            self._current_attention_boxes = []
        
        return self._current_boxes, self._current_classes, self._current_confidences

    # ========== GETTERS ==========
    
    @property
    def current_boxes(self):
        """Get current attention boxes (raw YOLO format)."""
        return self._current_boxes
    
    @property
    def current_classes(self):
        """Get current attention classes."""
        return self._current_classes
    
    @property
    def current_confidences(self):
        """Get current attention confidences."""
        return self._current_confidences
    
    @property
    def current_attention_boxes(self):
        """Get current attention boxes formatted for other modules ((x1, y1, x2, y2), class, conf)."""
        return self._current_attention_boxes
    
    @property
    def last_frame(self):
        """Get the last processed frame."""
        return self._last_frame
    
    @property
    def attention_model(self):
        """Get the YOLO attention model."""
        return self.model
    
    @property
    def names(self):
        """Get the class names from the attention model."""
        return self.model.names if hasattr(self.model, 'names') else {}
    
    def get_attention_stats(self):
        """Get statistics about current attention detections."""
        if not self._current_attention_boxes:
            return {
                'num_detections': 0,
                'avg_confidence': 0,
                'class_distribution': {}
            }
        
        confidences = [box[2] for box in self._current_attention_boxes]
        classes = [box[1] for box in self._current_attention_boxes]
        
        # Count class distribution
        class_distribution = {}
        for cls in classes:
            class_name = self.names.get(cls, str(cls))
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        return {
            'num_detections': len(self._current_attention_boxes),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'class_distribution': class_distribution
        }


class Attention:
    """
    Attention model/controller that retrieves frames and passes them to YOLO.
    Communicates with other modules through getters.
    """
    def __init__(self, frame_provider, yolo_model):
        self.frame_provider = frame_provider
        self.yolo_model = yolo_model

    def process(self, verbose=False, imgsz=960):
        """
        Process frame with attention model and store results internally.
        Args:
            verbose: Verbose output
            imgsz: Image size
        Returns:
            yolo_output (for backward compatibility)
        """
        # Retrieve frame from the provider
        frame = self.frame_provider.get_frame()
        if frame is None:
            print("No frame retrieved.")
            return None

        # Pass frame to YOLO model
        yolo_output = self.yolo_model.predict(frame, verbose=verbose, imgsz=imgsz)
        return yolo_output

    # ========== GETTERS ==========
    
    @property
    def current_attention_boxes(self):
        """Get current attention boxes from the YOLO model."""
        return self.yolo_model.current_attention_boxes
    
    @property
    def current_boxes(self):
        """Get current attention boxes (raw format)."""
        return self.yolo_model.current_boxes
    
    @property
    def current_classes(self):
        """Get current attention classes."""
        return self.yolo_model.current_classes
    
    @property
    def current_confidences(self):
        """Get current attention confidences."""
        return self.yolo_model.current_confidences
    
    @property
    def attention_model(self):
        """Get the attention model."""
        return self.yolo_model.attention_model
    
    @property
    def names(self):
        """Get the class names from the attention model."""
        return self.yolo_model.names
    
    def get_attention_stats(self):
        """Get statistics about current attention detections."""
        return self.yolo_model.get_attention_stats()
