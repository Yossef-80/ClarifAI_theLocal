from Capture import VideoCaptureFrameProvider
from Detection import YOLODetectionModel
from Attention import YOLOAttentionModel
from Face_recognition import FaceRecognitionSystem
from Fusion import FusionSystem
from Display import DisplayFrame
import torch
import numpy as np
import cv2
import time
from time import perf_counter

class CompleteSystem:
    """
    Complete system that orchestrates all modules for face detection, tracking, recognition, and attention analysis.
    """
    def __init__(self, video_path, face_db_path, detection_weights_path, attention_weights_path):
        # Initialize all modules
        self.frame_provider = VideoCaptureFrameProvider(video_path)
        self.detection = YOLODetectionModel(detection_weights_path)
        self.attention_model = YOLOAttentionModel(attention_weights_path)
        self.recognition_system = FaceRecognitionSystem(
            face_db_path=face_db_path,
            device='cuda',
            max_distance=40,
            max_history=20,
            max_missing_frames=10
        )
        self.fusion_system = FusionSystem(self.attention_model)
        self.display = DisplayFrame("Complete System")
        
        # Performance tracking
        self.frame_count = 0
        self.performance_stats = {}
        
    def process_frame(self):
        """Process a single frame through all modules."""
        start_time = time.perf_counter()
        
        # ========== MODULE 1: CAPTURE ==========
        t6 = perf_counter()
        frame = self.frame_provider.get_frame()
        if frame is None:
            return False
        self.frame_count += 1
        frame_read_time = perf_counter() - t6
        
        # ========== MODULE 2: DETECTION ==========
        t0 = perf_counter()
        self.detection.detect(frame, verbose=False, imgsz=1280)
        detection_boxes = self.detection.current_detection_boxes
        detection_model_time = perf_counter() - t0
        
        # ========== MODULE 3: ATTENTION ==========
        t_attn = perf_counter()
        self.attention_model.predict(frame, verbose=False, imgsz=960)
        attention_boxes = self.attention_model.current_attention_boxes
        attention_time = perf_counter() - t_attn
        
        print("Attention boxes:", self.attention_model.current_attention_boxes)
        
        # ========== MODULE 4: FACE RECOGNITION & TRACKING ==========
        t1 = perf_counter()
        annotated_frame, face_results = self.recognition_system.process_frame(frame, detection_boxes, self.frame_count)
        recognition_time = perf_counter() - t1
        
        # ========== MODULE 5: FUSION (ATTENTION ASSIGNMENT) ==========
        t2 = perf_counter()
        fusion_results = self.fusion_system.assign_attention_to_faces(face_results)
        fusion_time = perf_counter() - t2
        
        # ========== MODULE 6: ANNOTATION ==========
        t3 = perf_counter()
        final_frame = self.fusion_system.annotate_faces_with_attention(frame, fusion_results)
        annotation_time = perf_counter() - t3
        
        # ========== MODULE 7: DISPLAY ==========
        t4 = perf_counter()
        key = self.display.display_frame(final_frame)
        display_time = perf_counter() - t4
        
        # ========== PERFORMANCE LOGGING ==========
        elapsed_time = time.perf_counter() - start_time
        
        # Store performance stats
        self.performance_stats = {
            'frame_read_time': frame_read_time,
            'detection_time': detection_model_time,
            'attention_time': attention_time,
            'recognition_time': recognition_time,
            'fusion_time': fusion_time,
            'annotation_time': annotation_time,
            'display_time': display_time,
            'total_time': elapsed_time,
            'detection_count': len(detection_boxes),
            'attention_count': len(attention_boxes),
            'face_count': len(fusion_results)
        }
        
        # Print performance info
        self._print_performance_info()
        
        return key != ord('q')
    
    def _print_performance_info(self):
        """Print performance information for the current frame."""
        progress = self.frame_provider.get_progress()
        current_frame_num = self.frame_provider.current_frame_number
        tracking_stats = self.recognition_system.get_tracking_stats()
        
        print(f"Frame {current_frame_num}/{self.frame_provider.frame_count} ({progress:.1f}%)")
        print(f"Detection: {self.performance_stats['detection_count']} faces | Time: {self.performance_stats['detection_time']*1000:.1f}ms")
        print(f"Attention: {self.performance_stats['attention_count']} regions | Time: {self.performance_stats['attention_time']*1000:.1f}ms")
        print(f"Recognition: {self.performance_stats['recognition_time']*1000:.1f}ms")
        print(f"Fusion: {self.performance_stats['fusion_time']*1000:.1f}ms")
        print(f"Tracking: {tracking_stats['total_tracks']} total, {tracking_stats['active_tracks']} active")
        print(f"Total: {self.performance_stats['total_time']*1000:.1f}ms")
        print("-" * 50)
    
    def run(self):
        """Run the complete system."""
        try:
            while True:
                if not self.process_frame():
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.frame_provider.release()
        self.display.close()
        cv2.destroyAllWindows()
        print("System shutdown complete")

def main():
    """Main function to run the complete system."""
    # Configuration
    video_path = "../../Source Video/combined videos.mp4"
    face_db_path = "../../AI_VID_face_db.pkl"
    detection_weights_path = "../../yolo_detection_model/yolov11s-face.pt"
    attention_weights_path = "../../yolo_attention_model/attention_weights_16july2025.pt"
    
    # Create and run the complete system
    system = CompleteSystem(
        video_path=video_path,
        face_db_path=face_db_path,
        detection_weights_path=detection_weights_path,
        attention_weights_path=attention_weights_path
    )
    
    print("Attention model names:", system.attention_model.names)
    
    system.run()

if __name__ == "__main__":
    main()
