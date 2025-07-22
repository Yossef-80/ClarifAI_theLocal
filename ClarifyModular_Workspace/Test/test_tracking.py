#!/usr/bin/env python3
"""
Test script to verify improved tracking system.
This script helps compare the original vs improved tracking performance.
"""

import cv2
import numpy as np
import time
from Face_recognition import FaceRecognitionSystem
from Detection import YOLODetectionModel
from Capture import VideoCaptureFrameProvider

def test_tracking_performance(video_path, face_db_path, detection_weights_path, num_frames=100):
    """
    Test tracking performance on a video.
    Args:
        video_path: Path to test video
        face_db_path: Path to face database
        detection_weights_path: Path to detection model weights
        num_frames: Number of frames to test
    """
    print("=== Testing Tracking System ===")
    
    # Initialize systems
    frame_provider = VideoCaptureFrameProvider(video_path)
    detection = YOLODetectionModel(detection_weights_path)
    recognition_system = FaceRecognitionSystem(
        face_db_path=face_db_path,
        device='cuda',
        max_distance=40,
        max_history=20,
        max_missing_frames=10
    )
    
    # Tracking metrics
    id_switches = 0
    total_detections = 0
    frame_count = 0
    tracking_history = {}  # Track ID assignments over time
    
    try:
        while frame_count < num_frames:
            frame = frame_provider.get_frame()
            if frame is None:
                break
                
            frame_count += 1
            
            # Detect faces
            detection.detect(frame, verbose=False, imgsz=1280)
            detection_boxes = detection.current_detection_boxes
            
            # Track faces using process_frame method
            annotated_frame, face_results = recognition_system.process_frame(frame, detection_boxes, frame_count)
            current_assignments = {result['centroid']: result['id'] for result in face_results}
            total_detections += len(face_results)
            
            # Check for ID switches
            if frame_count > 1:
                for centroid, current_id in current_assignments.items():
                    if centroid in tracking_history.get(frame_count - 1, {}):
                        previous_id = tracking_history[frame_count - 1][centroid]
                        if current_id != previous_id:
                            id_switches += 1
                            print(f"ID Switch detected at frame {frame_count}: {previous_id} -> {current_id}")
            
            tracking_history[frame_count] = current_assignments
            
            # Print progress
            if frame_count % 10 == 0:
                tracking_stats = recognition_system.get_tracking_stats()
                print(f"Frame {frame_count}: {len(detection_boxes)} detections, {tracking_stats['total_tracks']} tracks")
        
        # Calculate metrics
        switch_rate = id_switches / max(total_detections, 1) * 100
        print(f"\n=== Tracking Performance Results ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"ID switches: {id_switches}")
        print(f"Switch rate: {switch_rate:.2f}%")
        print(f"Final active tracks: {recognition_system.get_tracking_stats()['total_tracks']}")
        
        return {
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'id_switches': id_switches,
            'switch_rate': switch_rate,
            'final_tracks': recognition_system.get_tracking_stats()['total_tracks']
        }
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None
    finally:
        frame_provider.release()

def compare_tracking_methods(video_path, face_db_path, detection_weights_path, num_frames=50):
    """
    Compare original vs improved tracking methods.
    """
    print("=== Comparing Tracking Methods ===")
    
    # Test improved tracking
    print("\n1. Testing Improved Tracking...")
    improved_results = test_tracking_performance(video_path, face_db_path, detection_weights_path, num_frames)
    
    if improved_results:
        print(f"\nImproved Tracking Results:")
        print(f"  - ID Switch Rate: {improved_results['switch_rate']:.2f}%")
        print(f"  - Final Tracks: {improved_results['final_tracks']}")
    
    print("\n=== Comparison Complete ===")
    print("Lower ID switch rates indicate better tracking stability.")
    print("The improved system should show fewer ID switches and more stable tracking.")

if __name__ == "__main__":
    # Update these paths to match your setup
    video_path = "../../Source Video/combined videos.mp4"
    face_db_path = "../../AI_VID_face_db.pkl"
    detection_weights_path = "../../yolo_detection_model/yolov11s-face.pt"
    
    # Run comparison
    compare_tracking_methods(video_path, face_db_path, detection_weights_path, num_frames=50) 