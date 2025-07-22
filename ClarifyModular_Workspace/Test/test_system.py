#!/usr/bin/env python3
"""
Test script to verify the CompleteSystem class works correctly.
"""

from Main import CompleteSystem
import time

def test_system():
    """Test the complete system with a few frames."""
    print("=== Testing CompleteSystem ===")
    
    # Configuration
    video_path = "../../Source Video/combined videos.mp4"
    face_db_path = "../../AI_VID_face_db.pkl"
    detection_weights_path = "../../yolo_detection_model/yolov11s-face.pt"
    attention_weights_path = "../../yolo_attention_model/attention_weights_16july2025.pt"
    
    try:
        # Create the complete system
        system = CompleteSystem(
            video_path=video_path,
            face_db_path=face_db_path,
            detection_weights_path=detection_weights_path,
            attention_weights_path=attention_weights_path
        )
        
        print("System initialized successfully!")
        print("Processing first 10 frames...")
        
        # Process a few frames to test
        frame_count = 0
        max_frames = 10
        
        while frame_count < max_frames:
            success = system.process_frame()
            if not success:
                print("End of video reached")
                break
            frame_count += 1
            time.sleep(0.1)  # Small delay to see the output
        
        print(f"Successfully processed {frame_count} frames!")
        print("System test completed successfully!")
        
    except Exception as e:
        print(f"Error during system test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system() 