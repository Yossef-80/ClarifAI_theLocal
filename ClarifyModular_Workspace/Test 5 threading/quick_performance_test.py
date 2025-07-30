#!/usr/bin/env python3
"""
Quick Performance Test for Optimized Threading System
This script tests the basic functionality and performance of the optimized system.
"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from pyqt_local_demo import VideoWindow, SimpleQueue, FrameData
from threading_config import *

def test_simple_queue():
    """Test the simplified queue performance"""
    print("Testing SimpleQueue performance...")
    
    queue = SimpleQueue(maxsize=3)
    
    # Test basic operations
    start_time = time.time()
    for i in range(1000):
        queue.put(f"item{i}")
        item = queue.get()
        if item != f"item{i}":
            print(f"Queue test failed: expected item{i}, got {item}")
            return False
    
    duration = time.time() - start_time
    print(f"  Queue operations: {duration:.4f}s for 1000 operations")
    print(f"  Throughput: {1000/duration:.0f} ops/sec")
    
    if duration < 0.1:  # Should be very fast
        print("✓ SimpleQueue performance test passed")
        return True
    else:
        print("✗ SimpleQueue performance test failed - too slow")
        return False

def test_frame_data():
    """Test FrameData creation and operations"""
    print("Testing FrameData operations...")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    for i in range(100):
        frame_data = FrameData(frame, i)
        frame_data.detections = []
        frame_data.face_crops = []
        frame_data.recognition_results = []
        frame_data.attention_results = []
        is_complete = frame_data.is_complete()
        if not is_complete:
            print("FrameData completion test failed")
            return False
    
    duration = time.time() - start_time
    print(f"  FrameData operations: {duration:.4f}s for 100 operations")
    
    if duration < 0.01:  # Should be very fast
        print("✓ FrameData performance test passed")
        return True
    else:
        print("✗ FrameData performance test failed - too slow")
        return False

def test_gui_creation():
    """Test GUI creation without starting video"""
    print("Testing GUI creation...")
    
    app = QApplication(sys.argv)
    
    try:
        start_time = time.time()
        window = VideoWindow()
        creation_time = time.time() - start_time
        
        print(f"  GUI creation: {creation_time:.4f}s")
        
        if creation_time < 1.0:  # Should be reasonably fast
            print("✓ GUI creation test passed")
            return True
        else:
            print("✗ GUI creation test failed - too slow")
            return False
            
    except Exception as e:
        print(f"✗ GUI creation test failed: {e}")
        return False
    finally:
        app.quit()

def test_configuration():
    """Test configuration loading"""
    print("Testing configuration...")
    
    try:
        # Test device detection
        device = get_device()
        print(f"  Device: {device}")
        
        # Test configuration validation
        validate_config()
        print("  Configuration validation: passed")
        
        # Test queue sizes
        print(f"  Queue sizes: {QUEUE_SIZES}")
        print(f"  Capture interval: {TIMING['capture_interval']}ms")
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run performance tests"""
    print("=" * 50)
    print("Quick Performance Test for Optimized Threading System")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_simple_queue,
        test_frame_data,
        test_gui_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance tests passed!")
        print("The optimized threading system should now provide better FPS.")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    print("=" * 50)
    
    # Performance recommendations
    print("\nPerformance Recommendations:")
    print("1. If FPS is still low, try increasing capture_interval in threading_config.py")
    print("2. If memory usage is high, reduce queue sizes in threading_config.py")
    print("3. For real-time applications, set capture_interval to 33ms (~30 FPS)")
    print("4. For better accuracy, set capture_interval to 50ms (~20 FPS)")

if __name__ == "__main__":
    main() 