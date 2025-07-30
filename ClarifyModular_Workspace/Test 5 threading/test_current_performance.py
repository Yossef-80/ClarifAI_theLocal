#!/usr/bin/env python3
"""
Test Current Performance
This script tests the current performance of the optimized threading system.
"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from pyqt_local_demo import VideoWindow, SimpleQueue, FrameData
from threading_config import *

def test_performance():
    """Test the current performance"""
    print("=" * 60)
    print("Current Performance Test")
    print("=" * 60)
    
    print(f"Configuration:")
    print(f"  Capture Interval: {TIMING['capture_interval']}ms (~{1000/TIMING['capture_interval']:.1f} FPS)")
    print(f"  Queue Sizes: {QUEUE_SIZES}")
    print(f"  Device: {get_device()}")
    print()
    
    # Test queue performance
    print("Testing queue performance...")
    queue = SimpleQueue(maxsize=3)
    
    start_time = time.time()
    for i in range(1000):
        queue.put(f"item{i}")
        item = queue.get()
    queue_time = time.time() - start_time
    
    print(f"  Queue throughput: {1000/queue_time:.0f} ops/sec")
    print(f"  Queue time per operation: {queue_time/1000*1000:.3f}ms")
    print()
    
    # Test frame data creation
    print("Testing frame data creation...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    for i in range(100):
        frame_data = FrameData(frame, i)
        frame_data.detections = []
        frame_data.face_crops = []
        frame_data.recognition_results = []
        frame_data.attention_results = []
        is_complete = frame_data.is_complete()
    frame_time = time.time() - start_time
    
    print(f"  Frame data throughput: {100/frame_time:.0f} ops/sec")
    print(f"  Frame data time per operation: {frame_time/100*1000:.3f}ms")
    print()
    
    # Test GUI creation
    print("Testing GUI creation...")
    app = QApplication(sys.argv)
    
    start_time = time.time()
    window = VideoWindow()
    gui_time = time.time() - start_time
    
    print(f"  GUI creation time: {gui_time:.3f}s")
    print()
    
    app.quit()
    
    # Performance analysis
    print("Performance Analysis:")
    print(f"  Queue operations: {'✓ Fast' if queue_time < 0.1 else '⚠ Slow'}")
    print(f"  Frame data operations: {'✓ Fast' if frame_time < 0.01 else '⚠ Slow'}")
    print(f"  GUI creation: {'✓ Fast' if gui_time < 1.0 else '⚠ Slow'}")
    print()
    
    # Recommendations
    print("Recommendations:")
    if TIMING['capture_interval'] < 40:
        print("  ✓ Capture interval is optimized for performance")
    else:
        print("  ⚠ Consider reducing capture_interval for higher FPS")
    
    if QUEUE_SIZES['capture'] <= 3:
        print("  ✓ Queue sizes are optimized for memory usage")
    else:
        print("  ⚠ Consider reducing queue sizes for lower memory usage")
    
    print()
    print("Expected Performance:")
    print(f"  Target FPS: {1000/TIMING['capture_interval']:.1f}")
    print(f"  Expected frame time: {TIMING['capture_interval']}ms")
    print(f"  Memory usage: Low (minimal queues)")
    print(f"  CPU usage: Optimized (no performance monitoring overhead)")
    
    print("=" * 60)

if __name__ == "__main__":
    test_performance() 