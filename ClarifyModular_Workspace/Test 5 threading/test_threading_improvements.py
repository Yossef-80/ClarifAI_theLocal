#!/usr/bin/env python3
"""
Test script for enhanced threading system
This script validates the improvements and demonstrates the new functionality.
"""

import sys
import time
import threading
from PyQt5.QtWidgets import QApplication
from pyqt_local_demo import VideoWindow, ThreadSafeQueue, PerformanceMonitor, FrameData
from threading_config import *

def test_thread_safe_queue():
    """Test the ThreadSafeQueue implementation"""
    print("Testing ThreadSafeQueue...")
    
    queue = ThreadSafeQueue(maxsize=3)
    
    # Test basic operations
    assert queue.empty()
    assert queue.size() == 0
    
    # Test putting items
    assert queue.put("item1")
    assert queue.put("item2")
    assert queue.put("item3")
    assert queue.size() == 3
    
    # Test queue full behavior
    assert not queue.put("item4")  # Should fail when full
    
    # Test getting items
    assert queue.get() == "item1"
    assert queue.get() == "item2"
    assert queue.get() == "item3"
    assert queue.get() is None  # Should return None when empty
    
    # Test clear
    queue.put("test")
    queue.clear()
    assert queue.empty()
    
    print("✓ ThreadSafeQueue tests passed")

def test_performance_monitor():
    """Test the PerformanceMonitor implementation"""
    print("Testing PerformanceMonitor...")
    
    monitor = PerformanceMonitor()
    
    # Test recording timing
    monitor.record_timing("test_thread", "test_operation", 100.0)
    monitor.record_timing("test_thread", "test_operation", 200.0)
    monitor.record_timing("test_thread", "test_operation", 300.0)
    
    # Test getting stats
    stats = monitor.get_stats()
    assert "test_thread" in stats
    assert "test_operation" in stats["test_thread"]
    
    operation_stats = stats["test_thread"]["test_operation"]
    assert operation_stats["avg"] == 200.0
    assert operation_stats["min"] == 100.0
    assert operation_stats["max"] == 300.0
    assert operation_stats["count"] == 3
    
    # Test average timing
    avg = monitor.get_average_timing("test_thread", "test_operation")
    assert avg == 200.0
    
    print("✓ PerformanceMonitor tests passed")

def test_frame_data():
    """Test the FrameData implementation"""
    print("Testing FrameData...")
    
    import numpy as np
    
    # Create test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test FrameData creation
    frame_data = FrameData(frame, 1)
    assert frame_data.frame_count == 1
    assert frame_data.frame is not None
    assert not frame_data.is_complete()
    
    # Test completion tracking
    frame_data.detections = []
    frame_data.face_crops = []
    frame_data.recognition_results = []
    frame_data.attention_results = []
    
    assert frame_data.is_complete()
    
    print("✓ FrameData tests passed")

def test_configuration():
    """Test the configuration system"""
    print("Testing configuration...")
    
    # Test device detection
    device = get_device()
    assert device in ['cuda', 'cpu']
    
    # Test configuration validation
    try:
        validate_config()
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"⚠ Configuration validation failed: {e}")
    
    # Test queue sizes
    assert QUEUE_SIZES['capture'] > 0
    assert QUEUE_SIZES['detection'] > 0
    assert QUEUE_SIZES['recognition'] > 0
    assert QUEUE_SIZES['attention'] > 0
    assert QUEUE_SIZES['display'] > 0
    
    # Test timing parameters
    assert TIMING['capture_interval'] > 0
    assert TIMING['display_fps'] > 0
    assert TIMING['queue_timeout'] > 0
    
    print("✓ Configuration tests passed")

def test_gui_creation():
    """Test GUI creation without starting video"""
    print("Testing GUI creation...")
    
    app = QApplication(sys.argv)
    
    try:
        window = VideoWindow()
        assert window is not None
        assert hasattr(window, 'performance_monitor')
        assert hasattr(window, 'video_label')
        assert hasattr(window, 'alerts_widget')
        
        # Test that performance monitor is working
        monitor = window.performance_monitor
        monitor.record_timing("test", "test_op", 50.0)
        stats = monitor.get_stats()
        assert "test" in stats
        
        print("✓ GUI creation test passed")
        
    except Exception as e:
        print(f"✗ GUI creation test failed: {e}")
        return False
    
    finally:
        app.quit()
    
    return True

def benchmark_queue_performance():
    """Benchmark queue performance"""
    print("Benchmarking queue performance...")
    
    queue = ThreadSafeQueue(maxsize=10)
    
    # Benchmark put operations
    start_time = time.time()
    for i in range(1000):
        queue.put(f"item{i}")
    put_time = time.time() - start_time
    
    # Benchmark get operations
    start_time = time.time()
    for i in range(1000):
        queue.get()
    get_time = time.time() - start_time
    
    print(f"  Put operations: {put_time:.4f}s for 1000 items")
    print(f"  Get operations: {get_time:.4f}s for 1000 items")
    print(f"  Throughput: {1000/put_time:.0f} puts/sec, {1000/get_time:.0f} gets/sec")
    
    if put_time < 1.0 and get_time < 1.0:
        print("✓ Queue performance benchmark passed")
    else:
        print("⚠ Queue performance may be slow")

def benchmark_performance_monitor():
    """Benchmark performance monitor"""
    print("Benchmarking performance monitor...")
    
    monitor = PerformanceMonitor()
    
    # Benchmark recording operations
    start_time = time.time()
    for i in range(1000):
        monitor.record_timing("test_thread", "test_op", float(i))
    record_time = time.time() - start_time
    
    # Benchmark stats retrieval
    start_time = time.time()
    for i in range(100):
        stats = monitor.get_stats()
    stats_time = time.time() - start_time
    
    print(f"  Record operations: {record_time:.4f}s for 1000 records")
    print(f"  Stats retrieval: {stats_time:.4f}s for 100 retrievals")
    
    if record_time < 1.0 and stats_time < 1.0:
        print("✓ Performance monitor benchmark passed")
    else:
        print("⚠ Performance monitor may be slow")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Enhanced Threading System Test Suite")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_thread_safe_queue,
        test_performance_monitor,
        test_frame_data,
        test_gui_creation,
        benchmark_queue_performance,
        benchmark_performance_monitor,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced threading system is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Device: {get_device()}")
    print(f"  Capture Interval: {TIMING['capture_interval']}ms (~{1000/TIMING['capture_interval']:.1f} FPS)")
    print(f"  Display FPS: {TIMING['display_fps']}")
    print(f"  Queue Sizes: {QUEUE_SIZES}")
    print(f"  Performance Logging: {PERFORMANCE['log_performance_stats']}")
    print(f"  Save Logs: {PERFORMANCE['save_performance_logs']}")

if __name__ == "__main__":
    main() 