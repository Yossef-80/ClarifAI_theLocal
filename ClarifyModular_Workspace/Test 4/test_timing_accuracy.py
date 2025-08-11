#!/usr/bin/env python3
"""
Test script to verify timing accuracy
This script tests the precision and accuracy of our timing measurements
"""

import time
import threading
from collections import deque
import statistics
import pandas as pd

def test_timing_precision():
    """Test the precision of time.time() measurements"""
    print("Testing timing precision...")
    
    # Test 1: Measure the overhead of time.time() calls
    overhead_times = []
    for i in range(1000):
        start = time.time()
        end = time.time()
        overhead_times.append((end - start) * 1000000)  # Convert to microseconds
    
    avg_overhead = statistics.mean(overhead_times)
    print(f"Average overhead of time.time() calls: {avg_overhead:.3f} microseconds")
    
    # Test 2: Measure known sleep durations
    known_durations = [0.001, 0.005, 0.010, 0.050, 0.100]  # 1ms, 5ms, 10ms, 50ms, 100ms
    measured_durations = []
    
    for duration in known_durations:
        times = []
        for i in range(10):
            start = time.time()
            time.sleep(duration)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        avg_measured = statistics.mean(times)
        error = abs(avg_measured - duration * 1000)
        error_percent = (error / (duration * 1000)) * 100
        
        measured_durations.append({
            'Expected_ms': duration * 1000,
            'Measured_ms': avg_measured,
            'Error_ms': error,
            'Error_percent': error_percent
        })
        
        print(f"Expected: {duration*1000:6.1f}ms, Measured: {avg_measured:6.1f}ms, Error: {error:6.2f}ms ({error_percent:5.1f}%)")
    
    return measured_durations

def test_thread_timing():
    """Test thread timing measurements"""
    print("\nTesting thread timing measurements...")
    
    timing_data = []
    lock = threading.Lock()
    
    def worker_thread(thread_id, work_duration):
        """Worker thread that does some work for a specified duration"""
        start_time = time.time()
        
        # Simulate work
        time.sleep(work_duration)
        
        end_time = time.time()
        
        with lock:
            timing_data.append({
                'thread_id': thread_id,
                'expected_duration_ms': work_duration * 1000,
                'measured_duration_ms': (end_time - start_time) * 1000,
                'start_time': start_time,
                'end_time': end_time
            })
    
    # Create threads with different work durations
    threads = []
    work_durations = [0.005, 0.010, 0.020, 0.050]  # 5ms, 10ms, 20ms, 50ms
    
    for i, duration in enumerate(work_durations):
        thread = threading.Thread(target=worker_thread, args=(f"Thread_{i}", duration))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Analyze results
    print("\nThread timing results:")
    print("Thread_ID    Expected_ms  Measured_ms  Error_ms  Error_%")
    print("-" * 55)
    
    for data in timing_data:
        error = abs(data['measured_duration_ms'] - data['expected_duration_ms'])
        error_percent = (error / data['expected_duration_ms']) * 100
        print(f"{data['thread_id']:10}  {data['expected_duration_ms']:10.1f}  {data['measured_duration_ms']:10.1f}  {error:7.1f}  {error_percent:6.1f}")
    
    return timing_data

def test_concurrent_timing():
    """Test timing measurements under concurrent load"""
    print("\nTesting concurrent timing measurements...")
    
    results = []
    lock = threading.Lock()
    
    def concurrent_worker(worker_id, iterations):
        """Worker that performs timing measurements concurrently"""
        worker_results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate some work
            time.sleep(0.001)  # 1ms work
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            worker_results.append({
                'worker_id': worker_id,
                'iteration': i,
                'duration_ms': duration,
                'timestamp': time.time()
            })
        
        with lock:
            results.extend(worker_results)
    
    # Create multiple workers
    workers = []
    for i in range(4):  # 4 concurrent workers
        worker = threading.Thread(target=concurrent_worker, args=(f"Worker_{i}", 50))
        workers.append(worker)
    
    # Start all workers
    for worker in workers:
        worker.start()
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    # Analyze concurrent results
    print(f"Total measurements: {len(results)}")
    
    # Group by worker
    worker_stats = {}
    for result in results:
        worker_id = result['worker_id']
        if worker_id not in worker_stats:
            worker_stats[worker_id] = []
        worker_stats[worker_id].append(result['duration_ms'])
    
    print("\nConcurrent timing statistics:")
    print("Worker_ID   Count   Avg_ms   Min_ms   Max_ms   Std_ms")
    print("-" * 55)
    
    for worker_id, durations in worker_stats.items():
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        print(f"{worker_id:10}  {len(durations):6}  {avg_duration:7.2f}  {min_duration:7.2f}  {max_duration:7.2f}  {std_duration:6.2f}")
    
    return results

def main():
    print("Timing Accuracy Test Suite")
    print("=" * 50)
    
    # Test 1: Basic timing precision
    precision_results = test_timing_precision()
    
    # Test 2: Thread timing
    thread_results = test_thread_timing()
    
    # Test 3: Concurrent timing
    concurrent_results = test_concurrent_timing()
    
    # Save results to Excel
    try:
        # Save precision test results
        df_precision = pd.DataFrame(precision_results)
        df_precision.to_excel('timing_precision_test.xlsx', index=False)
        print(f"\nPrecision test results saved to: timing_precision_test.xlsx")
        
        # Save thread test results
        df_thread = pd.DataFrame(thread_results)
        df_thread.to_excel('thread_timing_test.xlsx', index=False)
        print(f"Thread timing test results saved to: thread_timing_test.xlsx")
        
        # Save concurrent test results
        df_concurrent = pd.DataFrame(concurrent_results)
        df_concurrent.to_excel('concurrent_timing_test.xlsx', index=False)
        print(f"Concurrent timing test results saved to: concurrent_timing_test.xlsx")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nTiming accuracy tests completed!")
    print("Check the generated Excel files for detailed results.")

if __name__ == "__main__":
    main() 