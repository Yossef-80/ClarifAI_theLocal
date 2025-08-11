#!/usr/bin/env python3
"""
Test script to demonstrate thread timing analysis
This script simulates the timing analysis without requiring the full video processing pipeline
"""

import time
import threading
from collections import deque
import statistics
import pandas as pd

class TimingAnalyzer:
    def __init__(self):
        self.thread_timings = {
            'capture': deque(maxlen=100),
            'detection': deque(maxlen=100),
            'attention': deque(maxlen=100),
            'tracker': deque(maxlen=100),
            'gui_update': deque(maxlen=100)
        }
        self.timing_lock = threading.Lock()
    
    def record_thread_timing(self, thread_name, start_time, end_time):
        """Record timing for a specific thread"""
        with self.timing_lock:
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            self.thread_timings[thread_name].append(duration)
    
    def get_thread_stats(self, thread_name):
        """Get statistics for a specific thread"""
        with self.timing_lock:
            timings = list(self.thread_timings[thread_name])
            if not timings:
                return None
            return {
                'count': len(timings),
                'avg_ms': statistics.mean(timings),
                'min_ms': min(timings),
                'max_ms': max(timings),
                'median_ms': statistics.median(timings),
                'std_ms': statistics.stdev(timings) if len(timings) > 1 else 0
            }
    
    def print_thread_timing_report(self):
        """Print comprehensive timing report for all threads"""
        print("\n" + "="*60)
        print("THREAD TIMING ANALYSIS REPORT")
        print("="*60)
        
        total_frames = 0
        for thread_name in self.thread_timings.keys():
            stats = self.get_thread_stats(thread_name)
            if stats:
                total_frames = max(total_frames, stats['count'])
                print(f"\n{thread_name.upper()} THREAD:")
                print(f"  Frames processed: {stats['count']}")
                print(f"  Average time: {stats['avg_ms']:.2f} ms")
                print(f"  Min time: {stats['min_ms']:.2f} ms")
                print(f"  Max time: {stats['max_ms']:.2f} ms")
                print(f"  Median time: {stats['median_ms']:.2f} ms")
                print(f"  Std deviation: {stats['std_ms']:.2f} ms")
                print(f"  FPS potential: {1000/stats['avg_ms']:.1f} FPS")
        
        print(f"\n{'='*60}")
        print(f"TOTAL FRAMES ANALYZED: {total_frames}")
        print("="*60)
    
    def save_timing_report_to_excel(self):
        """Save detailed timing report to Excel file"""
        try:
            timing_data = []
            for thread_name in self.thread_timings.keys():
                timings = list(self.thread_timings[thread_name])
                for i, timing in enumerate(timings):
                    timing_data.append({
                        'Thread': thread_name,
                        'Frame': i + 1,
                        'Time_ms': timing,
                        'Timestamp': time.time()
                    })
            
            if timing_data:
                df = pd.DataFrame(timing_data)
                filename = f'thread_timing_analysis_{int(time.time())}.xlsx'
                df.to_excel(filename, index=False)
                print(f"Detailed timing report saved to: {filename}")
                
                # Also save summary statistics
                summary_data = []
                for thread_name in self.thread_timings.keys():
                    stats = self.get_thread_stats(thread_name)
                    if stats:
                        summary_data.append({
                            'Thread': thread_name,
                            'Frames_Processed': stats['count'],
                            'Avg_Time_ms': stats['avg_ms'],
                            'Min_Time_ms': stats['min_ms'],
                            'Max_Time_ms': stats['max_ms'],
                            'Median_Time_ms': stats['median_ms'],
                            'Std_Dev_ms': stats['std_ms'],
                            'Potential_FPS': 1000/stats['avg_ms']
                        })
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    summary_filename = f'thread_timing_summary_{int(time.time())}.xlsx'
                    df_summary.to_excel(summary_filename, index=False)
                    print(f"Summary statistics saved to: {summary_filename}")
                    
        except Exception as e:
            print(f"Error saving timing report: {e}")

def simulate_thread_work(thread_name, analyzer, duration_range=(0.01, 0.05)):
    """Simulate thread work with variable timing"""
    import random
    
    for i in range(20):  # Simulate 20 frames
        start_time = time.time()
        
        # Simulate work with variable duration
        work_time = random.uniform(*duration_range)
        time.sleep(work_time)
        
        end_time = time.time()
        analyzer.record_thread_timing(thread_name, start_time, end_time)
        
        print(f"[{thread_name.upper()}] Frame {i+1} completed in {work_time*1000:.1f}ms")
        time.sleep(0.1)  # Small delay between frames

def main():
    print("Thread Timing Analysis Demo")
    print("=" * 40)
    
    analyzer = TimingAnalyzer()
    
    # Simulate different thread workloads
    threads = []
    
    # Capture thread (fastest)
    t1 = threading.Thread(target=simulate_thread_work, args=('capture', analyzer, (0.005, 0.015)))
    threads.append(t1)
    
    # Detection thread (medium)
    t2 = threading.Thread(target=simulate_thread_work, args=('detection', analyzer, (0.02, 0.04)))
    threads.append(t2)
    
    # Attention thread (slowest)
    t3 = threading.Thread(target=simulate_thread_work, args=('attention', analyzer, (0.03, 0.06)))
    threads.append(t3)
    
    # Tracker thread (medium)
    t4 = threading.Thread(target=simulate_thread_work, args=('tracker', analyzer, (0.015, 0.035)))
    threads.append(t4)
    
    # GUI update thread (fast)
    t5 = threading.Thread(target=simulate_thread_work, args=('gui_update', analyzer, (0.008, 0.020)))
    threads.append(t5)
    
    # Start all threads
    print("Starting threads...")
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print("\nAll threads completed!")
    
    # Generate and display timing report
    analyzer.print_thread_timing_report()
    
    # Save to Excel
    analyzer.save_timing_report_to_excel()
    
    print("\nDemo completed! Check the generated Excel files for detailed analysis.")

if __name__ == "__main__":
    main() 