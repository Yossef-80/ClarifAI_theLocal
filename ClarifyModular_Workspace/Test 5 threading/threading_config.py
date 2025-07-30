"""
Threading Configuration for Enhanced Video Processing System
This file contains all configurable parameters for the threading system.
"""

# Thread Queue Sizes
QUEUE_SIZES = {
    'capture': 3,      # Number of frames to buffer in capture thread
    'detection': 2,    # Number of frames to buffer in detection thread
    'recognition': 2,  # Number of frames to buffer in recognition thread
    'attention': 2,    # Number of frames to buffer in attention thread
    'display': 1,      # Number of frames to buffer in display thread
}

# Thread Timing Parameters
TIMING = {
    'capture_interval': 40,    # Milliseconds between frame captures (~25 FPS)
    'display_fps': 25,         # Target display frame rate
    'queue_timeout': 0.1,      # Timeout for queue operations (seconds)
    'performance_log_interval': 1000,  # Performance logging interval (ms)
}

# Performance Monitoring
PERFORMANCE = {
    'max_timing_history': 100,  # Maximum number of timing samples to keep per operation
    'log_performance_stats': True,  # Whether to log performance stats to console
    'save_performance_logs': True,  # Whether to save performance logs to file
}

# Error Handling
ERROR_HANDLING = {
    'max_retries': 3,          # Maximum number of retries for failed operations
    'retry_delay': 0.1,        # Delay between retries (seconds)
    'continue_on_error': True, # Whether to continue processing on errors
}

# Memory Management
MEMORY = {
    'max_pending_frames': 10,  # Maximum number of pending frames to keep in memory
    'cleanup_interval': 100,   # Frames between memory cleanup operations
    'drop_old_frames': True,   # Whether to drop old frames when queues are full
}

# Model Configuration
MODELS = {
    'device': 'auto',          # 'auto', 'cuda', 'cpu'
    'detection_confidence': 0.5,  # Minimum confidence for face detection
    'recognition_threshold': 0.65, # Threshold for face recognition
    'attention_iou_threshold': 0.08, # IoU threshold for attention detection
}

# Display Configuration
DISPLAY = {
    'show_fps': True,          # Whether to show FPS counter
    'show_performance_stats': True,  # Whether to show performance stats
    'update_metrics_interval': 30,   # Frames between metrics updates
}

# Logging Configuration
LOGGING = {
    'log_level': 'INFO',       # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_to_file': True,       # Whether to log to file
    'log_file': 'threading_system.log',  # Log file name
    'max_log_size': 10 * 1024 * 1024,  # Maximum log file size (10MB)
}

# Thread Priority (Windows only)
THREAD_PRIORITIES = {
    'capture': 'normal',       # 'low', 'normal', 'high'
    'detection': 'high',       # Detection is CPU intensive
    'recognition': 'normal',   # Recognition is moderately intensive
    'attention': 'normal',     # Attention detection
    'display': 'low',          # Display can be lower priority
}

# Advanced Threading
ADVANCED = {
    'use_thread_pool': False,  # Whether to use QThreadPool for some operations
    'max_thread_pool_size': 4, # Maximum thread pool size
    'enable_work_stealing': True,  # Enable work stealing between threads
    'adaptive_queue_sizes': True,  # Dynamically adjust queue sizes based on performance
}

def get_device():
    """Get the device to use for models"""
    if MODELS['device'] == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    return MODELS['device']

def validate_config():
    """Validate the configuration parameters"""
    errors = []
    
    # Validate queue sizes
    for name, size in QUEUE_SIZES.items():
        if size < 1:
            errors.append(f"Queue size for {name} must be at least 1")
    
    # Validate timing parameters
    if TIMING['capture_interval'] < 1:
        errors.append("Capture interval must be at least 1ms")
    if TIMING['display_fps'] < 1:
        errors.append("Display FPS must be at least 1")
    
    # Validate performance parameters
    if PERFORMANCE['max_timing_history'] < 1:
        errors.append("Max timing history must be at least 1")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Validate configuration on import (only if not being imported as a module)
if __name__ == "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Using default configuration values.") 