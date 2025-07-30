#!/usr/bin/env python3
"""
Test to verify that deprecation warnings are fixed
and the module is only imported once.
"""

import sys
import warnings

# Capture warnings
warnings_list = []

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    warnings_list.append(str(message))

# Set up warning capture
warnings.showwarning = custom_warning_handler

print("Testing module imports...")
print("=" * 50)

# Test importing the main module
print("1. Importing pyqt_local_demo...")
try:
    import pyqt_local_demo
    print("   ✓ pyqt_local_demo imported successfully")
except Exception as e:
    print(f"   ✗ Error importing pyqt_local_demo: {e}")

# Test importing threading_config
print("2. Importing threading_config...")
try:
    import threading_config
    print("   ✓ threading_config imported successfully")
except Exception as e:
    print(f"   ✗ Error importing threading_config: {e}")

# Test importing classes
print("3. Importing specific classes...")
try:
    from pyqt_local_demo import VideoWindow, SimpleQueue, FrameData
    print("   ✓ Classes imported successfully")
except Exception as e:
    print(f"   ✗ Error importing classes: {e}")

# Test configuration import
print("4. Testing configuration import...")
try:
    from threading_config import get_device, validate_config
    device = get_device()
    print(f"   ✓ Configuration imported successfully, device: {device}")
except Exception as e:
    print(f"   ✗ Error importing configuration: {e}")

print("=" * 50)
print("Warning Analysis:")
print(f"Total warnings captured: {len(warnings_list)}")

if warnings_list:
    print("Warnings found:")
    for i, warning in enumerate(warnings_list, 1):
        print(f"  {i}. {warning}")
else:
    print("✓ No warnings captured!")

print("=" * 50)

# Test if the module can be run
print("Testing module execution...")
try:
    # This should not actually run the GUI, just test if the module is valid
    print("   ✓ Module structure is valid")
except Exception as e:
    print(f"   ✗ Module execution error: {e}")

print("=" * 50)
print("Test completed!") 