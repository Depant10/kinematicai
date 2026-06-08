"""
Quick test to verify everything is installed correctly.
Run this first: python 00_test_setup.py
"""

import sys

print("="*50)
print("SETUP VERIFICATION")
print("="*50)

errors = []

# 1. Python
print(f"\nPython: {sys.version}")

# 2. PyTorch + CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA GPU: {gpu_name}")
        x = torch.randn(100, 100, device='cuda')
        y = x @ x.T
        print(f"GPU compute test: PASSED")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple MPS: AVAILABLE")
        x = torch.randn(100, 100, device='mps')
        y = x @ x.T
        print(f"MPS compute test: PASSED")
    else:
        print("GPU: NOT AVAILABLE (will use CPU)")
        errors.append("No GPU detected - will fall back to CPU (much slower)")
except ImportError:
    errors.append("PyTorch not installed")
    print("PyTorch: NOT INSTALLED")

# 3. OpenCV
try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError:
    errors.append("OpenCV not installed: pip install opencv-python")

# 4. NumPy, SciPy, Matplotlib
for lib in ['numpy', 'scipy', 'matplotlib']:
    try:
        mod = __import__(lib)
        print(f"{lib}: {mod.__version__}")
    except ImportError:
        errors.append(f"{lib} not installed: pip install {lib}")

# 5. YOLO (Ultralytics)
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")

    # Check if pose model exists locally
    import os
    for model_name in ['yolo11m-pose.pt', 'yolo11n-pose.pt']:
        if os.path.exists(model_name):
            print(f"  Model {model_name}: FOUND")
        else:
            print(f"  Model {model_name}: not downloaded yet (will auto-download on first run)")
except ImportError:
    errors.append("Ultralytics not installed: pip install ultralytics")
    print("Ultralytics: NOT INSTALLED")

# 6. yt-dlp
import shutil
if shutil.which('yt-dlp'):
    print("yt-dlp: FOUND")
else:
    errors.append("yt-dlp not in PATH: pip install yt-dlp")

# Summary
print(f"\n{'='*50}")
if errors:
    print(f"ISSUES FOUND ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
    print("\nFix these before running the pipeline.")
else:
    print("ALL CHECKS PASSED - ready to process matches!")
print("="*50)
