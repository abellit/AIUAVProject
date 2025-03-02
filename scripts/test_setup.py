import tensorflow as tf
import numpy as np
import os

print("=== Environment Test Results ===")
print(f"Working directory: {os.getcwd()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available files: {os.listdir('.')}")