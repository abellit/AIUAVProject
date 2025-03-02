import sys
import os
import tensorflow as tf
import time  # Add missing import

# Docker container path
model_path = '/app/model/drone_mask_rcnn_coco_datasplit40.h5'

try:
    # For Mask R-CNN models, you might need to use a custom loading method
    # Option 1: Try using custom_objects if you have custom layers
    pretrained_cnn = tf.keras.models.load_model(
        model_path,
        custom_objects=None  # Add any custom layers here if needed
    )
    
    # Option 2: If it's a TensorFlow SavedModel format
    # pretrained_cnn = tf.saved_model.load(model_path)
    
    print(f"Model successfully loaded from: {model_path}")
    time.sleep(5)
    
    # If it's a Keras model
    if hasattr(pretrained_cnn, 'summary'):
        pretrained_cnn.summary()
    else:
        print("Model loaded but doesn't support summary() method")
        
except Exception as e:
    print(f"Error loading model from: {model_path}")
    print(f"Error details: {e}")
    pretrained_cnn = None  # Handle loading error