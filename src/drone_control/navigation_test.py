import airsim
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO

# Load trained YOLO model
pre_trained_model = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\model\TrainedWeights\best_yolov10n_obstacle.pt'
model = YOLO(pre_trained_model)
model.fuse()

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Function to get RGB image
def get_image():
    try:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        return img_rgb if img_rgb.size > 0 else None
    except:
        return None

# Function to get LiDAR data
def get_lidar_data():
    lidar_data = client.getLidarData()
    if len(lidar_data.point_cloud) < 3:
        return None
    return np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

# Perform object detection
def detect_obstacles(image):
    results = model(image)
    return results[0]

# Process detections and determine if an obstacle is within danger range
def is_obstacle_close(detections, lidar_points, threshold=3.0):
    for detection in detections.boxes.data.tolist():
        _, _, _, _, _, _ = detection
        depth = np.mean(lidar_points[:, 0]) if lidar_points is not None else None
        if depth and depth < threshold:
            return True, depth  # Return True and the depth
    return False, None

# Choose the best avoidance direction based on LiDAR data
def choose_avoidance_direction(lidar_points):
    left_space = np.mean(lidar_points[lidar_points[:, 1] < 0][:, 0])  # Check left side
    right_space = np.mean(lidar_points[lidar_points[:, 1] > 0][:, 0])  # Check right side
    up_space = np.mean(lidar_points[lidar_points[:, 2] > 0][:, 0])  # Check above
    
    if right_space > left_space and right_space > up_space:
        return "right"
    elif left_space > right_space and left_space > up_space:
        return "left"
    elif up_space > left_space and up_space > right_space:
        return "up"
    else:
        return "slow_down"  # No clear path, slow down

# Adaptive movement logic
def move_drone(avoid_direction, speed=5):
    if avoid_direction == "left":
        client.moveByVelocityAsync(0, -speed, 0, 1)  # Move left
    elif avoid_direction == "right":
        client.moveByVelocityAsync(0, speed, 0, 1)  # Move right
    elif avoid_direction == "up":
        client.moveByVelocityAsync(0, 0, speed, 1)  # Move up
    elif avoid_direction == "slow_down":
        client.moveByVelocityAsync(speed / 2, 0, 0, 1)  # Slow down
    else:
        client.moveByVelocityAsync(speed, 0, 0, 1)  # Move forward
    time.sleep(1)

# Main loop
try:
    while True:
        image = get_image()
        lidar_points = get_lidar_data()
        if image is None or lidar_points is None:
            continue
        
        detections = detect_obstacles(image)
        obstacle_close, obstacle_depth = is_obstacle_close(detections, lidar_points, threshold=3.0)
        
        if obstacle_close:
            print(f"Obstacle detected at {obstacle_depth:.2f}m! Avoiding...")
            best_direction = choose_avoidance_direction(lidar_points)
            move_drone(best_direction, speed=3)  # Move based on best direction
        else:
            move_drone("forward", speed=10)  # Move forward normally
        
        time.sleep(0.1)

except KeyboardInterrupt:
    client.enableApiControl(False)
    client.armDisarm(False)
    cv2.destroyAllWindows()
