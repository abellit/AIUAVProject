import os
import csv
import time
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

class SensorSynchronizer:
    def __init__(self, data_root="./data"):
        self.data_root = data_root
        self.frame_count = 0
        self.timestamp = 0
        
        # Create directory structure
        os.makedirs(f"{data_root}/rgb", exist_ok=True)
        os.makedirs(f"{data_root}/lidar", exist_ok=True)
        os.makedirs(f"{data_root}/imu", exist_ok=True)
        os.makedirs(f"{data_root}/sync", exist_ok=True)
        
    def save_rgb_image(self, image):
        """Save RGB image with timestamp"""
        if image is None:
            return None
            
        self.timestamp = time.time()
        filename = f"{self.data_root}/rgb/rgb_{self.frame_count:04d}.png"
        cv2.imwrite(filename, image)
        return self.timestamp
        
    def save_lidar_data(self, points, timestamp=None):
        """Save LiDAR point cloud with timestamp"""
        if len(points) < 3:
            return None
            
        if timestamp is None:
            timestamp = time.time()
            
        filename = f"{self.data_root}/lidar/lidar_{self.frame_count:04d}.npy"
        np.save(filename, points)
        return timestamp
        
    def save_imu_data(self, imu_data, timestamp=None):
        """Save IMU data as CSV with timestamp"""
        if imu_data is None:
            return None
            
        if timestamp is None:
            timestamp = time.time()
            
        filename = f"{self.data_root}/imu/imu_{self.frame_count:04d}.csv"
        
        # Convert IMU data to dictionary
        imu_dict = {
            'timestamp': timestamp,
            'acc_x': imu_data.linear_acceleration.x_val,
            'acc_y': imu_data.linear_acceleration.y_val,
            'acc_z': imu_data.linear_acceleration.z_val,
            'gyro_x': imu_data.angular_velocity.x_val,
            'gyro_y': imu_data.angular_velocity.y_val,
            'gyro_z': imu_data.angular_velocity.z_val,
            'orientation_w': imu_data.orientation.w_val,
            'orientation_x': imu_data.orientation.x_val,
            'orientation_y': imu_data.orientation.y_val,
            'orientation_z': imu_data.orientation.z_val
        }
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=imu_dict.keys())
            writer.writeheader()
            writer.writerow(imu_dict)
            
        return timestamp
        
    def create_synchronized_frame(self, rgb_image, lidar_points, imu_data):
        """Create a synchronized frame with all sensor data"""
        # Create frame directory
        frame_dir = f"{self.data_root}/sync/frame_{self.frame_count:04d}"
        os.makedirs(frame_dir, exist_ok=True)
        
        # Save RGB image
        timestamp = time.time()
        cv2.imwrite(f"{frame_dir}/rgb.png", rgb_image)
        
        # Save LiDAR data
        np.save(f"{frame_dir}/lidar.npy", lidar_points)
        
        # Save IMU data
        imu_dict = {
            'timestamp': timestamp,
            'acc_x': imu_data.linear_acceleration.x_val,
            'acc_y': imu_data.linear_acceleration.y_val,
            'acc_z': imu_data.linear_acceleration.z_val,
            'gyro_x': imu_data.angular_velocity.x_val,
            'gyro_y': imu_data.angular_velocity.y_val,
            'gyro_z': imu_data.angular_velocity.z_val,
            'orientation_w': imu_data.orientation.w_val,
            'orientation_x': imu_data.orientation.x_val,
            'orientation_y': imu_data.orientation.y_val,
            'orientation_z': imu_data.orientation.z_val
        }
        
        # Write to CSV
        with open(f"{frame_dir}/imu.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=imu_dict.keys())
            writer.writeheader()
            writer.writerow(imu_dict)
            
        # Create metadata file
        metadata = {
            'timestamp': timestamp,
            'frame_id': self.frame_count,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        }
        
        with open(f"{frame_dir}/metadata.json", 'w') as f:
            import json
            json.dump(metadata, f)
            
        self.frame_count += 1
        return timestamp
        
    def load_synchronized_frame(self, frame_id):
        """Load a synchronized frame by ID"""
        frame_dir = f"{self.data_root}/sync/frame_{frame_id:04d}"
        
        if not os.path.exists(frame_dir):
            return None
            
        # Load RGB
        rgb = cv2.imread(f"{frame_dir}/rgb.png")
        
        # Load LiDAR
        lidar = np.load(f"{frame_dir}/lidar.npy")
        
        # Load IMU
        imu = pd.read_csv(f"{frame_dir}/imu.csv")
        
        # Load metadata
        with open(f"{frame_dir}/metadata.json", 'r') as f:
            import json
            metadata = json.load(f)
            
        return {
            'rgb': rgb,
            'lidar': lidar,
            'imu': imu,
            'metadata': metadata
        }