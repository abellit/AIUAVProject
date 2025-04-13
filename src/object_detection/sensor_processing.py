import numpy as np
import airsim
import cv2
import time
from lidar_preprocessor import LidarProcessor

class SensorProcessor:
    def __init__(self, client):
        self.client = client
        self.lidar_processor = LidarProcessor(voxel_size=0.5, fps_samples=1024)
        
        # State variables
        self.current_position = np.zeros(3)
        self.current_orientation = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.imu_data = None
    
    def update_state(self):
        """Update drone state from various sensors"""
        state = self.client.getMultirotorState()
        
        # Position
        self.current_position = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        
        # Orientation (quaternion to Euler)
        q = state.kinematics_estimated.orientation
        self.current_orientation = self.quaternion_to_euler(
            q.w_val, q.x_val, q.y_val, q.z_val
        )
        
        # Velocity
        self.current_velocity = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        
        # IMU data
        self.imu_data = self.client.getImuData()
    
    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_rgb_image(self):
        """Retrieve RGB image from AirSim"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            
            if not responses or len(responses) == 0:
                return None
                
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[0].height, responses[0].width, 3)
            
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                return None
                
            return img
        except Exception as e:
            print(f"Error getting RGB image: {e}")
            return None
    
    def get_depth_image(self):
        """Retrieve depth image from AirSim"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if not responses or len(responses) == 0:
                return None
                
            img = np.array(responses[0].image_data_float, dtype=np.float32)
            img = img.reshape(responses[0].height, responses[0].width)
            
            if np.isnan(img).any() or np.isinf(img).any():
                img[np.isnan(img)] = 100.0
                img[np.isinf(img)] = 100.0
                
            return img
        except Exception as e:
            print(f"Error getting depth image: {e}")
            return None
    
    def get_lidar_data(self):
        """Get LiDAR data for 3D obstacle detection"""
        lidar_data = self.client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            return np.array([])
            
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return points
    
    def advanced_lidar_processing(self):
        """Process LiDAR data using advanced techniques"""
        raw_points = self.get_lidar_data()
        if len(raw_points) < 3:
            return np.array([])
            
        # Process with voxelization and FPS
        processed_points = self.lidar_processor.process_lidar_data(raw_points)
        return processed_points
    
    def move_drone(self, direction, duration=1.0, speed=3.0):
        """Move the drone based on direction"""
        try:
            if direction == "left":
                self.client.moveByVelocityAsync(0, -speed, 0, duration)
            elif direction == "right":
                self.client.moveByVelocityAsync(0, speed, 0, duration)
            elif direction == "up":
                self.client.moveByVelocityAsync(0, 0, -speed, duration)
            elif direction == "down":
                self.client.moveByVelocityAsync(0, 0, speed, duration)
            elif direction == "forward":
                yaw = self.current_orientation[2]
                vx = speed * np.cos(yaw)
                vy = speed * np.sin(yaw)
                self.client.moveByVelocityAsync(vx, vy, 0, duration)
            elif direction == "backward":
                yaw = self.current_orientation[2]
                vx = -speed * np.cos(yaw)
                vy = -speed * np.sin(yaw)
                self.client.moveByVelocityAsync(vx, vy, 0, duration)
            else:
                self.client.hoverAsync()
                
            time.sleep(duration)
            self.update_state()
        except Exception as e:
            print(f"Error during movement: {e}")
            self.client.hoverAsync()