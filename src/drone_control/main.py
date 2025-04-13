import airsim
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import cv2
import torch
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
from object_detection.lidar_preprocessor import LidarProcessor
from object_detection.sensor_fusion import SensorSynchronizer
from rrt_star import RRTStar

# Modified DroneNavigationSystem class with advanced features
class EnhancedDroneNavigationSystem:
    def __init__(self):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off
        self.client.takeoffAsync().join()
        
        # Load YOLO model
        self.yolo_model = YOLO(r'model/TrainedWeights/best_yolov10n_obstacle.pt')
        self.yolo_model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference
        print("YOLO model loaded successfully")
        
        # Initialize LiDAR processor
        self.lidar_processor = LidarProcessor(voxel_size=0.5, fps_samples=1024)
        
        # Initialize sensor synchronizer
        self.sensor_sync = SensorSynchronizer(data_root="./data")
        
        # Navigation parameters
        self.obstacle_threshold = 3.0  # meters
        self.safe_height = -5.0  # meters (negative is up in AirSim)
        self.cruise_speed = 3.0  # m/s
        self.current_position = np.zeros(3)
        self.current_orientation = np.zeros(3)  # roll, pitch, yaw in radians
        self.current_velocity = np.zeros(3)
        self.goal_reached_threshold = 2.0  # meters
        
        # Data collection parameters
        self.collect_data = True  # Set to True to collect synchronized data
        self.collection_frequency = 5  # Collect data every N frames
        self.frame_counter = 0

        # Collision avoidance parameters
        self.grid_size = 10  # Number of grid cells for obstacle detection
        self.obstacle_memory = {}  # Dictionary to store obstacle locations with timestamp
        self.obstacle_memory_timeout = 5.0  # Seconds to remember an obstacle
        
        # YOLO detection parameters
        self.yolo_confidence_threshold = 0.25  # Minimum confidence for a valid detection
        self.yolo_detections = []  # Store recent detections
        self.detection_weight = 1.5  # How much to weight YOLO detections vs. other sensors
        
        # IMU data
        self.imu_data = None
        self.last_position = np.zeros(3)
        self.position_history = []  # To store position history for path smoothing
        
        # Path planning
        self.path = []
        self.replanning_distance = 5.0  # Distance to trigger replanning
        
        # Visualization
        self.show_detections = True  # Set to False to disable visualization
        
        # Initialize state
        self.update_state()
        
    def update_state(self):
        """Update drone state from various sensors"""
        # Original state update code...
        state = self.client.getMultirotorState()
        self.current_position = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        
        # Get orientation (quaternion to Euler)
        q = state.kinematics_estimated.orientation
        self.current_orientation = self.quaternion_to_euler(
            q.w_val, q.x_val, q.y_val, q.z_val
        )
        
        # Get velocity
        self.current_velocity = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        
        # Get IMU data
        self.imu_data = self.client.getImuData()
        
        # Collect synchronized data if enabled
        if self.collect_data and self.frame_counter % self.collection_frequency == 0:
            rgb_image = self.get_rgb_image()
            lidar_points = self.get_lidar_data()
            
            if rgb_image is not None and len(lidar_points) > 0:
                self.sensor_sync.create_synchronized_frame(rgb_image, lidar_points, self.imu_data)
                
        self.frame_counter += 1
    
    def advanced_lidar_processing(self):
        """Process LiDAR data using advanced techniques"""
        raw_points = self.get_lidar_data()
        if len(raw_points) < 3:
            return np.array([]), None
            
        # Process with voxelization and FPS
        processed_points, voxel_grid = self.lidar_processor.process_lidar_data(raw_points)
        
        # Create voxel grid feature (for ML models)
        voxel_features = self.lidar_processor.create_voxel_grid_feature(processed_points)
        
        return processed_points, voxel_features
    
    def detect_obstacles_enhanced(self):
        """Enhanced obstacle detection using voxelized LiDAR and YOLO"""
        obstacles = set()
        current_time = time.time()
        
        # Process LiDAR with advanced techniques
        processed_points, voxel_features = self.advanced_lidar_processing()
        
        # Add obstacles from processed LiDAR points
        if len(processed_points) > 0:
            # Transform LiDAR points to world coordinates
            yaw = self.current_orientation[2]
            for point in processed_points:
                if np.linalg.norm(point) < self.obstacle_threshold:
                    world_point = (
                        self.current_position[0] + point[0] * np.cos(yaw) - point[1] * np.sin(yaw),
                        self.current_position[1] + point[0] * np.sin(yaw) + point[1] * np.cos(yaw)
                    )
                    obstacle_key = (round(world_point[0]), round(world_point[1]))
                    self.obstacle_memory[obstacle_key] = current_time
                    obstacles.add(obstacle_key)
        
        # Run YOLO detection
        yolo_detections = self.run_yolo_detection()
        for detection in yolo_detections:
            if detection["depth"] < self.obstacle_threshold * 1.5:
                obstacle_pos = self.convert_detection_to_obstacle(detection)
                obstacle_key = (round(obstacle_pos[0]), round(obstacle_pos[1]))
                self.obstacle_memory[obstacle_key] = current_time
                obstacles.add(obstacle_key)
        
        # Clean up old obstacles
        expired_obstacles = []
        for obs_key, timestamp in self.obstacle_memory.items():
            if current_time - timestamp > self.obstacle_memory_timeout:
                expired_obstacles.append(obs_key)
        
        for obs_key in expired_obstacles:
            del self.obstacle_memory[obs_key]
        
        return obstacles
    
    def save_training_sample(self):
        """Save a synchronized sample for model training"""
        rgb_image = self.get_rgb_image()
        lidar_points = self.get_lidar_data()
        
        if rgb_image is not None and len(lidar_points) > 0:
            timestamp = self.sensor_sync.save_rgb_image(rgb_image)
            self.sensor_sync.save_lidar_data(lidar_points, timestamp)
            self.sensor_sync.save_imu_data(self.imu_data, timestamp)
            
            print(f"Saved training sample {self.sensor_sync.frame_count}")
            self.sensor_sync.frame_count += 1
    
    # Other methods remain the same as in your original code...

    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_rgb_image(self):
        """Retrieve RGB image from AirSim for YOLO processing"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            
            if not responses or len(responses) == 0:
                print("Warning: No RGB image received")
                return None
                
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[0].height, responses[0].width, 3)
            
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print("Warning: Invalid RGB image dimensions")
                return None
                
            return img
        except Exception as e:
            print(f"Error getting RGB image: {e}")
            return None
    
    def get_depth_image(self):
        """Retrieve depth image from AirSim with error handling"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if not responses or len(responses) == 0:
                print("Warning: No depth image received")
                return None
                
            img = np.array(responses[0].image_data_float, dtype=np.float32)
            img = img.reshape(responses[0].height, responses[0].width)
            
            # Check for invalid values
            if np.isnan(img).any() or np.isinf(img).any():
                print("Warning: Depth image contains invalid values")
                img[np.isnan(img)] = 100.0  # Replace NaN with a large distance
                img[np.isinf(img)] = 100.0  # Replace Inf with a large distance
                
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
    
    def run_yolo_detection(self):
        """Perform YOLO object detection on the current camera view"""
        rgb_image = self.get_rgb_image()
        if rgb_image is None:
            return []
        
        # Get LiDAR data for depth estimation
        lidar_points = self.get_lidar_data()
        
        # Run YOLO inference
        try:
            results = self.yolo_model(rgb_image)
            detections = results[0]
            
            # Process detections
            detected_objects = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = detection
                
                if conf < self.yolo_confidence_threshold:
                    continue
                    
                label = detections.names[int(cls)]
                
                # Calculate center point of bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Estimate distance using depth image or LiDAR
                depth_img = self.get_depth_image()
                if depth_img is not None:
                    h, w = depth_img.shape
                    # Convert bbox center to depth image coordinates
                    cx = min(int(center_x * w / rgb_image.shape[1]), w-1)
                    cy = min(int(center_y * h / rgb_image.shape[0]), h-1)
                    
                    # Get depth at center point (with small averaging window)
                    window_size = 5
                    x_start = max(0, cx - window_size//2)
                    x_end = min(w, cx + window_size//2)
                    y_start = max(0, cy - window_size//2)
                    y_end = min(h, cy + window_size//2)
                    
                    depth_window = depth_img[y_start:y_end, x_start:x_end]
                    depth = np.mean(depth_window)
                elif len(lidar_points) > 0:
                    # Use average LiDAR point depth as fallback
                    depth = np.mean(lidar_points[:, 0])
                else:
                    # If no depth information, use a conservative estimate
                    depth = 5.0  # Default to 5 meters if no depth data
                
                detected_objects.append({
                    "label": label,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "depth": depth,
                    "center": (center_x, center_y)
                })
            
            # Visualize detections if enabled
            if self.show_detections and rgb_image is not None:
                vis_img = rgb_image.copy()
                for obj in detected_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"{obj['label']} {obj['depth']:.1f}m", 
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow("YOLO Detections", vis_img)
                cv2.waitKey(1)
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def convert_detection_to_obstacle(self, detection):
        """Convert a YOLO detection to world coordinates"""
        # Get the depth from the detection
        depth = detection["depth"]
        
        # Calculate field of view angles
        h, w = 480, 640  # Assuming standard resolution, adjust as needed
        fov_h = np.pi/3  # Horizontal FOV (60 degrees)
        fov_v = np.pi/4  # Vertical FOV (45 degrees)
        
        # Get normalized image coordinates (-1 to 1)
        center_x, center_y = detection["center"]
        norm_x = (center_x / w) * 2 - 1
        norm_y = (center_y / h) * 2 - 1
        
        # Calculate 3D direction in camera frame
        angle_h = norm_x * fov_h/2
        angle_v = norm_y * fov_v/2
        
        # Calculate 3D position of obstacle in drone's frame
        x = depth * np.cos(angle_v) * np.cos(angle_h)
        y = depth * np.cos(angle_v) * np.sin(angle_h)
        z = depth * np.sin(angle_v)
        
        # Transform to world coordinates using drone's position and orientation
        yaw = self.current_orientation[2]
        obstacle_pos = (
            self.current_position[0] + x * np.cos(yaw) - y * np.sin(yaw),
            self.current_position[1] + x * np.sin(yaw) + y * np.cos(yaw),
            self.current_position[2] + z
        )
        
        return obstacle_pos
    
    def detect_obstacles(self):
        """Advanced obstacle detection using YOLO, depth and LiDAR data"""
        obstacles = set()
        current_time = time.time()

        # Process LiDAR with advanced techniques
        processed_points, voxel_features = self.advanced_lidar_processing()
        
        # Add obstacles from processed LiDAR points with INCREASED buffer
        if len(processed_points) > 0:
            # Transform LiDAR points to world coordinates
            yaw = self.current_orientation[2]
            for point in processed_points:
                # Reduce threshold to detect obstacles further away
                if np.linalg.norm(point) < self.obstacle_threshold * 1.5:  # Increased from just threshold
                    world_point = (
                        self.current_position[0] + point[0] * np.cos(yaw) - point[1] * np.sin(yaw),
                        self.current_position[1] + point[0] * np.sin(yaw) + point[1] * np.cos(yaw)
                    )
                    obstacle_key = (round(world_point[0]), round(world_point[1]))
                    self.obstacle_memory[obstacle_key] = current_time
                    obstacles.add(obstacle_key)
                    
                    # Add additional buffer points for tree branches (thin objects)
                    for dx in range(-2, 3):  # Increased buffer from default
                        for dy in range(-2, 3):  # Increased buffer from default
                            if dx == 0 and dy == 0:
                                continue
                            buffer_key = (round(world_point[0] + dx), round(world_point[1] + dy))
                            self.obstacle_memory[buffer_key] = current_time
                            obstacles.add(buffer_key)
        
        # YOLO detection section needs more specific tree handling
        yolo_detections = self.run_yolo_detection()
        for detection in yolo_detections:
            # Reduce distance threshold for trees specifically
            tree_labels = ["tree", "branch", "plant", "foliage"]
            is_tree = any(label in detection["label"].lower() for label in tree_labels)
            
            threshold_multiplier = 2.0 if is_tree else 1.5  # Higher multiplier for trees
            
            if detection["depth"] < self.obstacle_threshold * threshold_multiplier:
                obstacle_pos = self.convert_detection_to_obstacle(detection)
                obstacle_key = (round(obstacle_pos[0]), round(obstacle_pos[1]))
                self.obstacle_memory[obstacle_key] = current_time
                obstacles.add(obstacle_key)
                
                # Add larger buffer for tree objects
                buffer_size = 3 if is_tree else 2
                for dx in range(-buffer_size, buffer_size + 1):
                    for dy in range(-buffer_size, buffer_size + 1):
                        buffer_key = (round(obstacle_pos[0] + dx), round(obstacle_pos[1] + dy))
                        self.obstacle_memory[buffer_key] = current_time
                        obstacles.add(buffer_key)
        
        # Process depth image for forward-facing obstacles
        depth_img = self.get_depth_image()
        if depth_img is not None:
            # Divide image into a grid for better obstacle localization
            h, w = depth_img.shape
            cell_h, cell_w = h // self.grid_size, w // self.grid_size
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    roi = depth_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    avg_depth = np.mean(roi)
                    
                    if avg_depth < self.obstacle_threshold:
                        # Convert image coordinates to world coordinates (approximate)
                        angle_h = (j - self.grid_size/2) / (self.grid_size/2) * np.pi/4  # Field of view assumption: 90 degrees
                        angle_v = (i - self.grid_size/2) / (self.grid_size/2) * np.pi/6  # Field of view assumption: 60 degrees
                        
                        # Calculate 3D position of obstacle in drone's frame
                        x = avg_depth * np.cos(angle_v) * np.cos(angle_h)
                        y = avg_depth * np.cos(angle_v) * np.sin(angle_h)
                        z = avg_depth * np.sin(angle_v)
                        
                        # Transform to world coordinates using drone's position and orientation
                        yaw = self.current_orientation[2]
                        obstacle_pos = (
                            self.current_position[0] + x * np.cos(yaw) - y * np.sin(yaw),
                            self.current_position[1] + x * np.sin(yaw) + y * np.cos(yaw),
                            self.current_position[2] + z
                        )
                        
                        # Add to obstacle memory with timestamp
                        obstacle_key = (round(obstacle_pos[0]), round(obstacle_pos[1]))
                        self.obstacle_memory[obstacle_key] = current_time
                        obstacles.add(obstacle_key)
        
        # Process LiDAR data for 360-degree obstacle detection
        lidar_points = self.get_lidar_data()
        if len(lidar_points) > 0:
            # Transform LiDAR points to world coordinates
            yaw = self.current_orientation[2]
            for point in lidar_points:
                if np.linalg.norm(point) < self.obstacle_threshold:
                    world_point = (
                        self.current_position[0] + point[0] * np.cos(yaw) - point[1] * np.sin(yaw),
                        self.current_position[1] + point[0] * np.sin(yaw) + point[1] * np.cos(yaw)
                    )
                    obstacle_key = (round(world_point[0]), round(world_point[1]))
                    self.obstacle_memory[obstacle_key] = current_time
                    obstacles.add(obstacle_key)

        branch_obstacles = self.detect_thin_branches()
        obstacles = obstacles.union(branch_obstacles)
        
        # Clean up old obstacles
        expired_obstacles = []
        for obs_key, timestamp in self.obstacle_memory.items():
            if current_time - timestamp > self.obstacle_memory_timeout:
                expired_obstacles.append(obs_key)
        
        for obs_key in expired_obstacles:
            del self.obstacle_memory[obs_key]
        
        return obstacles
    
    def detect_thin_branches(self):
        """Specialized detection for thin vertical obstacles"""
        points = self.get_lidar_data()
        branch_obstacles = set()
        
        if len(points) > 100:
            # Cluster points using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
            labels = clustering.labels_
            
            # Analyze clusters for vertical structures
            for label in set(labels):
                if label == -1: 
                    continue
                cluster = points[labels == label]
                z_range = np.max(cluster[:,2]) - np.min(cluster[:,2])
                xy_span = np.max(cluster[:,:2]) - np.min(cluster[:,:2])
                
                # Detect vertical structures (tall and thin)
                if z_range > 3.0 and xy_span < 1.0:
                    center = np.mean(cluster[:,:2], axis=0)
                    branch_obstacles.add((round(center[0]), round(center[1])))  # Fixed missing parenthesis
        
        return branch_obstacles
    
    def evaluate_movement_direction(self):
        """Determine best movement direction with improved tree detection"""
        # Get YOLO detections for obstacle avoidance
        yolo_detections = self.run_yolo_detection()
        
        # Check if there are any close obstacles detected by YOLO
        close_obstacles = False
        front_obstacle = False
        left_obstacle = False
        right_obstacle = False
        h, w = 480, 640  # Assuming standard resolution, adjust as needed
        
        for detection in yolo_detections:
            # Check specifically for tree-related objects
            tree_labels = ["tree", "branch", "plant", "foliage"]
            is_tree = any(label in detection["label"].lower() for label in tree_labels)
            
            # Use a larger threshold for trees
            threshold = self.obstacle_threshold * 1.8 if is_tree else self.obstacle_threshold
            
            if detection["depth"] < threshold:
                close_obstacles = True
                center_x = detection["center"][0]
                
                # Determine if obstacle is in front, left, or right
                if center_x < w/3:
                    left_obstacle = True
                elif center_x > 2*w/3:
                    right_obstacle = True
                else:
                    front_obstacle = True
        
        # Get depth image for additional information
        depth_img = self.get_depth_image()
        if depth_img is None and not close_obstacles:
            return "hover"  # Safety first
        
        # Check depth image if available
        if depth_img is not None:
            h, w = depth_img.shape
            left = np.mean(depth_img[:, :w//3])
            center = np.mean(depth_img[:, w//3:2*w//3])
            right = np.mean(depth_img[:, 2*w//3:])
            top = np.mean(depth_img[:h//3, :])
            bottom = np.mean(depth_img[2*h//3:, :])
            
            # Combine depth and YOLO information
            if left < self.obstacle_threshold:
                left_obstacle = True
            if right < self.obstacle_threshold:
                right_obstacle = True
            if center < self.obstacle_threshold:
                front_obstacle = True
        
        # Use IMU data to detect if we're tilting (which might indicate pushing against an obstacle)
        excessive_tilt = False
        if self.imu_data:
            # Check for excessive acceleration that might indicate collision
            acc_magnitude = np.sqrt(
                self.imu_data.linear_acceleration.x_val**2 +
                self.imu_data.linear_acceleration.y_val**2 +
                self.imu_data.linear_acceleration.z_val**2
            )
            if acc_magnitude > 15:  # m/s² threshold
                excessive_tilt = True
                print(f"Warning: High acceleration detected: {acc_magnitude} m/s²")
        
        # Decision logic for movement
        if front_obstacle or excessive_tilt:
            # Obstacle ahead or collision imminent
            if not left_obstacle and (depth_img is None or left > self.obstacle_threshold):
                return "left"
            elif not right_obstacle and (depth_img is None or right > self.obstacle_threshold):
                return "right"
            elif depth_img is not None and top > self.obstacle_threshold:
                return "up"
            elif depth_img is not None and bottom > self.obstacle_threshold:
                return "down"
            else:
                return "backward"  # Retreat if no clear path
        
        return "forward"
    
    def plan_path(self, start, goal, obstacles):
        """Improved RRT* planning with dynamic parameters for forests"""
        start_2d = (start[0], start[1])
        goal_2d = (goal[0], goal[1])
        
        # Adaptive parameters based on obstacle density
        density = len(obstacles)/(self.grid_size**2)
        step_size = max(2.0, 5.0 - density*3)  # Smaller steps in dense areas
        search_radius = max(5.0, 15.0 - density*10)  # Wider search in open areas
        
        buffered_obstacles = set()
        for obs in obstacles:
            # Reduced buffer for trees
            if any(label in obs for label in ["tree", "branch"]):
                buffer_range = 1
            else:
                buffer_range = 2
                
            for dx in range(-buffer_range, buffer_range+1):
                for dy in range(-buffer_range, buffer_range+1):
                    buffered_obstacles.add((obs[0]+dx, obs[1]+dy))

        try:
            rrt_star = RRTStar(
                start_2d, goal_2d, buffered_obstacles,
                step_size=step_size,
                max_iter=2000,  # Increased iterations
                search_radius=search_radius,
                goal_sample_rate=0.3  # More biased towards goal
            )
            path = rrt_star.plan()
            
            # Add vertical waypoints for 3D navigation
            if path:
                return self.add_vertical_components(path, start[2])
            return []
            
        except Exception as e:
            print(f"Path planning error: {e}")
            return []

    def add_vertical_components(self, path, current_z):
        """Add vertical exploration components to 2D path"""
        vertical_path = []
        for i, point in enumerate(path):
            # Alternate heights between waypoints
            z = current_z if i%2 == 0 else current_z - 2.0
            vertical_path.append((point[0], point[1], z))
        return vertical_path
    
    def smooth_path(self, path):
        """Apply path smoothing to reduce jerkiness"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            # Simple moving average
            smoothed.append((
                (path[i-1][0] + path[i][0] + path[i+1][0]) / 3,
                (path[i-1][1] + path[i][1] + path[i+1][1]) / 3,
                path[i][2]  # Keep original height
            ))
        smoothed.append(path[-1])
        return smoothed
    
    def move_drone(self, direction, duration=1.0, speed=3.0):
        """Move the drone based on detected obstacle direction"""
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
                
            # Wait for the movement to complete
            time.sleep(duration)
            self.update_state()
        except Exception as e:
            print(f"Error during movement: {e}")
            self.client.hoverAsync()  # Safety hover



    def calculate_dynamic_obstacle_weight(self, detection):
        """Calculate weight/importance of an obstacle based on type, size and distance"""
        # Base weight starts from confidence
        weight = detection["confidence"]
        
        # Adjust weight based on object type - trees are most important to avoid
        tree_labels = ["tree", "branch", "plant", "foliage", "trunk"]
        if any(label in detection["label"].lower() for label in tree_labels):
            weight *= 1.5  # Higher weight for trees
        
        # Adjust by distance - closer objects are more important
        # Exponential falloff with distance
        distance_factor = np.exp(-detection["depth"] / self.obstacle_threshold)
        weight *= distance_factor
        
        # Adjust by size - larger objects are more important
        bbox = detection["bbox"]
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = 480 * 640  # Assuming standard resolution
        size_factor = np.sqrt(box_area / image_area)  # Square root to moderate the effect
        weight *= (1 + size_factor)
        
        return weight

    def adaptive_velocity_control(self, target_position, obstacles=None, yolo_detections=None):
        """Generate velocity commands with adaptive obstacle avoidance specifically for forest environments"""
        # Calculate base velocity vector toward target
        direction = np.array([
            target_position[0] - self.current_position[0],
            target_position[1] - self.current_position[1],
            target_position[2] - self.current_position[2]
        ])
        
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Very close to target
            return np.zeros(3)
        
        # Normalize direction
        direction = direction / distance
        
        # Base velocity scaled by distance (slower when close)
        max_speed = self.cruise_speed
        target_speed = min(max_speed, max_speed * min(distance / 5.0, 1.0))
        velocity = direction * target_speed
        
        # Apply obstacle avoidance if obstacles present
        if obstacles or (yolo_detections and len(yolo_detections) > 0):
            # Initialize repulsive vector
            repulsive = np.zeros(3)
            
            # Process YOLO detections - more precise for trees
            if yolo_detections:
                for detection in yolo_detections:
                    # Only consider relevant detections
                    if detection["depth"] > self.obstacle_threshold * 2:
                        continue
                    
                    # Calculate repulsive force weight
                    weight = self.calculate_dynamic_obstacle_weight(detection)
                    
                    # Convert detection to 3D position
                    obstacle_pos = self.convert_detection_to_obstacle(detection)
                    
                    # Vector from obstacle to drone
                    vec_from_obstacle = np.array([
                        self.current_position[0] - obstacle_pos[0],
                        self.current_position[1] - obstacle_pos[1],
                        self.current_position[2] - obstacle_pos[2]
                    ])
                    
                    # Normalize and apply inverse square law
                    obstacle_dist = np.linalg.norm(vec_from_obstacle)
                    if obstacle_dist > 0.1:  # Avoid division by very small numbers
                        repulsive_magnitude = weight / (obstacle_dist * obstacle_dist)
                        repulsive += (vec_from_obstacle / obstacle_dist) * repulsive_magnitude
            
            # Process general obstacles
            if obstacles:
                for obs in obstacles:
                    # Convert to 3D by adding current height
                    obs_3d = (obs[0], obs[1], self.current_position[2])
                    
                    # Vector from obstacle to drone
                    vec_from_obstacle = np.array([
                        self.current_position[0] - obs_3d[0],
                        self.current_position[1] - obs_3d[1],
                        0  # Don't consider vertical for these obstacles
                    ])
                    
                    # Normalize and apply inverse square law
                    obstacle_dist = np.linalg.norm(vec_from_obstacle)
                    if obstacle_dist > 0.1:  # Avoid division by very small numbers
                        repulsive_magnitude = 1.0 / (obstacle_dist * obstacle_dist)
                        repulsive += (vec_from_obstacle / obstacle_dist) * repulsive_magnitude
            
            # Normalize repulsive vector if non-zero
            repulsive_magnitude = np.linalg.norm(repulsive)
            if repulsive_magnitude > 0:
                repulsive = repulsive / repulsive_magnitude
                
                # Scale based on closest obstacle
                closest_obstacle_distance = float('inf')
                
                if yolo_detections:
                    for detection in yolo_detections:
                        closest_obstacle_distance = min(closest_obstacle_distance, detection["depth"])
                
                # Adjust repulsive influence based on closest obstacle
                # Closer obstacles have more influence
                repulsive_influence = min(1.0, self.obstacle_threshold / max(closest_obstacle_distance, 0.1))
                repulsive *= target_speed * repulsive_influence
                
                # Combine attractive and repulsive vectors with weighted sum
                velocity = (1 - repulsive_influence) * velocity + repulsive_influence * repulsive
        
        # Specific forest navigation - check for openings above
        # Forest environments often have vertical passages
        depth_img = self.get_depth_image()
        if depth_img is not None:
            h, w = depth_img.shape
            vertical_clearance = np.percentile(depth_img[:h//3, :], 25)  # Top 1/3 of image
            h, w = depth_img.shape  # Define height (h) and width (w) of the depth image
            horizontal_blocked = np.percentile(depth_img[h//3:, :], 25) < self.obstacle_threshold
            
            if horizontal_blocked and vertical_clearance > self.obstacle_threshold*2:
                # Clear path above, add strong vertical component
                velocity[2] = -min(1.0 + (vertical_clearance/10), 3.0)  # Up to 3m/s ascent

        # Add tangential movement component around obstacles
        if len(yolo_detections) > 0:
            closest_obstacle = min(yolo_detections, key=lambda x: x['depth'])
            if closest_obstacle['depth'] < self.obstacle_threshold*1.5:
                # Calculate tangential direction
                obstacle_vec = np.array(self.convert_detection_to_obstacle(closest_obstacle))[:2]
                drone_vec = np.array(self.current_position)[:2]
                direction_to_obstacle = obstacle_vec - drone_vec
                tangent = np.array([-direction_to_obstacle[1], direction_to_obstacle[0]])
                tangent /= np.linalg.norm(tangent)
                
                # Blend with original velocity
                velocity[:2] = 0.7*velocity[:2] + 0.3*tangent*self.cruise_speed
                
        # return velocity
        
        # Ensure velocity magnitude doesn't exceed max_speed
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = velocity / speed * max_speed
            
        return velocity

    def maintain_optimal_height(self):
        """Adjust height based on canopy density"""
        depth_img = self.get_depth_image()
        if depth_img is None:
            return
            
        # Calculate canopy openness
        upper_third = np.mean(depth_img[:depth_img.shape[0]//3, :])
        lower_two_thirds = np.mean(depth_img[depth_img.shape[0]//3:, :])
        
        # Adjust height based on vertical clearance
        if upper_third > 8.0 and self.current_position[2] > self.safe_height:
            # Move up to exploit vertical space
            self.client.moveToZAsync(self.safe_height - 2.0, 1.0)
        elif lower_two_thirds < 4.0 and self.current_position[2] < self.safe_height:
            # Move down to navigate under obstacles
            self.client.moveToZAsync(self.safe_height + 1.0, 1.0)

    def improved_follow_path(self, path):
        """Follow path using adaptive velocity control for forest environments"""
        if not path:
            print("No path to follow")
            return False
            
        # Track progress along the path
        current_target_idx = 0
        path_length = len(path)
        arrival_threshold = 2.0  # Distance at which we consider a waypoint reached
        
        # Main path following loop
        while current_target_idx < path_length:
            self.update_state()
            
            # Get current waypoint target
            current_target = path[current_target_idx]
            
            # Check for obstacles
            yolo_detections = self.run_yolo_detection()
            obstacles = self.detect_obstacles()
            
            # Calculate velocity using potential field approach
            velocity = self.adaptive_velocity_control(current_target, obstacles, yolo_detections)
            
            # Apply velocity for a short duration
            try:
                self.client.moveByVelocityAsync(
                    velocity[0], velocity[1], velocity[2], 0.5
                ).join()
            except Exception as e:
                print(f"Error in velocity control: {e}")
                self.client.hoverAsync()
                return False
            
            # Update state and check progress
            self.update_state()
            distance = np.linalg.norm(np.array([
                current_target[0] - self.current_position[0],
                current_target[1] - self.current_position[1],
                current_target[2] - self.current_position[2]
            ]))
            
            # If we've reached this waypoint, move to next one
            if distance < arrival_threshold:
                current_target_idx += 1
                if current_target_idx >= path_length:
                    print("Path completed")
                    return True
        
        return True
    
    def navigate_in_forest(self, goal_position):
        """Specialized navigation function for dense forest environments"""
        goal = np.array([goal_position[0], goal_position[1], self.safe_height])
        print(f"Starting forest navigation to goal: {goal}")

        # Initialize state
        self.update_state()
        start_time = time.time()
        stuck_time = 0
        last_replanning_time = 0
        last_progress_check_time = time.time()
        last_position = self.current_position.copy()
        stalled_count = 0
        exploration_mode = False
        exploration_path = []
        last_clear_direction = None
        progress_history = []
        
        # Adjust parameters for forest navigation
        self.obstacle_threshold = 4.0  # Increased for forest (was 3.0)
        self.replanning_distance = 3.0  # Shorter replanning distance (was 5.0)
        self.cruise_speed = 2.5  # Slightly slower for more precision (was 3.0)
        
        # Initial path planning
        obstacles = self.detect_obstacles()
        self.path = self.plan_path(self.current_position, goal, obstacles)
        if self.path:
            self.path = self.smooth_path(self.path)
            print(f"Initial path planned with {len(self.path)} waypoints")
        else:
            print("Failed to find initial path, will use reactive navigation")

        # Main navigation loop
        while np.linalg.norm(self.current_position[:2] - goal[:2]) > self.goal_reached_threshold:
            loop_start = time.time()
            self.update_state()
            
            # Progress check - are we making progress toward the goal?
            current_time = time.time()
            if current_time - last_progress_check_time >= 5.0:
                distance_moved = np.linalg.norm(self.current_position - last_position)
                distance_to_goal = np.linalg.norm(self.current_position[:2] - goal[:2])
                
                print(f"Progress check: Moved {distance_moved:.2f}m in last 5s, Distance to goal: {distance_to_goal:.2f}m")
                
                if distance_moved < 1.0:  # Less than 1m in 5 seconds
                    stalled_count += 1
                    print(f"Limited progress detected (stall count: {stalled_count})")
                    
                    if stalled_count >= 3:  # Third consecutive stall
                        print("Navigation stalled - attempting special forest maneuvers")
                        # Try vertical exploration in forests
                        print("Attempting vertical exploration...")
                        self.move_drone("up", 2.0, 2.0)
                        self.update_state()
                        
                        # Then try horizontal movement
                        # Try moving in a random direction to break free
                        directions = ["forward", "left", "right"]
                        random_dir = random.choice(directions)
                        print(f"Moving randomly: {random_dir}")
                        self.move_drone(random_dir, 2.5, 2.5)
                        
                        # Reset path and counters
                        self.path = []
                        stalled_count = 0
                else:
                    stalled_count = 0  # Reset if we're making progress
                    
                last_progress_check_time = current_time
                last_position = self.current_position.copy()

             # Enhanced stuck detection
            if len(progress_history) > 5:
                recent_progress = sum(progress_history[-5:])
                if recent_progress < 2.0:  # <2m progress in 25s
                    print("Entering exploration mode")
                    exploration_mode = True
                    exploration_path = self.generate_exploration_pattern()
                    
            if exploration_mode:
                if self.follow_exploration_path(exploration_path):
                    exploration_mode = False
                    progress_history = []
            
            progress_history.append(distance_moved)

            # Detect obstacles and check environment
            yolo_detections = self.run_yolo_detection()
            obstacles = self.detect_obstacles()
            
            # Specialized tree detection logic
            tree_detected = False
            tree_direction = "none"
            if yolo_detections:
                for detection in yolo_detections:
                    tree_labels = ["tree", "branch", "plant", "foliage", "trunk"]
                    is_tree = any(label in detection["label"].lower() for label in tree_labels)
                    
                    if is_tree and detection["depth"] < self.obstacle_threshold * 1.3:
                        tree_detected = True
                        # Determine direction of tree
                        center_x = detection["center"][0]
                        if center_x < 213:  # Left third of image
                            tree_direction = "left"
                        elif center_x > 426:  # Right third of image
                            tree_direction = "right"
                        else:
                            tree_direction = "center"
                        break
            
            # Periodic replanning based on time or conditions
            should_replan = False
            
            # Replan if we detect trees close to our path
            if tree_detected and tree_direction == "center":
                should_replan = True
            
            # Replan periodically
            if time.time() - last_replanning_time > 5.0:  # Replan every 5 seconds
                should_replan = True
            
            if should_replan:
                print("Replanning path...")
                obstacles = self.detect_obstacles()
                self.path = self.plan_path(self.current_position, goal, obstacles)
                if self.path:
                    self.path = self.smooth_path(self.path)
                    print(f"New path planned with {len(self.path)} waypoints")
                last_replanning_time = time.time()
            
            # Choose navigation approach
            if tree_detected:
                # React to detected trees
                print(f"Tree detected in {tree_direction} direction")
                
                if tree_direction == "center":
                    # Tree directly ahead - choose alternative direction
                    if random.random() < 0.5:
                        print("Avoiding tree by moving left")
                        self.move_drone("left", 1.5, 2.0)
                    else:
                        print("Avoiding tree by moving right")
                        self.move_drone("right", 1.5, 2.0)
                elif tree_direction == "left":
                    print("Tree on left - moving right")
                    self.move_drone("right", 1.0, 2.0)
                elif tree_direction == "right":
                    print("Tree on right - moving left")
                    self.move_drone("left", 1.0, 2.0)
                    
                self.update_state()
            elif self.path:
                # Follow planned path with velocity control
                if len(self.path) > 0:
                    # Get next waypoint
                    next_waypoint = self.path[0] if len(self.path) == 1 else self.path[1]
                    
                    # Calculate velocity using potential field approach
                    velocity = self.adaptive_velocity_control(next_waypoint, obstacles, yolo_detections)
                    
                    # Apply velocity
                    try:
                        print(f"Moving with velocity: {velocity}")
                        self.client.moveByVelocityAsync(
                            velocity[0], velocity[1], velocity[2], 0.5
                        ).join()
                    except Exception as e:
                        print(f"Error in velocity control: {e}")
                        self.client.hoverAsync()
                    
                    # Update state and check if we reached the current waypoint
                    self.update_state()
                    if len(self.path) > 0 and np.linalg.norm(self.current_position - np.array(self.path[0])) < 2.0:
                        print("Waypoint reached, moving to next")
                        self.path.pop(0)
            else:
                # No path - use reactive navigation directly toward goal
                print("Using reactive navigation toward goal")
                
                # Direct vector to goal
                vec_to_goal = np.array([
                    goal[0] - self.current_position[0],
                    goal[1] - self.current_position[1],
                    goal[2] - self.current_position[2]
                ])
                distance_to_goal = np.linalg.norm(vec_to_goal)
                
                # Normalize
                if distance_to_goal > 0:
                    vec_to_goal = vec_to_goal / distance_to_goal
                
                # Scale by speed
                target_velocity = vec_to_goal * min(self.cruise_speed, distance_to_goal)
                
                # Adjust for obstacles
                if obstacles or yolo_detections:
                    # Use potential field for reactive movement
                    velocity = self.adaptive_velocity_control(goal, obstacles, yolo_detections)
                else:
                    velocity = target_velocity
                
                # Apply velocity
                try:
                    self.client.moveByVelocityAsync(
                        velocity[0], velocity[1], velocity[2], 0.5
                    ).join()
                except Exception as e:
                    print(f"Error in velocity control: {e}")
                    self.client.hoverAsync()
            
            # Safety check - make sure we don't get stuck forever
            if time.time() - start_time > 600:  # 5 minutes timeout
                print("Navigation timeout reached. Ending navigation.")
                break
            
            # Maintain loop rate
            loop_duration = time.time() - loop_start
            if loop_duration < 0.1:
                time.sleep(0.1 - loop_duration)

        print("Goal reached!")
        self.client.hoverAsync().join()
        return True
    
    def generate_exploration_pattern(self):
        """Generate spiral exploration pattern when stuck"""
        pattern = []
        current_pos = self.current_position
        steps = 5
        radius = 3.0
        
        for i in range(1, steps+1):
            angle = i * np.pi/2
            x = current_pos[0] + radius*i * np.cos(angle)
            y = current_pos[1] + radius*i * np.sin(angle)
            pattern.append((x, y, self.safe_height))
            
            # Add vertical component
            pattern.append((x, y, self.safe_height-3.0))
            
        return pattern
        
    def follow_exploration_path(self, path):
        """Follow exploration path with obstacle checking"""
        for point in path:
            if self.is_point_blocked(point):
                continue
                
            while np.linalg.norm(self.current_position - point) > 1.0:
                velocity = self.adaptive_velocity_control(point)
                self.client.moveByVelocityAsync(*velocity, 0.5).join()
                self.update_state()
                
                if self.check_for_escape_route():
                    return True
        return False
    
    # def follow_path(self, path):
    #     """Follow a computed path using velocity vectors instead of waypoint movement"""
    #     if not path:
    #         print("No path to follow")
    #         return False
            
    #     # Track progress along the path
    #     current_target_idx = 0
    #     path_length = len(path)
        
    #     # Parameters for velocity-based control
    #     look_ahead_distance = 5.0  # Look ahead distance in meters
    #     max_velocity = self.cruise_speed  # Maximum velocity magnitude
    #     arrival_threshold = 2.0  # Distance at which we consider a waypoint reached
        
    #     # Main path following loop
    #     while current_target_idx < path_length:
    #         self.update_state()
            
    #         # Get current waypoint target
    #         current_target = path[current_target_idx]
            
    #         # Check for obstacles using YOLO and other sensors
    #         yolo_detections = self.run_yolo_detection()
    #         close_obstacles = any(d["depth"] < self.obstacle_threshold * 1.5 for d in yolo_detections)
            
    #         # Also check traditional sensors for obstacles
    #         obstacles = self.detect_obstacles()
            
    #         # If we detect obstacles, handle them reactively
    #         if close_obstacles or obstacles:
    #             print("Obstacle detected during path following, using reactive avoidance")
    #             move_direction = self.evaluate_movement_direction()
                
    #             if move_direction != "forward":
    #                 print(f"Avoiding obstacle by moving {move_direction}")
    #                 self.move_drone(move_direction, duration=1.0, speed=2.0)
    #                 return False  # Signal for replanning
            
    #         # Calculate vector to current waypoint
    #         direction_vector = np.array([
    #             current_target[0] - self.current_position[0],
    #             current_target[1] - self.current_position[1],
    #             current_target[2] - self.current_position[2]
    #         ])
            
    #         # Distance to current waypoint
    #         distance = np.linalg.norm(direction_vector)
            
    #         # If we've reached this waypoint, move to next one
    #         if distance < arrival_threshold:
    #             current_target_idx += 1
    #             if current_target_idx >= path_length:
    #                 print("Path completed")
    #                 return True
    #             continue
            
    #         # Normalize direction vector
    #         if distance > 0:
    #             direction_vector = direction_vector / distance
            
    #         # Scale by desired speed (slower when close to waypoint)
    #         speed_factor = min(distance / look_ahead_distance, 1.0)
    #         velocity = direction_vector * max_velocity * speed_factor
            
    #         # Execute velocity command
    #         try:
    #             # Move for a short duration using velocity control
    #             self.client.moveByVelocityAsync(
    #                 velocity[0],  # vx
    #                 velocity[1],  # vy
    #                 velocity[2],  # vz
    #                 0.5  # Duration - short to allow frequent updates
    #             ).join()
                
    #         except Exception as e:
    #             print(f"Error in velocity control: {e}")
    #             self.client.hoverAsync()
    #             return False
        
    #     return True


    # def navigate_to_goal(self, goal_position):
    #     """Main navigation function to reach a goal point using velocity control."""
    #     goal = np.array([goal_position[0], goal_position[1], self.safe_height])
    #     print(f"Starting navigation to goal: {goal}")

    #     self.update_state()

    #     # Initialize stuck-checking state
    #     current_time = time.time()
    #     self.last_position_check_time = current_time
    #     self.last_checked_position = self.current_position.copy()
    #     stuck_counter = 0  # Count consecutive times the drone appears stuck
        
    #     # Initial path planning
    #     obstacles = self.detect_obstacles()
    #     self.path = self.plan_path(self.current_position, goal, obstacles)
    #     if self.path:
    #         self.path = self.smooth_path(self.path)
    #     else:
    #         print("Failed to find initial path, will attempt reactive exploration...")

    #     while np.linalg.norm(self.current_position[:2] - goal[:2]) > self.goal_reached_threshold:
    #         self.update_state()

    #         # Periodically check for being stuck
    #         current_time = time.time()
    #         if current_time - self.last_position_check_time >= 3.0:
    #             distance_moved = np.linalg.norm(self.current_position - self.last_checked_position)
    #             if distance_moved < 0.5:
    #                 print("Drone seems stuck. Attempting recovery...")
    #                 stuck_counter += 1
                    
    #                 # Try more aggressive recovery if stuck multiple times
    #                 recovery_speed = min(3.0 + stuck_counter * 0.5, 6.0)  # Increase speed with stuck count
    #                 recovery_duration = min(1.0 + stuck_counter * 0.3, 2.5)  # Increase duration with stuck count
                    
    #                 # Try different recovery directions
    #                 recovery_directions = ["up", "backward", "left", "right"]
    #                 recovery_successful = False
                    
    #                 for recovery_dir in recovery_directions:
    #                     self.move_drone(recovery_dir, duration=recovery_duration, speed=recovery_speed)
    #                     self.update_state()
    #                     new_distance = np.linalg.norm(self.current_position - self.last_checked_position)
    #                     if new_distance > 0.8:  # Higher threshold for successful recovery
    #                         print(f"Recovery successful using {recovery_dir}.")
    #                         recovery_successful = True
    #                         stuck_counter = 0  # Reset counter on success
    #                         break
                    
    #                 if not recovery_successful:
    #                     print("Recovery failed. Using direct velocity control to break free.")
    #                     # Try random velocity burst
    #                     vx = np.random.uniform(-2.0, 2.0)
    #                     vy = np.random.uniform(-2.0, 2.0)
    #                     vz = np.random.uniform(-1.0, -0.2)  # Slight upward bias
    #                     self.client.moveByVelocityAsync(vx, vy, vz, 2.0).join()
    #             else:
    #                 stuck_counter = 0  # Reset counter if moving normally
                    
    #             self.last_position_check_time = current_time
    #             self.last_checked_position = self.current_position.copy()

    #         # If no valid path, explore reactively to find one
    #         if not self.path:
    #             print("No path available - using reactive exploration")
    #             # Check sensor data to determine best exploration direction
    #             move_direction = self.evaluate_movement_direction()
                
    #             if move_direction == "forward":
    #                 print("Exploring forward")
    #                 # Direct velocity control toward goal with obstacle checking
    #                 vec_to_goal = np.array([goal[0] - self.current_position[0], 
    #                                     goal[1] - self.current_position[1]])
    #                 dist_to_goal = np.linalg.norm(vec_to_goal)
                    
    #                 if dist_to_goal > 0:
    #                     # Normalize and scale
    #                     vec_to_goal = vec_to_goal / dist_to_goal * min(self.cruise_speed, dist_to_goal)
                        
    #                     # Apply velocity toward goal
    #                     self.client.moveByVelocityAsync(
    #                         vec_to_goal[0], vec_to_goal[1], 0, 1.0
    #                     ).join()
                        
    #             else:
    #                 print(f"Avoiding obstacle by moving {move_direction}")
    #                 self.move_drone(move_direction, 1.0)
                    
    #             self.update_state()
    #             obstacles = self.detect_obstacles()
    #             self.path = self.plan_path(self.current_position, goal, obstacles)
                
    #             if self.path:
    #                 print("Found a path after exploration.")
    #                 self.path = self.smooth_path(self.path)
    #             continue

    #         # Check if we need to replan due to obstacles
    #         yolo_detections = self.run_yolo_detection()
    #         close_obstacles = any(d["depth"] < self.obstacle_threshold * 1.2 for d in yolo_detections)
            
    #         if close_obstacles:
    #             move_direction = self.evaluate_movement_direction()
    #             if move_direction != "forward":
    #                 print(f"Obstacle detected, moving {move_direction}")
    #                 self.move_drone(move_direction, 1.0)
    #                 self.update_state()

    #                 obstacles = self.detect_obstacles()
    #                 self.path = self.plan_path(self.current_position, goal, obstacles)
    #                 if self.path:
    #                     self.path = self.smooth_path(self.path)
    #                 continue

    #         # Try to follow the planned path using velocity control
    #         path_completed = self.follow_path(self.path)
    #         self.update_state()

    #         if not path_completed:
    #             print("Path interrupted - replanning")
    #             obstacles = self.detect_obstacles()
    #             self.path = self.plan_path(self.current_position, goal, obstacles)
    #             if self.path:
    #                 self.path = self.smooth_path(self.path)
    #         else:
    #             # We completed the path but haven't reached the goal yet
    #             obstacles = self.detect_obstacles()
    #             self.path = self.plan_path(self.current_position, goal, obstacles)
    #             if self.path:
    #                 self.path = self.smooth_path(self.path)

    #         distance_to_goal = np.linalg.norm(self.current_position[:2] - goal[:2])
    #         print(f"Distance to goal: {distance_to_goal:.2f} meters")

    #     print("Goal reached!")
    #     self.client.hoverAsync().join()
    #     return True
    
    def land_and_disarm(self):
        """Land the drone and disarm"""
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Navigation complete!")
        
        # Clean up OpenCV windows if any
        cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    try:
        nav_system = EnhancedDroneNavigationSystem()
        
        # Define the goal position (x, y) in meters
        goal_position = (50, 50)
        
        # Navigate to the goal
        nav_system.navigate_in_forest(goal_position)
        
        # Land and cleanup
        nav_system.land_and_disarm()
        
    except KeyboardInterrupt:
        print("Navigation interrupted by user")
        # Make sure we land safely on interruption
        client = airsim.MultirotorClient()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Emergency landing
        client = airsim.MultirotorClient()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()




# import airsim
# import numpy as np
# import time
# import sys
# import os
# import math
# import cv2
# import torch
# from ultralytics import YOLO

# class ImprovedDroneNavigationSystem:
#     def __init__(self):
#         # Connect to AirSim
#         self.client = airsim.MultirotorClient()
#         self.client.confirmConnection()
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)
        
#         # Take off
#         self.client.takeoffAsync().join()
        
#         # Load YOLO model
#         self.yolo_model = YOLO(r'model/TrainedWeights/best_yolov10n_obstacle.pt')
#         self.yolo_model.fuse()  # Fuse layers for faster inference
#         print("YOLO model loaded successfully")
        
#         # Navigation parameters
#         self.obstacle_threshold = 5.0  # Increased from 3.0 meters for safety
#         self.safe_height = -5.0  # meters (negative is up in AirSim)
#         self.cruise_speed = 2.5  # Reduced from 3.0 m/s for better control
#         self.current_position = np.zeros(3)
#         self.current_orientation = np.zeros(3)  # roll, pitch, yaw in radians
#         self.goal_reached_threshold = 2.0  # meters
        
#         # Enhanced obstacle avoidance parameters
#         self.obstacle_memory = {}  # Dictionary to store obstacle locations with timestamp
#         self.obstacle_memory_timeout = 10.0  # Increased from 5.0 seconds
#         self.safety_buffer = 5.0  # Increased buffer around obstacles
#         self.emergency_stop_distance = 3.0  # Distance at which to stop forward movement
        
#         # Visualization
#         self.show_detections = True  # Set to False to disable visualization
        
#         # Initialize state
#         self.update_state()
        
#         # Path planning parameters
#         self.path = []
#         self.last_obstacle_detected = False
#         self.required_clear_frames = 5  # Number of frames with no obstacles before resuming
#         self.clear_frame_counter = 0
        
#         # Free path detection parameters
#         self.path_sectors = 3  # Left, Center, Right
#         self.current_free_sector = 1  # Center by default
#         self.sector_scores = np.zeros(self.path_sectors)  # Scores for each sector
#         self.sector_history = []  # Keep history of good sectors
#         self.sector_history_length = 5
        
#     def update_state(self):
#         """Update drone state from sensors"""
#         state = self.client.getMultirotorState()
#         self.current_position = np.array([
#             state.kinematics_estimated.position.x_val,
#             state.kinematics_estimated.position.y_val,
#             state.kinematics_estimated.position.z_val
#         ])
        
#         # Get orientation (quaternion to Euler)
#         q = state.kinematics_estimated.orientation
#         self.current_orientation = self.quaternion_to_euler(
#             q.w_val, q.x_val, q.y_val, q.z_val
#         )
    
#     def quaternion_to_euler(self, w, x, y, z):
#         """Convert quaternion to Euler angles (roll, pitch, yaw)"""
#         # Roll (x-axis rotation)
#         sinr_cosp = 2 * (w * x + y * z)
#         cosr_cosp = 1 - 2 * (x * x + y * y)
#         roll = np.arctan2(sinr_cosp, cosr_cosp)
        
#         # Pitch (y-axis rotation)
#         sinp = 2 * (w * y - z * x)
#         if np.abs(sinp) >= 1:
#             pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
#         else:
#             pitch = np.arcsin(sinp)
        
#         # Yaw (z-axis rotation)
#         siny_cosp = 2 * (w * z + x * y)
#         cosy_cosp = 1 - 2 * (y * y + z * z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
        
#         return np.array([roll, pitch, yaw])
    
#     def get_rgb_image(self):
#         """Retrieve RGB image from AirSim for YOLO processing"""
#         try:
#             responses = self.client.simGetImages([
#                 airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
#             ])
            
#             if not responses or len(responses) == 0:
#                 print("Warning: No RGB image received")
#                 return None
                
#             img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
#             img = img.reshape(responses[0].height, responses[0].width, 3)
            
#             if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
#                 print("Warning: Invalid RGB image dimensions")
#                 return None
                
#             return img
#         except Exception as e:
#             print(f"Error getting RGB image: {e}")
#             return None
    
#     def get_depth_image(self):
#         """Retrieve depth image from AirSim with error handling"""
#         try:
#             responses = self.client.simGetImages([
#                 airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
#             ])
            
#             if not responses or len(responses) == 0:
#                 print("Warning: No depth image received")
#                 return None
                
#             img = np.array(responses[0].image_data_float, dtype=np.float32)
#             img = img.reshape(responses[0].height, responses[0].width)
            
#             # Check for invalid values
#             if np.isnan(img).any() or np.isinf(img).any():
#                 print("Warning: Depth image contains invalid values")
#                 img[np.isnan(img)] = 100.0  # Replace NaN with a large distance
#                 img[np.isinf(img)] = 100.0  # Replace Inf with a large distance
                
#             return img
#         except Exception as e:
#             print(f"Error getting depth image: {e}")
#             return None
    
#     def get_lidar_data(self):
#         """Get LiDAR data for 3D obstacle detection"""
#         lidar_data = self.client.getLidarData()
#         if len(lidar_data.point_cloud) < 3:
#             return np.array([])
            
#         points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
#         return points

#     def detect_free_path(self, rgb_image=None, depth_image=None):
#         """
#         Analyze the image to find the most traversable path
#         Returns the best sector (0=left, 1=center, 2=right) and its score
#         """
#         if rgb_image is None:
#             rgb_image = self.get_rgb_image()
        
#         if depth_image is None:
#             depth_image = self.get_depth_image()
            
#         if rgb_image is None or depth_image is None:
#             return 1, 0.5  # Default to center with medium confidence
        
#         # Run YOLO detection
#         yolo_results = self.yolo_model(rgb_image)
        
#         # Prepare obstacle mask - start with all free
#         h, w = rgb_image.shape[:2]
#         obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        
#         # Add obstacles from YOLO detections
#         for detection in yolo_results[0].boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = detection
#             if conf > 0.3:  # Confidence threshold
#                 # Draw the obstacle on the mask
#                 cv2.rectangle(obstacle_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
                
#                 # Add extra safety margin around obstacles
#                 cv2.rectangle(obstacle_mask, 
#                               (max(0, int(x1)-20), max(0, int(y1)-20)), 
#                               (min(w, int(x2)+20), min(h, int(y2)+20)), 
#                               255, -1)
        
#         # Use depth information to enhance obstacle detection
#         if depth_image is not None:
#             # Mark areas with close depth as obstacles
#             close_mask = (depth_image < self.obstacle_threshold).astype(np.uint8) * 255
#             obstacle_mask = cv2.bitwise_or(obstacle_mask, close_mask)
        
#         # Divide image into sectors and score each sector
#         sector_width = w // self.path_sectors
#         sectors = []
#         sector_scores = []
        
#         for i in range(self.path_sectors):
#             # Extract sector
#             sector_start = i * sector_width
#             sector_end = (i + 1) * sector_width
#             sector_mask = obstacle_mask[:, sector_start:sector_end]
            
#             # Calculate free space percentage
#             free_space = 1.0 - (np.sum(sector_mask) / (255 * sector_mask.size))
#             sector_scores.append(free_space)
#             sectors.append(sector_mask)
        
#         # Add depth-based weighting - prefer further paths
#         if depth_image is not None:
#             for i in range(self.path_sectors):
#                 sector_start = i * sector_width
#                 sector_end = (i + 1) * sector_width
#                 sector_depth = depth_image[:, sector_start:sector_end]
                
#                 # Calculate average depth in the non-obstacle areas
#                 valid_depths = sector_depth[sectors[i] == 0]
#                 if len(valid_depths) > 0:
#                     avg_depth = np.mean(valid_depths)
#                     # Normalize depth score (0-1) and weight it
#                     depth_score = min(avg_depth / 20.0, 1.0)  # Cap at 20 meters
#                     # Combine scores (70% free space, 30% depth)
#                     sector_scores[i] = 0.7 * sector_scores[i] + 0.3 * depth_score
        
#         # Apply temporal smoothing using history
#         self.sector_scores = 0.7 * np.array(sector_scores) + 0.3 * self.sector_scores
        
#         # Find best sector
#         best_sector = np.argmax(self.sector_scores)
#         best_score = self.sector_scores[best_sector]
        
#         # Update history
#         self.sector_history.append(best_sector)
#         if len(self.sector_history) > self.sector_history_length:
#             self.sector_history.pop(0)
        
#         # Use mode of recent history for stability
#         from scipy import stats
#         stable_sector = stats.mode(self.sector_history)[0][0]
        
#         # Visualization
#         if self.show_detections:
#             vis_img = rgb_image.copy()
            
#             # Draw sectors
#             for i in range(self.path_sectors):
#                 sector_start = i * sector_width
#                 sector_end = (i + 1) * sector_width
#                 color = (0, 255, 0) if i == stable_sector else (0, 0, 255)
#                 cv2.line(vis_img, (sector_start, 0), (sector_start, h), (255, 255, 255), 1)
#                 cv2.putText(vis_img, f"S{i}:{self.sector_scores[i]:.2f}", 
#                           (sector_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                           0.5, color, 2)
            
#             # Highlight the chosen sector
#             cv2.rectangle(vis_img, 
#                          (stable_sector * sector_width, 0), 
#                          ((stable_sector + 1) * sector_width, 50), 
#                          (0, 255, 0), 2)
            
#             # Show obstacle mask (semi-transparent overlay)
#             overlay = vis_img.copy()
#             obstacle_mask_color = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR)
#             obstacle_mask_color[obstacle_mask > 0] = [0, 0, 255]  # Red obstacles
#             cv2.addWeighted(obstacle_mask_color, 0.3, overlay, 0.7, 0, vis_img)
            
#             # Display
#             cv2.imshow("Free Path Detection", vis_img)
#             cv2.waitKey(1)
        
#         return stable_sector, best_score
    
#     def run_yolo_detection(self):
#         """Perform YOLO object detection on the current camera view"""
#         rgb_image = self.get_rgb_image()
#         if rgb_image is None:
#             return []
        
#         depth_img = self.get_depth_image()
        
#         # Run YOLO inference
#         try:
#             results = self.yolo_model(rgb_image)
#             detections = results[0]
            
#             # Process detections
#             detected_objects = []
#             for detection in detections.boxes.data.tolist():
#                 x1, y1, x2, y2, conf, cls = detection
                
#                 if conf < 0.3:  # Increased confidence threshold
#                     continue
                    
#                 label = detections.names[int(cls)]
                
#                 # Calculate center point of bounding box
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
                
#                 # Calculate width and height of bounding box
#                 width = x2 - x1
#                 height = y2 - y1
                
#                 # Size-based depth estimation (larger objects are likely closer)
#                 # This is a fallback when no depth info is available
#                 size_based_depth = 10.0 * (300.0 / (width * height)) ** 0.5
                
#                 # Estimate distance using depth image or LiDAR
#                 if depth_img is not None:
#                     h, w = depth_img.shape
#                     # Convert bbox center to depth image coordinates
#                     cx = min(int(center_x * w / rgb_image.shape[1]), w-1)
#                     cy = min(int(center_y * h / rgb_image.shape[0]), h-1)
                    
#                     # Get depth at center point (with small averaging window)
#                     window_size = 9  # Increased window size
#                     x_start = max(0, cx - window_size//2)
#                     x_end = min(w, cx + window_size//2)
#                     y_start = max(0, cy - window_size//2)
#                     y_end = min(h, cy + window_size//2)
                    
#                     depth_window = depth_img[y_start:y_end, x_start:x_end]
#                     if depth_window.size > 0:
#                         # Use a more conservative estimate (closer distance)
#                         valid_depths = depth_window[depth_window < 50]  # Filter out very large values
#                         if len(valid_depths) > 0:
#                             # Use 20th percentile (more conservative than mean)
#                             depth = np.percentile(valid_depths, 20)
#                         else:
#                             depth = size_based_depth
#                     else:
#                         depth = size_based_depth
#                 else:
#                     depth = size_based_depth
                
#                 # Make tree objects appear closer for safety
#                 tree_labels = ["tree", "branch", "plant", "foliage", "trunk"]
#                 if any(tree_term in label.lower() for tree_term in tree_labels):
#                     depth *= 0.7  # Trees are treated as 30% closer than they appear
                
#                 detected_objects.append({
#                     "label": label,
#                     "bbox": (int(x1), int(y1), int(x2), int(y2)),
#                     "confidence": conf,
#                     "depth": depth,
#                     "center": (center_x, center_y),
#                     "size": width * height
#                 })
            
#             # Visualize detections if enabled
#             if self.show_detections and rgb_image is not None:
#                 vis_img = rgb_image.copy()
#                 for obj in detected_objects:
#                     x1, y1, x2, y2 = obj['bbox']
#                     color = (0, 0, 255) if obj['depth'] < self.obstacle_threshold else (0, 255, 0)
#                     cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                     cv2.putText(vis_img, f"{obj['label']} {obj['depth']:.1f}m", 
#                                 (int(x1), int(y1) - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
#                 cv2.imshow("YOLO Detections", vis_img)
#                 cv2.waitKey(1)
            
#             return detected_objects
            
#         except Exception as e:
#             print(f"Error in YOLO detection: {e}")
#             return []

#     def detect_obstacles(self):
#         """Detect obstacles using YOLO, depth image, and LiDAR"""
#         current_time = time.time()
#         close_obstacles_detected = False
        
#         # Get YOLO detections
#         detections = self.run_yolo_detection()
        
#         # Check if any obstacles are too close (emergency stop condition)
#         for detection in detections:
#             if detection["depth"] < self.emergency_stop_distance:
#                 close_obstacles_detected = True
#                 print(f"WARNING: {detection['label']} detected at {detection['depth']:.1f}m - too close!")
                
#         # Get depth image to check if there's a clear path
#         depth_img = self.get_depth_image()
#         if depth_img is not None:
#             # Check for close objects in center of view
#             h, w = depth_img.shape
#             center_region = depth_img[h//4:3*h//4, w//3:2*w//3]
#             if np.min(center_region) < self.emergency_stop_distance:
#                 close_obstacles_detected = True
#                 print(f"WARNING: Close obstacle detected at {np.min(center_region):.1f}m in center view!")
        
#         # Use LiDAR for 360 degree detection
#         lidar_points = self.get_lidar_data()
#         if len(lidar_points) > 0:
#             # Check only points in front hemisphere
#             front_points = lidar_points[lidar_points[:, 0] > 0]
#             if len(front_points) > 0:
#                 distances = np.linalg.norm(front_points, axis=1)
#                 if np.min(distances) < self.emergency_stop_distance:
#                     close_obstacles_detected = True
#                     print(f"WARNING: Close obstacle detected at {np.min(distances):.1f}m by LiDAR!")
        
#         return close_obstacles_detected

#     def navigate_with_free_path_detection(self, goal_position):
#         """Main navigation function using free path detection"""
#         goal = np.array([goal_position[0], goal_position[1], self.safe_height])
#         print(f"Starting navigation to goal: {goal}")
        
#         # Safety parameters
#         max_speed = self.cruise_speed
#         min_speed = 0.5
#         rotation_speed = 15  # degrees per second
        
#         while np.linalg.norm(self.current_position[:2] - goal[:2]) > self.goal_reached_threshold:
#             self.update_state()
            
#             # Check for obstacles
#             obstacle_detected = self.detect_obstacles()
            
#             if obstacle_detected:
#                 print("Obstacle detected! Stopping and finding new path...")
#                 self.client.hoverAsync()
#                 self.last_obstacle_detected = True
#                 self.clear_frame_counter = 0
#                 time.sleep(0.5)  # Pause briefly
#                 continue
            
#             # If we recently detected an obstacle, wait for several clear frames
#             if self.last_obstacle_detected:
#                 self.clear_frame_counter += 1
#                 if self.clear_frame_counter < self.required_clear_frames:
#                     print(f"Waiting for clear path confirmation ({self.clear_frame_counter}/{self.required_clear_frames})")
#                     self.client.hoverAsync()
#                     time.sleep(0.2)
#                     continue
#                 else:
#                     print("Path appears clear, resuming navigation")
#                     self.last_obstacle_detected = False
            
#             # Find best free path sector
#             rgb_image = self.get_rgb_image()
#             depth_image = self.get_depth_image()
#             best_sector, confidence = self.detect_free_path(rgb_image, depth_image)
            
#             # Calculate vector to goal
#             goal_vector = goal[:2] - self.current_position[:2]
#             distance_to_goal = np.linalg.norm(goal_vector)
            
#             # Calculate current yaw
#             yaw = self.current_orientation[2]
#             heading_vector = np.array([np.cos(yaw), np.sin(yaw)])
            
#             # Calculate angle to goal
#             goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
#             angle_diff = (goal_angle - yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            
#             # Combine free path direction with goal direction
#             target_angle = yaw
#             if distance_to_goal > 10:  # Far from goal - prioritize free path
#                 # Convert sector to angle
#                 sector_angles = {
#                     0: -np.pi/6,  # Left: -30 degrees
#                     1: 0,         # Center: 0 degrees
#                     2: np.pi/6    # Right: 30 degrees
#                 }
#                 path_angle = yaw + sector_angles[best_sector]
                
#                 # Weight between goal and free path
#                 path_weight = min(0.8, confidence) 
#                 goal_weight = 1.0 - path_weight
                
#                 # Blend angles with weight
#                 x = path_weight * np.cos(path_angle) + goal_weight * np.cos(goal_angle)
#                 y = path_weight * np.sin(path_angle) + goal_weight * np.sin(goal_angle)
#                 target_angle = np.arctan2(y, x)
#             else:  # Close to goal - focus more on goal direction
#                 target_angle = goal_angle
            
#             # Calculate turn angle
#             turn_angle = (target_angle - yaw + np.pi) % (2 * np.pi) - np.pi
            
#             # If we need to turn significantly, do that first
#             if abs(turn_angle) > np.radians(20):
#                 print(f"Turning {np.degrees(turn_angle):.1f} degrees")
#                 turn_duration = abs(turn_angle) / np.radians(rotation_speed)
#                 turn_direction = np.sign(turn_angle)
                
#                 # Yaw rate in radians/second
#                 yaw_rate = turn_direction * np.radians(rotation_speed)
                
#                 # Use moveByVelocityAsync with yaw rate
#                 self.client.moveByVelocityAsync(0, 0, 0, turn_duration, 
#                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
#                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))
#                 time.sleep(turn_duration)
#                 continue
            
#             # Calculate forward speed based on confidence and distance
#             forward_speed = max_speed * confidence
#             if forward_speed < min_speed:
#                 forward_speed = min_speed
                
#             # If close to goal, slow down
#             if distance_to_goal < 10:
#                 forward_speed = min(forward_speed, distance_to_goal / 3)
            
#             # Move forward in current direction
#             print(f"Moving forward at {forward_speed:.1f} m/s (confidence: {confidence:.2f})")
#             forward_duration = 0.5  # Move in short bursts to react quickly
#             vx = forward_speed * np.cos(yaw)
#             vy = forward_speed * np.sin(yaw)
#             self.client.moveByVelocityAsync(vx, vy, 0, forward_duration)
#             time.sleep(forward_duration)
            
#             # Print progress
#             distance_to_goal = np.linalg.norm(self.current_position[:2] - goal[:2])
#             print(f"Distance to goal: {distance_to_goal:.2f} meters")
        
#         print("Goal reached!")
#         self.client.hoverAsync().join()
#         return True
    
#     def land_and_disarm(self):
#         """Land the drone and disarm"""
#         print("Landing...")
#         self.client.landAsync().join()
#         self.client.armDisarm(False)
#         self.client.enableApiControl(False)
#         print("Navigation complete!")
        
#         # Clean up OpenCV windows if any
#         cv2.destroyAllWindows()

# # Main execution
# if __name__ == "__main__":
#     try:
#         nav_system = ImprovedDroneNavigationSystem()
        
#         # Define the goal position (x, y) in meters
#         goal_position = (50, 50)
        
#         # Navigate to the goal
#         nav_system.navigate_with_free_path_detection(goal_position)
        
#         # Land and cleanup
#         nav_system.land_and_disarm()
        
#     except KeyboardInterrupt:
#         print("Navigation interrupted by user")
#         # Make sure we land safely on interruption
#         client = airsim.MultirotorClient()
#         client.landAsync().join()
#         client.armDisarm(False)
#         client.enableApiControl(False)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         # Emergency landing
#         client = airsim.MultirotorClient()
#         client.landAsync().join()
#         client.armDisarm(False)
#         client.enableApiControl(False)
#         cv2.destroyAllWindows()