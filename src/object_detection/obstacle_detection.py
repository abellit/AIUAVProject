import numpy as np
import time
import cv2
from ultralytics import YOLO
from sensor_processing import SensorProcessor

class ObstacleDetector:
    def __init__(self, client):
        self.client = client
        self.sensor_processor = SensorProcessor(client)
        
        # Detection parameters
        self.obstacle_threshold = 3.0
        self.yolo_confidence_threshold = 0.25
        self.obstacle_memory = {}
        self.obstacle_memory_timeout = 5.0
        self.grid_size = 10
        self.detection_weight = 1.5
        self.show_detections = True
        
        # Load YOLO model
        self.yolo_model = YOLO(r'model/TrainedWeights/best_yolov10n_obstacle.pt')
        self.yolo_model.fuse()
        print("YOLO model loaded successfully")
    
    def detect_obstacles(self):
        """Enhanced obstacle detection using voxelized LiDAR and YOLO"""
        obstacles = set()
        current_time = time.time()
        
        # Process LiDAR with advanced techniques
        processed_points = self.sensor_processor.advanced_lidar_processing()
        
        # Add obstacles from processed LiDAR points
        if len(processed_points) > 0:
            self._add_lidar_obstacles(processed_points, obstacles, current_time)
        
        # Run YOLO detection
        yolo_detections = self.run_yolo_detection()
        for detection in yolo_detections:
            if detection["depth"] < self.obstacle_threshold * 1.5:
                self._add_yolo_obstacle(detection, obstacles, current_time)
        
        # Clean up old obstacles
        self._clean_obstacle_memory(current_time)
        
        return obstacles
    
    def _add_lidar_obstacles(self, points, obstacles, current_time):
        """Add obstacles from LiDAR points"""
        yaw = self.sensor_processor.current_orientation[2]
        current_pos = self.sensor_processor.current_position
        
        for point in points:
            if np.linalg.norm(point) < self.obstacle_threshold:
                world_point = (
                    current_pos[0] + point[0] * np.cos(yaw) - point[1] * np.sin(yaw),
                    current_pos[1] + point[0] * np.sin(yaw) + point[1] * np.cos(yaw)
                )
                obstacle_key = (round(world_point[0]), round(world_point[1]))
                self.obstacle_memory[obstacle_key] = current_time
                obstacles.add(obstacle_key)
    
    def _add_yolo_obstacle(self, detection, obstacles, current_time):
        """Add obstacle from YOLO detection"""
        obstacle_pos = self.convert_detection_to_obstacle(detection)
        obstacle_key = (round(obstacle_pos[0]), round(obstacle_pos[1]))
        self.obstacle_memory[obstacle_key] = current_time
        obstacles.add(obstacle_key)
    
    def _clean_obstacle_memory(self, current_time):
        """Remove old obstacles from memory"""
        expired_obstacles = [
            obs_key for obs_key, timestamp in self.obstacle_memory.items()
            if current_time - timestamp > self.obstacle_memory_timeout
        ]
        
        for obs_key in expired_obstacles:
            del self.obstacle_memory[obs_key]
    
    def run_yolo_detection(self):
        """Perform YOLO object detection on the current camera view"""
        rgb_image = self.sensor_processor.get_rgb_image()
        if rgb_image is None:
            return []
        
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
                depth = self._estimate_detection_depth(rgb_image, detection)
                
                detected_objects.append({
                    "label": label,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "depth": depth,
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2)
                })
            
            self._visualize_detections(rgb_image, detected_objects)
            return detected_objects
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def _estimate_detection_depth(self, rgb_image, detection):
        """Estimate depth for a detection using depth image or LiDAR"""
        depth_img = self.sensor_processor.get_depth_image()
        lidar_points = self.sensor_processor.get_lidar_data()
        
        if depth_img is not None:
            return self._get_depth_from_image(detection, rgb_image, depth_img)
        elif len(lidar_points) > 0:
            return np.mean(lidar_points[:, 0])
        else:
            return 5.0  # Default depth
    
    def _get_depth_from_image(self, detection, rgb_image, depth_img):
        """Get depth from depth image at detection center"""
        x1, y1, x2, y2, _, _ = detection
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        h, w = depth_img.shape
        cx = min(int(center_x * w / rgb_image.shape[1]), w-1)
        cy = min(int(center_y * h / rgb_image.shape[0]), h-1)
        
        window_size = 5
        x_start = max(0, cx - window_size//2)
        x_end = min(w, cx + window_size//2)
        y_start = max(0, cy - window_size//2)
        y_end = min(h, cy + window_size//2)
        
        depth_window = depth_img[y_start:y_end, x_start:x_end]
        return np.mean(depth_window)
    
    def _visualize_detections(self, rgb_image, detections):
        """Visualize detections if enabled"""
        if self.show_detections and rgb_image is not None:
            vis_img = rgb_image.copy()
            for obj in detections:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_img, f"{obj['label']} {obj['depth']:.1f}m", 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Detections", vis_img)
            cv2.waitKey(1)
    
    def convert_detection_to_obstacle(self, detection):
        """Convert a YOLO detection to world coordinates"""
        depth = detection["depth"]
        center_x, center_y = detection["center"]
        
        # Calculate field of view angles
        h, w = 480, 640  # Assuming standard resolution
        fov_h = np.pi/3  # Horizontal FOV (60 degrees)
        fov_v = np.pi/4  # Vertical FOV (45 degrees)
        
        # Get normalized image coordinates (-1 to 1)
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
        yaw = self.sensor_processor.current_orientation[2]
        current_pos = self.sensor_processor.current_position
        
        obstacle_pos = (
            current_pos[0] + x * np.cos(yaw) - y * np.sin(yaw),
            current_pos[1] + x * np.sin(yaw) + y * np.cos(yaw),
            current_pos[2] + z
        )
        
        return obstacle_pos
    
    def adaptive_velocity_control(self, target_position, obstacles=None, yolo_detections=None, 
                                cruise_speed=3.0, obstacle_threshold=3.0):
        """Generate velocity commands with adaptive obstacle avoidance"""
        current_pos = self.sensor_processor.current_position
        
        # Calculate base velocity vector toward target
        direction = np.array([
            target_position[0] - current_pos[0],
            target_position[1] - current_pos[1],
            target_position[2] - current_pos[2]
        ])
        
        distance = np.linalg.norm(direction)
        if distance < 0.1:
            return np.zeros(3)
        
        direction = direction / distance
        target_speed = min(cruise_speed, cruise_speed * min(distance / 5.0, 1.0))
        velocity = direction * target_speed
        
        # Apply obstacle avoidance if obstacles present
        if obstacles or (yolo_detections and len(yolo_detections) > 0):
            repulsive = self._calculate_repulsive_vector(
                current_pos, obstacles, yolo_detections, obstacle_threshold
            )
            
            if np.linalg.norm(repulsive) > 0:
                velocity = self._combine_velocity_vectors(
                    velocity, repulsive, target_speed, obstacle_threshold, yolo_detections
                )
        
        # Forest-specific vertical check
        velocity = self._check_vertical_passage(velocity, cruise_speed, obstacle_threshold)
        
        # Ensure velocity magnitude doesn't exceed max_speed
        speed = np.linalg.norm(velocity)
        if speed > cruise_speed:
            velocity = velocity / speed * cruise_speed
            
        return velocity
    
    def _calculate_repulsive_vector(self, current_pos, obstacles, yolo_detections, obstacle_threshold):
        """Calculate repulsive vector from obstacles"""
        repulsive = np.zeros(3)
        
        # Process YOLO detections
        if yolo_detections:
            for detection in yolo_detections:
                if detection["depth"] > obstacle_threshold * 2:
                    continue
                
                weight = self._calculate_dynamic_obstacle_weight(detection)
                obstacle_pos = self.convert_detection_to_obstacle(detection)
                
                vec_from_obstacle = np.array([
                    current_pos[0] - obstacle_pos[0],
                    current_pos[1] - obstacle_pos[1],
                    current_pos[2] - obstacle_pos[2]
                ])
                
                obstacle_dist = np.linalg.norm(vec_from_obstacle)
                if obstacle_dist > 0.1:
                    repulsive_magnitude = weight / (obstacle_dist * obstacle_dist)
                    repulsive += (vec_from_obstacle / obstacle_dist) * repulsive_magnitude
        
        # Process general obstacles
        if obstacles:
            for obs in obstacles:
                obs_3d = (obs[0], obs[1], current_pos[2])
                vec_from_obstacle = np.array([
                    current_pos[0] - obs_3d[0],
                    current_pos[1] - obs_3d[1],
                    0
                ])
                
                obstacle_dist = np.linalg.norm(vec_from_obstacle)
                if obstacle_dist > 0.1:
                    repulsive_magnitude = 1.0 / (obstacle_dist * obstacle_dist)
                    repulsive += (vec_from_obstacle / obstacle_dist) * repulsive_magnitude
        
        return repulsive
    
    def _combine_velocity_vectors(self, velocity, repulsive, target_speed, obstacle_threshold, yolo_detections):
        """Combine attractive and repulsive velocity vectors"""
        repulsive_magnitude = np.linalg.norm(repulsive)
        repulsive = repulsive / repulsive_magnitude
        
        # Find closest obstacle distance
        closest_obstacle_distance = float('inf')
        if yolo_detections:
            for detection in yolo_detections:
                closest_obstacle_distance = min(closest_obstacle_distance, detection["depth"])
        
        # Adjust repulsive influence
        repulsive_influence = min(1.0, obstacle_threshold / max(closest_obstacle_distance, 0.1))
        repulsive *= target_speed * repulsive_influence
        
        # Combine vectors
        return (1 - repulsive_influence) * velocity + repulsive_influence * repulsive
    
    def _check_vertical_passage(self, velocity, cruise_speed, obstacle_threshold):
        """Check for vertical passages in forest environments"""
        depth_img = self.sensor_processor.get_depth_image()
        if depth_img is not None:
            h, w = depth_img.shape
            top_depth = np.mean(depth_img[:h//3, :])
            front_depth = np.mean(depth_img[h//3:2*h//3, w//3:2*w//3])
            
            if top_depth > obstacle_threshold * 1.5 and front_depth < obstacle_threshold:
                velocity[2] = -0.5 * cruise_speed  # Negative is up in AirSim
        
        return velocity
    
    def _calculate_dynamic_obstacle_weight(self, detection):
        """Calculate weight/importance of an obstacle"""
        weight = detection["confidence"]
        
        # Adjust weight based on object type
        tree_labels = ["tree", "branch", "plant", "foliage", "trunk"]
        if any(label in detection["label"].lower() for label in tree_labels):
            weight *= 1.5
        
        # Adjust by distance
        distance_factor = np.exp(-detection["depth"] / self.obstacle_threshold)
        weight *= distance_factor
        
        # Adjust by size
        bbox = detection["bbox"]
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = 480 * 640
        size_factor = np.sqrt(box_area / image_area)
        weight *= (1 + size_factor)
        
        return weight