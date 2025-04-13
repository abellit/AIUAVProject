import numpy as np
import time
import random
import cv2
from path_planning import PathPlanner
from object_detection.obstacle_detection import ObstacleDetector
from object_detection.sensor_processing import SensorProcessor
from object_detection.lidar_preprocessor import LidarProcessor
from object_detection.sensor_fusion import SensorSynchronizer
from ultralytics import YOLO
import airsim

class EnhancedDroneNavigationSystem:
    def __init__(self):
        # Initialize subsystems
        self._initialize_airsim_connection()
        self.sensor_processor = SensorProcessor(self.client)
        self.obstacle_detector = ObstacleDetector(self.client)
        self.path_planner = PathPlanner()
        
        # Navigation parameters
        self.obstacle_threshold = 3.0
        self.safe_height = -5.0
        self.cruise_speed = 3.0
        self.goal_reached_threshold = 2.0
        self.replanning_distance = 5.0
        
        # Initialize state
        self.update_state()
        
    def _initialize_airsim_connection(self):
        """Initialize connection to AirSim"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
    
    def update_state(self):
        """Update drone state from various sensors"""
        self.sensor_processor.update_state()
        self.current_position = self.sensor_processor.current_position
        self.current_orientation = self.sensor_processor.current_orientation
        self.current_velocity = self.sensor_processor.current_velocity
        self.imu_data = self.sensor_processor.imu_data
    
    def navigate_in_forest(self, goal_position):
        """Specialized navigation function for dense forest environments"""
        goal = np.array([goal_position[0], goal_position[1], self.safe_height])
        print(f"Starting forest navigation to goal: {goal}")

        # Initialize state
        self.update_state()
        start_time = time.time()
        stalled_count = 0
        last_position = self.current_position.copy()
        
        # Adjust parameters for forest navigation
        self.obstacle_threshold = 4.0
        self.replanning_distance = 3.0
        self.cruise_speed = 2.5
        
        # Initial path planning
        obstacles = self.obstacle_detector.detect_obstacles()
        path = self.path_planner.plan_path(self.current_position, goal, obstacles)
        if path:
            path = self.path_planner.smooth_path(path)
            print(f"Initial path planned with {len(path)} waypoints")

        # Main navigation loop
        while np.linalg.norm(self.current_position[:2] - goal[:2]) > self.goal_reached_threshold:
            loop_start = time.time()
            self.update_state()
            
            # Progress check
            distance_moved = np.linalg.norm(self.current_position - last_position)
            if distance_moved < 1.0:  # Less than 1m in 5 seconds
                stalled_count += 1
                if stalled_count >= 3:
                    self._execute_forest_maneuvers()
                    stalled_count = 0
            else:
                stalled_count = 0
                
            last_position = self.current_position.copy()
            
            # Detect obstacles
            yolo_detections = self.obstacle_detector.run_yolo_detection()
            obstacles = self.obstacle_detector.detect_obstacles()
            
            # Navigation logic
            if path:
                self._follow_path(path, obstacles, yolo_detections)
            else:
                self._reactive_navigation(goal, obstacles, yolo_detections)
            
            # Maintain loop rate
            loop_duration = time.time() - loop_start
            if loop_duration < 0.1:
                time.sleep(0.1 - loop_duration)

        print("Goal reached!")
        self.client.hoverAsync().join()
        return True
    
    def _execute_forest_maneuvers(self):
        """Execute special maneuvers for forest navigation"""
        print("Attempting vertical exploration...")
        self.move_drone("up", 2.0, 2.0)
        self.update_state()
        
        # Try moving in a random direction to break free
        directions = ["forward", "left", "right"]
        random_dir = random.choice(directions)
        print(f"Moving randomly: {random_dir}")
        self.move_drone(random_dir, 2.5, 2.5)
    
    def _follow_path(self, path, obstacles, yolo_detections):
        """Follow planned path with obstacle avoidance"""
        if len(path) > 0:
            next_waypoint = path[0] if len(path) == 1 else path[1]
            velocity = self.obstacle_detector.adaptive_velocity_control(
                next_waypoint, obstacles, yolo_detections, 
                self.cruise_speed, self.obstacle_threshold
            )
            
            try:
                print(f"Moving with velocity: {velocity}")
                self.client.moveByVelocityAsync(
                    velocity[0], velocity[1], velocity[2], 0.5
                ).join()
            except Exception as e:
                print(f"Error in velocity control: {e}")
                self.client.hoverAsync()
            
            # Check if waypoint reached
            self.update_state()
            if len(path) > 0 and np.linalg.norm(self.current_position - np.array(path[0])) < 2.0:
                print("Waypoint reached, moving to next")
                path.pop(0)
    
    def _reactive_navigation(self, goal, obstacles, yolo_detections):
        """Navigate reactively toward goal"""
        print("Using reactive navigation toward goal")
        velocity = self.obstacle_detector.adaptive_velocity_control(
            goal, obstacles, yolo_detections,
            self.cruise_speed, self.obstacle_threshold
        )
        
        try:
            self.client.moveByVelocityAsync(
                velocity[0], velocity[1], velocity[2], 0.5
            ).join()
        except Exception as e:
            print(f"Error in velocity control: {e}")
            self.client.hoverAsync()
    
    def move_drone(self, direction, duration=1.0, speed=3.0):
        """Move the drone based on direction"""
        self.sensor_processor.move_drone(direction, duration, speed)
    
    def land_and_disarm(self):
        """Land the drone and disarm"""
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Navigation complete!")
        cv2.destroyAllWindows()