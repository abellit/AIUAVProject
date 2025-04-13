import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2


class AirSimColabSimulator:
    """Simulates essential AirSim behaviors for training in Colab without the actual simulator."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.drone_state = {
            'position': np.array([0, 0, 0]),
            'orientation': np.array([0, 0, 0]),
            'velocity': np.array([0, 0, 0]),
            'collision': False
        }
        # Create simulated environment with obstacles
        self.obstacles = self._generate_obstacles()
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        
    def _generate_obstacles(self):
        """Generate some simulated obstacles for navigation training."""
        num_obstacles = 30
        # Create random obstacles in a 100x100 area
        obstacles = []
        for _ in range(num_obstacles):
            pos = np.random.uniform(-50, 50, 3)
            size = np.random.uniform(1, 5, 3)
            obstacles.append({'position': pos, 'size': size})
        return obstacles
        
    def _generate_waypoints(self):
        """Generate a path of waypoints for the drone to follow."""
        # Create a series of waypoints that form a path
        waypoints = []
        for i in range(5):  # 5 waypoints
            # Create waypoints in increasing x distance
            pos = np.array([i * 10, np.sin(i) * 5, -5])  # Keep z at -5 (5m height)
            waypoints.append(pos)
        return waypoints
    
    def getMultirotorState(self):
        """Simulate the AirSim getMultirotorState API."""
        class State:
            def __init__(self, pos, orientation):
                self.kinematics_estimated = type('obj', (object,), {
                    'position': type('obj', (object,), {'x_val': pos[0], 'y_val': pos[1], 'z_val': pos[2]}),
                    'orientation': type('obj', (object,), {'w_val': orientation[0], 'x_val': orientation[1], 
                                                          'y_val': orientation[2], 'z_val': orientation[3]})
                })
        return State(self.drone_state['position'], [1, 0, 0, 0])  # Default quaternion
    
    def simGetImages(self, camera_configs):
        """Simulate getting images from cameras."""
        # Create synthetic images based on state
        images = []
        for config in camera_configs:
            if config.image_type == 0:  # RGB
                # Create a synthetic RGB image based on environment state
                img = np.ones((84, 84, 3), dtype=np.uint8) * 128  # Gray background
                
                # Draw obstacles as dark patches if they're in view
                for obstacle in self.obstacles:
                    # Calculate if obstacle is in field of view and how far
                    rel_pos = obstacle['position'] - self.drone_state['position']
                    distance = np.linalg.norm(rel_pos)
                    if distance < 20:  # Only show nearby obstacles
                        # Calculate position in image (simplified projection)
                        img_x = int(42 + rel_pos[1] * 2)  # y axis maps to horizontal
                        img_y = int(42 - rel_pos[2] * 2)  # z axis maps to vertical
                        # Only draw if in image bounds
                        if 0 <= img_x < 84 and 0 <= img_y < 84:
                            # Draw a rectangle
                            size = int(10 - distance / 4)  # Bigger if closer
                            x1, y1 = max(0, img_x - size), max(0, img_y - size)
                            x2, y2 = min(83, img_x + size), min(83, img_y + size)
                            img[y1:y2, x1:x2] = [50, 50, 50]  # Dark gray
                
                # Draw current waypoint as a bright spot
                if self.current_waypoint_idx < len(self.waypoints):
                    waypoint = self.waypoints[self.current_waypoint_idx]
                    rel_pos = waypoint - self.drone_state['position']
                    distance = np.linalg.norm(rel_pos)
                    if distance < 30:  # Only show if not too far
                        img_x = int(42 + rel_pos[1] * 2)
                        img_y = int(42 - rel_pos[2] * 2)
                        if 0 <= img_x < 84 and 0 <= img_y < 84:
                            # Draw a bright marker
                            size = 3
                            x1, y1 = max(0, img_x - size), max(0, img_y - size)
                            x2, y2 = min(83, img_x + size), min(83, img_y + size)
                            img[y1:y2, x1:x2] = [0, 255, 0]  # Green
                
                # Convert to proper response format
                image_response = type('obj', (object,), {
                    'image_data_uint8': img.tobytes(),
                    'camera_name': config.camera_name,
                    'pixel_as_float': False,
                    'compress': False,
                    'width': 84,
                    'height': 84
                })
                images.append(image_response)
            
            elif config.image_type == 1:  # Depth
                # Create a synthetic depth image
                depth = np.ones((84, 84), dtype=np.float32) * 100.0  # Default far distance
                
                # Add depth values for obstacles
                for obstacle in self.obstacles:
                    rel_pos = obstacle['position'] - self.drone_state['position']
                    distance = np.linalg.norm(rel_pos)
                    if distance < 20:
                        img_x = int(42 + rel_pos[1] * 2)
                        img_y = int(42 - rel_pos[2] * 2)
                        if 0 <= img_x < 84 and 0 <= img_y < 84:
                            size = int(10 - distance / 4)
                            x1, y1 = max(0, img_x - size), max(0, img_y - size)
                            x2, y2 = min(83, img_x + size), min(83, img_y + size)
                            depth[y1:y2, x1:x2] = distance
                
                # Convert to bytes for proper format
                float_array = depth.astype(np.float32)
                image_response = type('obj', (object,), {
                    'image_data_float': float_array.tobytes(),
                    'camera_name': config.camera_name,
                    'pixel_as_float': True,
                    'compress': False,
                    'width': 84,
                    'height': 84
                })
                images.append(image_response)
        
        return images
    
    def getLidarData(self, lidar_name=None):
        """Simulate LiDAR data based on obstacles in environment."""
        # Generate point cloud based on obstacles
        point_cloud = []
        
        # For each obstacle, generate points
        for obstacle in self.obstacles:
            rel_pos = obstacle['position'] - self.drone_state['position']
            distance = np.linalg.norm(rel_pos)
            if distance < 50:  # LiDAR range
                # Generate points along the surface of the obstacle
                # This is simplified - real LiDAR would be more complex
                num_points = max(1, int(20 / distance))  # More points for closer objects
                for _ in range(num_points):
                    # Random point on obstacle surface
                    offset = np.random.uniform(-1, 1, 3) * obstacle['size'] / 2
                    point = obstacle['position'] + offset - self.drone_state['position']
                    # Only add if in front of the drone
                    if point[0] > 0:
                        point_cloud.append(point)
        
        # Create response format
        lidar_data = type('obj', (object,), {
            'point_cloud': point_cloud,
            'time_stamp': 0
        })
        return lidar_data
    
    def getImuData(self, imu_name=None):
        """Simulate IMU data."""
        # Create response format
        imu_data = type('obj', (object,), {
            'time_stamp': 0,
            'angular_velocity': type('obj', (object,), {
                'x_val': 0.01, 'y_val': 0.01, 'z_val': 0.01
            }),
            'linear_acceleration': type('obj', (object,), {
                'x_val': 0.0, 'y_val': 0.0, 'z_val': 9.8  # Gravity
            })
        })
        return imu_data
    
    def simGetCollisionInfo(self):
        """Check if drone has collided with any obstacles."""
        # Check distance to all obstacles
        for obstacle in self.obstacles:
            rel_pos = obstacle['position'] - self.drone_state['position']
            distance = np.linalg.norm(rel_pos)
            if distance < np.mean(obstacle['size']) * 1.2:  # Some buffer for collision
                self.drone_state['collision'] = True
                collision_info = type('obj', (object,), {
                    'has_collided': True,
                    'impact_point': type('obj', (object,), {
                        'x_val': self.drone_state['position'][0],
                        'y_val': self.drone_state['position'][1],
                        'z_val': self.drone_state['position'][2]
                    }),
                    'object_name': 'obstacle',
                    'time_stamp': 0
                })
                return collision_info
        
        # No collision
        collision_info = type('obj', (object,), {
            'has_collided': False
        })
        return collision_info
    
    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain=None, yaw_mode=None):
        """Move drone by velocity - updates internal state."""
        # Update position based on velocity and duration
        self.drone_state['velocity'] = np.array([vx, vy, vz])
        self.drone_state['position'] += self.drone_state['velocity'] * duration
        
        # Check if we've reached the current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(waypoint - self.drone_state['position'])
            if distance < 3.0:  # Waypoint reached
                self.current_waypoint_idx += 1
                
        # Check collisions after moving
        self.simGetCollisionInfo()
        
        # Return a dummy future
        return type('obj', (object,), {'join': lambda: None})
    
    def moveByRollPitchYawZAsync(self, roll, pitch, yaw, z, duration):
        """Simulate the moveByRollPitchYawZAsync method in real AirSim."""
        # Convert roll/pitch/yaw control inputs to velocity changes
        # This is a simplification, but should work for training
        vx = pitch * 5.0  # Forward/backward based on pitch
        vy = roll * -5.0  # Left/right based on roll
        vz = (z - self.drone_state['position'][2]) * 2.0  # Up/down to target z
        
        # Update position based on velocity and duration
        self.drone_state['velocity'] = np.array([vx, vy, vz])
        self.drone_state['position'] += self.drone_state['velocity'] * duration
        
        # Constrain height to be negative (below ground level is positive in AirSim)
        if self.drone_state['position'][2] > 0:
            self.drone_state['position'][2] = 0
        
        # Update orientation
        self.drone_state['orientation'] = np.array([roll, pitch, yaw])
        
        # Check for waypoint reaching and collisions
        self._check_waypoint_reached()
        self._check_collision()
        
        # Return a dummy future object
        return type('obj', (object,), {'join': lambda: None})
    
    def reset(self):
        """Reset the simulator state."""
        self.drone_state = {
            'position': np.array([0, 0, 0]),
            'orientation': np.array([0, 0, 0]),
            'velocity': np.array([0, 0, 0]),
            'collision': False
        }
        # Generate new environment
        self.obstacles = self._generate_obstacles()
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
    
    def step(self, action):

        # Move drone based on action
        self.moveByRollPitchYawZAsync(*action, 0.0, 0.1)
        
        # Get state
        state = self.getMultirotorState()
        
        # Get reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._check_done()
        
        # Get info
        info = {}
        
        return state, reward, done, info
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        reward = 0
        # Negative reward for collisions
        if self.drone_state['collision']:
            reward -= 100
        # Positive reward for reaching waypoints
        reward += self.current_waypoint_idx * 10
        return reward
    
    def _check_done(self):
        """Check if episode is done."""
        # Done if collision occurred or all waypoints reached
        return self.drone_state['collision'] or self.current_waypoint_idx >= len(self.waypoints)
    
    
    def confirmConnection(self):
        """Simulate connection confirmation."""
        return True
    
    def enableApiControl(self, enable):
        """Simulate API control."""
        return True
    
    def armDisarm(self, arm):
        """Simulate arming/disarming."""
        return True