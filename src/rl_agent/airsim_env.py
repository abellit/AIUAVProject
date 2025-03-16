# import os
# import time
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import airsim
# import cv2

# class AirSimForestEnv(gym.Env):
#     """
#     Gymnasium environment for Search and Rescue (SAR) drone navigation 
#     in dense forest environments using AirSim.
#     """
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

#     def __init__(self, ip_address='', config=None, client=None):
#         super().__init__()
#         self.config = config or {}

#         # Connect to AirSim
#         if client:
#             self.client = client
#         else:
#             ip_address = self.config.get('airsim_ip', '')
#             self.client = airsim.MultirotorClient(ip=ip_address) if ip_address else airsim.MultirotorClient()

#         self.client.confirmConnection()
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)
        
#         # Enhanced simulation parameters
#         self.max_steps = self.config.get('max_steps', 1000)
#         self.current_step = 0
        
#         # Image capture setup - using multiple cameras for better perception
#         self.front_camera = self.config.get('front_camera', "0")
#         self.depth_camera = self.config.get('depth_camera', "0")
        
#         # Define action space: Roll, Pitch, Yaw Rate, Throttle - continuous
#         self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

#         # Define observation space - RGB image with 3 channels
#         # Increased resolution for better feature detection
#         self.image_height = 84  # Increased from 64
#         self.image_width = 84   # Increased from 64
#         self.observation_space = spaces.Box(
#             low=0, high=255, 
#             shape=(3, self.image_height, self.image_width),
#             dtype=np.uint8
#         )

#         self.observation_space.is_image = True

#         # Target waypoints for navigation - these would be generated based on the mission
#         self.waypoints = self.config.get('waypoints', [
#             (7, 0, -3),    # Starting area
#             (7, 23, -3),   # Through first section of forest
#             (14, 35, -3),  # Deeper into forest
#             (0, 40, -3),   # Potential search area
#             (0, 0, -3)     # Return to start
#         ])
        
#         # Improved waypoint tracking
#         self.current_waypoint_index = 0
#         self.target_reached_threshold = self.config.get('target_threshold', 3.0)
#         self.previous_distance_to_waypoint = float('inf')
        
#         # Improved reward configuration with better scaling
#         self.collision_penalty = self.config.get('collision_penalty', -100.0)
#         self.waypoint_reward = self.config.get('waypoint_reward', 200.0)  # Increased reward for reaching waypoint
#         self.progress_reward_factor = self.config.get('progress_reward', 1.0)  # Increased for better positive feedback
#         self.energy_penalty_factor = self.config.get('energy_penalty', 0.005)  # Reduced penalty
#         self.staying_alive_reward = 0.1  # Small reward for each step without collision
        
#         # Improved obstacle avoidance with more granular reward structure
#         self.obstacle_distance_thresholds = {
#             'critical': 0.2,   # Very close - high penalty
#             'danger': 0.5,     # Close - moderate penalty
#             'caution': 1.0,    # Approaching - small penalty
#             'safe': 2.0        # Safe distance - small reward
#         }
        
#         # Dynamic reward coefficients that change as training progresses
#         self.current_collision_coef = 1.0
#         self.curriculum_phase = 0  # For progressive difficulty
        
#         # Tracking metrics
#         self.collisions = 0
#         self.waypoints_reached = 0
#         self.min_distance_to_obstacles = float('inf')
#         self.total_reward = 0.0
#         self.success = False
        
#         # For simulation stability
#         self.last_position = self._get_position()
#         self.position_history = []  # Track positions to detect being stuck
        
#         # Initialize obstacle detection
#         self._update_obstacle_distances()
        
#         # Performance metrics
#         self.episode_start_time = time.time()

#     def step(self, action):
#         """
#         Execute action and return new state, reward, terminal flags, and info.
#         """
#         self.current_step += 1
        
#         # Scale actions appropriately for AirSim control
#         roll, pitch, yaw_rate, throttle = action
        
#         try:
#             # Improved drone control with smoother movements
#             self.client.moveByRollPitchYawZAsync(
#                 float(roll) * 0.4,      # Reduced for more stability
#                 float(pitch) * 0.4,     # Reduced for more stability
#                 float(yaw_rate) * 0.4,  # Reduced for more stability
#                 -3 + float(throttle) * 1.0,  # Base altitude with limited throttle adjustment
#                 0.05  # Shorter duration for more responsive control
#             ).join()
            
#             # Allow small time for physics to stabilize
#             time.sleep(0.05)  
#         except Exception as e:
#             print(f"Control error: {e}")
        
#         # Get observation
#         observation = self._get_observation()
        
#         # Update obstacle distances for reward calculation
#         self._update_obstacle_distances()
        
#         # Store current position
#         current_position = self._get_position()
#         self.position_history.append(current_position)
#         if len(self.position_history) > 10:  # Keep only last 10 positions
#             self.position_history.pop(0)
        
#         # Calculate reward
#         reward, reward_info = self._calculate_reward()
#         self.total_reward += reward
        
#         # Check termination conditions
#         terminated = self._is_terminated()
#         truncated = self._is_truncated()
        
#         # Determine success (for metrics)
#         if self.current_waypoint_index >= len(self.waypoints) and not terminated:
#             self.success = True
        
#         # Prepare info dict with useful metrics
#         info = {
#             'waypoint_index': self.current_waypoint_index,
#             'waypoints_total': len(self.waypoints),
#             'waypoints_reached': self.waypoints_reached,
#             'collisions': self.collisions,
#             'min_obstacle_distance': self.min_distance_to_obstacles,
#             'total_reward': self.total_reward,
#             'steps': self.current_step,
#             'reward_breakdown': reward_info,
#             'success': self.success
#         }
        
#         # If episode is ending, add episode info
#         if terminated or truncated:
#             info['episode'] = {
#                 'r': self.total_reward,
#                 'l': self.current_step,
#                 'success': self.success
#             }

#         return observation, reward, terminated, truncated, info

#     def reset(self, seed=None, options=None):
#         """
#         Reset environment for new episode with enhanced randomization.
#         """
#         super().reset(seed=seed, options=options)
        
#         # Reset simulation
#         self.client.reset()
#         for attempt in range(5):
#             try:
#                 self.client.confirmConnection()
#                 self.client.enableApiControl(True)
#                 self.client.armDisarm(True)
#                 print(f"AirSim reconnected successfully on attempt {attempt + 1}")
#                 break  # Exit loop if successful
#             except Exception as e:
#                 print(f"AirSim reconnection attempt {attempt + 1} failed: {e}")
#                 time.sleep(2)  # Wait before retrying
#         else:
#             raise RuntimeError("AirSim failed to reconnect after 5 attempts.")
        
#         # Reset internal state
#         self.current_step = 0
#         self.current_waypoint_index = 0
#         self.collisions = 0
#         self.waypoints_reached = 0
#         self.min_distance_to_obstacles = float('inf')
#         self.total_reward = 0.0
#         self.success = False
#         self.position_history = []
#         self.previous_distance_to_waypoint = float('inf')
#         self.episode_start_time = time.time()
        
#         # Enhanced initial position randomization for better generalization
#         initial_position = self.config.get('initial_position', [0, 0, -3])
        
#         # Progressively increase randomization as training advances
#         randomization_scale = min(1.0, 0.2 + self.curriculum_phase * 0.2)  # Start with 0.2, gradually increase
        
#         random_offset = np.random.uniform(-randomization_scale, randomization_scale, 3) if self.config.get('randomize_start', True) else np.zeros(3)
        
#         initial_position = np.array(initial_position) + random_offset
#         initial_pose = airsim.Pose(
#             airsim.Vector3r(initial_position[0], initial_position[1], initial_position[2]),
#             airsim.Quaternionr()  # Default orientation
#         )
#         self.client.simSetVehiclePose(initial_pose, True)
        
#         # Randomize waypoints slightly for better generalization
#         if self.curriculum_phase > 1:  # Only after some initial training
#             self.waypoints = [
#                 (wp[0] + np.random.uniform(-0.5, 0.5), 
#                  wp[1] + np.random.uniform(-0.5, 0.5), 
#                  wp[2] + np.random.uniform(-0.2, 0.2))
#                 for wp in self.config.get('waypoints', self.waypoints)
#             ]
        
#         # Wait for physics to stabilize
#         time.sleep(0.2)
        
#         # Get initial observation
#         observation = self._get_observation()
        
#         # Update obstacle distances
#         self._update_obstacle_distances()
        
#         # Reset position tracking
#         self.last_position = self._get_position()
        
#         # Calculate initial distance to waypoint
#         if self.waypoints:
#             current_waypoint = np.array(self.waypoints[0])
#             self.previous_distance_to_waypoint = np.linalg.norm(self.last_position - current_waypoint)
        
#         info = {
#             'reset': True,
#             'initial_position': initial_position.tolist(),
#             'curriculum_phase': self.curriculum_phase
#         }
        
#         return observation, info

#     def _get_observation(self):
#         """
#         Get RGB camera image with better pre-processing for neural network input.
#         """
#         try:
#             # Request RGB image
#             responses = self.client.simGetImages([
#                 airsim.ImageRequest(self.front_camera, airsim.ImageType.Scene, False, False)
#             ])
            
#             if not responses:
#                 raise Exception("No image returned from AirSim")
                
#             response = responses[0]
            
#             # Convert to numpy array
#             img_rgba = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
#             img_rgba = img_rgba.reshape(response.height, response.width, 3)  # RGB format
            
#             # Enhanced image processing:
#             # 1. Resize to the dimensions expected by the CNN
#             img_resized = cv2.resize(img_rgba, (self.image_width, self.image_height))
            
#             # 2. Apply simple image enhancement to improve feature detection
#             img_enhanced = cv2.convertScaleAbs(img_resized, alpha=1.1, beta=5)  # Slight contrast enhancement
            
#             # Convert to channel-first format (C, H, W) as expected by PyTorch
#             img_channels_first = np.transpose(img_enhanced, (2, 0, 1))
            
#             return img_channels_first.astype(np.uint8)
                
#         except Exception as e:
#             print(f"Error getting observation: {e}")
#             # Return a blank observation as fallback
#             return np.zeros((3, self.image_height, self.image_width), dtype=np.uint8)

#     def _update_obstacle_distances(self):
#         """
#         Update distances to obstacles using depth image with improved filtering.
#         """
#         try:
#             # Get depth image
#             responses = self.client.simGetImages([
#                 airsim.ImageRequest(self.depth_camera, airsim.ImageType.DepthPerspective, True, False)
#             ])
            
#             if responses:
#                 depth_response = responses[0]
#                 depth_img = airsim.list_to_2d_float_array(
#                     depth_response.image_data_float, 
#                     depth_response.width, 
#                     depth_response.height
#                 )
                
#                 # Improved depth image processing
#                 depth_img = np.array(depth_img)
                
#                 # Focus on obstacles in the forward direction (center of image)
#                 h, w = depth_img.shape
#                 center_region = depth_img[h//4:3*h//4, w//4:3*w//4]
                
#                 # Filter out very large values (sky, etc.) and very small values (noise)
#                 valid_depths = center_region[(center_region < 50) & (center_region > 0.1)]
                
#                 if valid_depths.size > 0:
#                     # Calculate different percentiles for more robust obstacle detection
#                     min_depth = np.min(valid_depths)  # Closest obstacle
#                     percentile_10 = np.percentile(valid_depths, 10)  # 10th percentile for robustness
                    
#                     # Use the 10th percentile as the distance to obstacles (more robust to noise)
#                     self.min_distance_to_obstacles = min(
#                         self.min_distance_to_obstacles, 
#                         percentile_10
#                     )
                    
#                     # Also store the closest obstacle for critical avoidance
#                     self.closest_obstacle = min_depth
#         except Exception as e:
#             print(f"Error updating obstacle distances: {e}")

#     def _get_position(self):
#         """
#         Get current drone position as numpy array.
#         """
#         state = self.client.getMultirotorState()
#         pos = state.kinematics_estimated.position
#         return np.array([pos.x_val, pos.y_val, pos.z_val])


#     def _calculate_reward(self):
#         """
#         Improved reward function with better balancing between waypoint progress and obstacle avoidance.
#         """
#         reward = 0.0
#         reward_info = {}
        
#         # Get current position
#         current_position = self._get_position()
        
#         # Base reward: small positive reward for staying alive
#         alive_reward = self.staying_alive_reward
#         reward += alive_reward
#         reward_info['alive'] = alive_reward
        
#         # Check for collisions - keep this strong negative reward
#         collision_info = self.client.simGetCollisionInfo()
#         if collision_info.has_collided:
#             adjusted_collision_penalty = self.collision_penalty * self.current_collision_coef
#             reward += adjusted_collision_penalty
#             reward_info['collision'] = adjusted_collision_penalty
#             self.collisions += 1
            
#             # Gradually increase collision penalty for curriculum learning
#             if self.curriculum_phase >= 2:
#                 self.current_collision_coef = min(2.0, self.current_collision_coef * 1.01)
        
#         # Calculate distance to current waypoint with improved waypoint attraction
#         if self.current_waypoint_index < len(self.waypoints):
#             current_waypoint = np.array(self.waypoints[self.current_waypoint_index])
#             distance_to_waypoint = np.linalg.norm(current_position - current_waypoint)
            
#             # Check if waypoint reached - keep strong positive reward
#             if distance_to_waypoint < self.target_reached_threshold:
#                 progression_factor = 1.0 + 0.2 * self.current_waypoint_index
#                 waypoint_reward = self.waypoint_reward * progression_factor
                
#                 reward += waypoint_reward
#                 reward_info['waypoint_reached'] = waypoint_reward
#                 self.waypoints_reached += 1
#                 self.current_waypoint_index += 1
                
#                 # Reset for next waypoint
#                 if self.current_waypoint_index < len(self.waypoints):
#                     next_waypoint = np.array(self.waypoints[self.current_waypoint_index])
#                     self.previous_distance_to_waypoint = np.linalg.norm(current_position - next_waypoint)
            
#             else:
#                 # INCREASED: Progress reward coefficient for stronger waypoint attraction
#                 # Calculate progress toward waypoint
#                 progress = self.previous_distance_to_waypoint - distance_to_waypoint
                
#                 # Update previous distance for next step
#                 self.previous_distance_to_waypoint = distance_to_waypoint
                
#                 # STRONGER WAYPOINT ATTRACTION: Amplify progress rewards significantly
#                 if progress > 0:
#                     # Positive progress toward waypoint - INCREASED FACTOR
#                     progress_reward = progress * (self.progress_reward_factor * 3.0)  # Tripled from original
#                     # Extra bonus for consistent progress
#                     if self.current_step > 1 and progress > 0.1:
#                         progress_reward *= 1.5  # Increased bonus
#                 else:
#                     # Negative progress (moving away from waypoint)
#                     # More severe penalty to discourage moving away from waypoint
#                     progress_reward = progress * (self.progress_reward_factor * 2.0)  # Doubled penalty
                
#                 reward += progress_reward
#                 reward_info['progress'] = progress_reward
                
#                 # STRONGER WAYPOINT ATTRACTION: Add stronger distance-based component
#                 # Exponential reward gradient that gets stronger as drone gets closer to waypoint
#                 # This creates a "gravity well" effect toward the waypoint
#                 distance_factor = max(1.0, 10.0 - distance_to_waypoint) / 10.0  # Scales from 0.1 to 1.0
#                 distance_component = -0.05 * distance_to_waypoint * distance_factor  # 5x stronger
#                 reward += distance_component
#                 reward_info['distance'] = distance_component
                
#                 # DIRECTIONAL REWARD: Add reward for facing toward the waypoint
#                 drone_orientation = self.client.getMultirotorState().kinematics_estimated.orientation
#                 drone_heading = airsim.to_eularian_angles(drone_orientation)[2]  # Yaw in radians
                
#                 # Calculate direction vector to waypoint
#                 direction_to_waypoint = current_waypoint - current_position
#                 target_heading = np.arctan2(direction_to_waypoint[1], direction_to_waypoint[0])
                
#                 # Calculate heading difference (-pi to pi)
#                 heading_diff = np.abs(drone_heading - target_heading)
#                 if heading_diff > np.pi:
#                     heading_diff = 2 * np.pi - heading_diff
                    
#                 # Reward for facing toward waypoint (1.0 when perfectly aligned, 0.0 when opposite)
#                 heading_alignment = 1.0 - (heading_diff / np.pi)
#                 heading_reward = 0.5 * heading_alignment  # New reward component
#                 reward += heading_reward
#                 reward_info['heading'] = heading_reward
        
#         # ADJUSTED: Obstacle avoidance reward - with more balanced thresholds
#         if self.min_distance_to_obstacles < float('inf'):
#             obstacle_reward = 0.0
            
#             # REBALANCED: More balanced obstacle avoidance rewards
#             # Only penalize when very close to obstacles
#             if self.min_distance_to_obstacles < self.obstacle_distance_thresholds['critical']:
#                 # Critical danger - still high penalty
#                 obstacle_reward = -8.0 * (self.obstacle_distance_thresholds['critical'] - self.min_distance_to_obstacles)
#             elif self.min_distance_to_obstacles < self.obstacle_distance_thresholds['danger']:
#                 # Danger zone - moderate penalty but reduced
#                 obstacle_reward = -3.0 * (self.obstacle_distance_thresholds['danger'] - self.min_distance_to_obstacles)
#             elif self.min_distance_to_obstacles < self.obstacle_distance_thresholds['caution']:
#                 # Caution zone - very small penalty to allow closer approach if needed
#                 obstacle_reward = -0.5 * (self.obstacle_distance_thresholds['caution'] - self.min_distance_to_obstacles)
#             else:
#                 # Beyond caution threshold - no reward/penalty to avoid distraction from waypoint goal
#                 obstacle_reward = 0.0
                
#             reward += obstacle_reward
#             reward_info['obstacle_distance'] = obstacle_reward
        
#         # Keep the energy efficiency and stuck detection components
#         velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
#         velocity_mag = np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
        
#         # MODIFIED: Energy efficiency reward - encourage movement toward waypoint
#         if velocity_mag > 10.0:  # Too fast
#             energy_penalty = -(velocity_mag - 10.0)**2 * self.energy_penalty_factor
#         elif velocity_mag < 0.5:  # Too slow or stuck
#             energy_penalty = -(0.5 - velocity_mag)**2 * self.energy_penalty_factor * 2.0
#         else:
#             # Optimal velocity range - small reward
#             energy_penalty = 0.1
            
#         reward += energy_penalty
#         reward_info['energy'] = energy_penalty
        
#         # Detect if drone is stuck (not moving sufficiently)
#         if len(self.position_history) >= 10:
#             positions = np.array(self.position_history)
#             total_movement = np.linalg.norm(positions[-1] - positions[0])
#             if total_movement < 0.5:  # Very little movement over 10 steps
#                 stuck_penalty = -3.0  # Increased penalty for being stuck
#                 reward += stuck_penalty
#                 reward_info['stuck'] = stuck_penalty
        
#         # Update position tracking
#         self.last_position = current_position
        
#         return reward, reward_info

#     def _is_terminated(self):
#         """
#         Enhanced termination conditions
#         """
#         # Check for collision
#         collision_info = self.client.simGetCollisionInfo()
#         if collision_info.has_collided:
#             return True
        
#         # Check if all waypoints have been reached
#         if self.current_waypoint_index >= len(self.waypoints):
#             return True
        
#         # Check if drone is severely out of bounds
#         current_position = self._get_position()
#         max_bounds = 100  # Maximum distance from origin
#         if np.any(np.abs(current_position) > max_bounds):
#             return True
        
#         return False

#     def _is_truncated(self):
#         """
#         Check if episode should be truncated
#         """
#         # Check for maximum steps
#         if self.current_step >= self.max_steps:
#             return True
            
#         # Detect if completely stuck (no movement for extended period)
#         if len(self.position_history) >= 20:
#             positions = np.array(self.position_history)
#             total_movement = np.linalg.norm(positions[-1] - positions[0])
#             if total_movement < 0.2:  # Almost no movement over 20 steps
#                 return True
                
#         # Timeout based on real time (useful for debugging)
#         current_time = time.time()
#         if current_time - self.episode_start_time > 300:  # 5 minute timeout
#             return True
            
#         return False

#     def render(self, mode="human"):
#         """
#         Render the environment.
#         """
#         if mode == "rgb_array":
#             return self._get_observation()
#         return super().render(mode=mode)

#     def close(self):
#         """
#         Clean up resources.
#         """
#         self.client.armDisarm(False)
#         self.client.enableApiControl(False)
        
#     def update_curriculum(self, success_rate):
#         """
#         Update the curriculum phase based on success rate
#         """
#         if success_rate > 0.8 and self.curriculum_phase < 4:
#             self.curriculum_phase += 1
#             print(f"Advancing to curriculum phase {self.curriculum_phase}")
            
#             # Adjust parameters based on curriculum phase
#             if self.curriculum_phase == 1:
#                 # Phase 1: Increase obstacle penalties
#                 self.obstacle_distance_thresholds['safe'] = 4.0
#             elif self.curriculum_phase == 2:
#                 # Phase 2: Increase randomization and reduce waypoint threshold
#                 self.target_reached_threshold = 2.5
#             elif self.curriculum_phase == 3:
#                 # Phase 3: Further increase difficulty
#                 self.target_reached_threshold = 2.0
#             elif self.curriculum_phase == 4:
#                 # Phase 4: Final difficulty
#                 self.target_reached_threshold = 1.5
#                 self.obstacle_distance_thresholds['safe'] = 3.0


import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2



class AirSimForestEnv(gym.Env):
    """
    Gymnasium environment for Search and Rescue (SAR) drone navigation 
    in dense forest environments using AirSim with enhanced sensor suite.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, ip_address='', config=None, client=None):
        super().__init__()
        self.config = config or {}

        # Connect to AirSim
        if client:
            self.client = client
        else:
            ip_address = self.config.get('airsim_ip', '')
            self.client = airsim.MultirotorClient(ip=ip_address) if ip_address else airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Simulation parameters
        self.max_steps = self.config.get('max_steps', 1000)
        self.current_step = 0
        
        # Image and sensor capture setup
        self.front_camera = self.config.get('front_camera', "0")
        self.depth_camera = self.config.get('depth_camera', "0")
        
        # Configure LiDAR sensor
        self.use_lidar = self.config.get('use_lidar', True)
        if self.use_lidar:
            self.lidar_points = self.config.get('lidar_points', 1024)
            self.lidar_range = self.config.get('lidar_range', 50.0)
            # Configure lidar sensing parameters if needed
            # self.client.simAddLidarRange(...)
        
        # Define action space: Roll, Pitch, Yaw Rate, Throttle - continuous
        # Scaled to be between -1 and 1 for easier learning
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space - combining multiple sensor inputs
        self.image_height = 64
        self.image_width = 64
        
        # Calculate total observation space size
        self.use_rgb = self.config.get('use_rgb', True)
        self.use_depth = self.config.get('use_depth', True)
        self.use_gps = self.config.get('use_gps', True)
        self.use_imu = self.config.get('use_imu', True)
        self.use_barometer = self.config.get('use_barometer', True)
        self.use_distance_sensor = self.config.get('use_distance_sensor', True)
        
        # Initialize observation spaces dictionary
        self.observation_components = {}
        sensor_space_list = []
        
        # RGB camera
        if self.use_rgb:
            rgb_space = spaces.Box(
                low=0, high=255, 
                shape=(3, self.image_height, self.image_width),
                dtype=np.uint8
            )
            self.observation_components['rgb'] = rgb_space
            sensor_space_list.append(rgb_space)
        
        # Depth camera
        if self.use_depth:
            depth_space = spaces.Box(
                low=0, high=255,
                shape=(1, self.image_height, self.image_width),
                dtype=np.uint8
            )
            self.observation_components['depth'] = depth_space
            sensor_space_list.append(depth_space)
        
        # LiDAR sensor
        if self.use_lidar:
            # 3D LiDAR returns points in 3D space
            lidar_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.lidar_points, 3),  # x, y, z coordinates for each point
                dtype=np.float32
            )
            self.observation_components['lidar'] = lidar_space
            sensor_space_list.append(lidar_space)
        
        # GPS sensor (x, y, z global position)
        if self.use_gps:
            gps_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),  # x, y, z coordinates
                dtype=np.float32
            )
            self.observation_components['gps'] = gps_space
            sensor_space_list.append(gps_space)
        
        # IMU sensor (orientation quaternion, angular velocity, linear acceleration)
        if self.use_imu:
            imu_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(10,),  # quaternion (4) + angular velocity (3) + linear acceleration (3)
                dtype=np.float32
            )
            self.observation_components['imu'] = imu_space
            sensor_space_list.append(imu_space)
        
        # Barometer (altitude)
        if self.use_barometer:
            barometer_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1,),  # altitude
                dtype=np.float32
            )
            self.observation_components['barometer'] = barometer_space
            sensor_space_list.append(barometer_space)
        
        # Distance sensors (multiple directions)
        if self.use_distance_sensor:
            # Front, back, left, right, up, down distance readings
            distance_space = spaces.Box(
                low=0, high=np.inf,
                shape=(6,),  # 6 directions
                dtype=np.float32
            )
            self.observation_components['distance'] = distance_space
            sensor_space_list.append(distance_space)

        # Create a dictionary observation space if multiple sensors are used
        if len(sensor_space_list) > 1:
            sensor_names = list(self.observation_components.keys())
            self.observation_space = spaces.Dict({
                name: space for name, space in self.observation_components.items()
            })
            self.is_dict_observation = True
        else:
            # If only one sensor is used, use that directly as the observation space
            self.observation_space = sensor_space_list[0]
            self.is_dict_observation = False

        # Target waypoints for navigation
        self.waypoints = self.config.get('waypoints', [
            (7, 0, -3),    # Starting area
            (7, 23, -3),   # Through first section of forest
            (14, 35, -3),  # Deeper into forest
            (0, 40, -3),   # Potential search area
            (0, 0, -3)     # Return to start
        ])
        self.current_waypoint_index = 0
        self.target_reached_threshold = self.config.get('target_threshold', 3.0)  # meters
        
        # Rewards configuration
        self.collision_penalty = self.config.get('collision_penalty', -100.0)
        self.waypoint_reward = self.config.get('waypoint_reward', 50.0)
        self.progress_reward_factor = self.config.get('progress_reward', 0.1)
        self.energy_penalty_factor = self.config.get('energy_penalty', 0.01)
        self.obstacle_distance_factor = self.config.get('obstacle_distance_factor', 0.5)
        
        # Tracking metrics
        self.collisions = 0
        self.waypoints_reached = 0
        self.min_distance_to_obstacles = float('inf')
        self.total_reward = 0.0
        
        # For simulation stability
        self.last_position = np.zeros(3)
        self.last_orientation = np.zeros(4)  # Quaternion
        
        # Initialize path planning variables
        self.path_history = []
        self.planned_path = []

    def step(self, action):
        """
        Execute action and return new state, reward, terminal flags, and info.
        """
        self.current_step += 1
        
        # Scale actions appropriately for AirSim control
        roll, pitch, yaw_rate, throttle = action
        
        try:
            # Move drone based on action
            self.client.moveByRollPitchYawZAsync(
                float(roll) * 0.4,    # Scale down for stability
                float(pitch) * 0.4,   # Scale down for stability
                float(yaw_rate) * 0.4,  
                -3 + float(throttle) * 1.7,  # Base altitude of -3m with throttle adjustment
                0.1  # Duration - keep short for responsive control
            ).join()
            
            # Allow small time for physics to stabilize
            time.sleep(0.05)  
        except Exception as e:
            print(f"Control error: {e}")
        
        # Get observation from all sensors
        observation = self._get_observation()
        
        # Update obstacle distances for reward calculation
        self._update_obstacle_distances()
        
        # Record current position for path history
        current_position = self._get_position()
        self.path_history.append(current_position)
        
        # Calculate reward
        reward, reward_info = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Prepare info dict with useful metrics
        info = {
            'waypoint_index': self.current_waypoint_index,
            'waypoints_total': len(self.waypoints),
            'waypoints_reached': self.waypoints_reached,
            'collisions': self.collisions,
            'min_obstacle_distance': self.min_distance_to_obstacles,
            'total_reward': self.total_reward,
            'steps': self.current_step,
            'reward_breakdown': reward_info,
            'current_position': current_position.tolist(),
            'current_waypoint': self.waypoints[min(self.current_waypoint_index, len(self.waypoints)-1)]
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode.
        """
        super().reset(seed=seed, options=options)
        
        # Reset simulation
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Reset internal state
        self.current_step = 0
        self.current_waypoint_index = 0
        self.collisions = 0
        self.waypoints_reached = 0
        self.min_distance_to_obstacles = float('inf')
        self.total_reward = 0.0
        self.path_history = []
        self.planned_path = []
        
        # Set initial pose - slight randomization for robustness
        initial_position = self.config.get('initial_position', [0, 0, -3])
        random_offset = np.random.uniform(-0.5, 0.5, 3) if self.config.get('randomize_start', True) else np.zeros(3)
        
        initial_position = np.array(initial_position) + random_offset
        initial_pose = airsim.Pose(
            airsim.Vector3r(initial_position[0], initial_position[1], initial_position[2]),
            airsim.Quaternionr()  # Default orientation
        )
        self.client.simSetVehiclePose(initial_pose, True)
        
        # Wait for physics to stabilize
        time.sleep(0.2)
        
        # Get initial observation from all sensors
        observation = self._get_observation()
        
        # Update obstacle distances
        self._update_obstacle_distances()
        
        # Reset position tracking
        self.last_position = self._get_position()
        self.last_orientation = self._get_orientation()
        
        # Add initial position to path history
        self.path_history.append(self.last_position)
        
        # Generate initial path plan if using path planning
        if self.config.get('use_path_planning', False):
            self._plan_path()
        
        info = {
            'reset': True,
            'initial_position': initial_position.tolist()
        }
        
        return observation, info

    def _get_observation(self):
        """
        Get combined observations from all enabled sensors.
        """
        observations = {}
        
        try:
            # RGB Camera
            if self.use_rgb:
                observations['rgb'] = self._get_rgb_observation()
            
            # Depth Camera
            if self.use_depth:
                observations['depth'] = self._get_depth_observation()
            
            # LiDAR
            if self.use_lidar:
                observations['lidar'] = self._get_lidar_observation()
            
            # GPS
            if self.use_gps:
                observations['gps'] = self._get_gps_observation()
            
            # IMU
            if self.use_imu:
                observations['imu'] = self._get_imu_observation()
            
            # Barometer
            if self.use_barometer:
                observations['barometer'] = self._get_barometer_observation()
            
            # Distance Sensors
            if self.use_distance_sensor:
                observations['distance'] = self._get_distance_observation()
            
            # Return dictionary if using multiple sensors
            if self.is_dict_observation:
                return observations
            else:
                # Return the single observation directly
                return list(observations.values())[0]
                
        except Exception as e:
            print(f"Error getting observation: {e}")
            
            # Return empty observations as fallback
            if self.is_dict_observation:
                return {name: np.zeros(space.shape, dtype=space.dtype) 
                        for name, space in self.observation_components.items()}
            else:
                return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def _get_rgb_observation(self):
        """
        Get RGB camera image and process it for the neural network.
        """
        try:
            # Request RGB image
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.front_camera, airsim.ImageType.Scene, False, False)
            ])
            
            if not responses:
                raise Exception("No RGB image returned from AirSim")
                
            response = responses[0]
            
            # Convert to numpy array
            img_rgba = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgba = img_rgba.reshape(response.height, response.width, 3)  # RGB format
            
            # Resize to the dimensions expected by the CNN
            img_resized = cv2.resize(img_rgba, (self.image_width, self.image_height))
            
            # Return in channel-first format (C, H, W) as expected by PyTorch
            img_channels_first = np.transpose(img_resized, (2, 0, 1))
            
            return img_channels_first.astype(np.uint8)
                
        except Exception as e:
            print(f"Error getting RGB observation: {e}")
            # Return a blank observation as fallback
            return np.zeros((3, self.image_height, self.image_width), dtype=np.uint8)

    def _get_depth_observation(self):
        """
        Get depth camera image and process it.
        """
        try:
            # Request depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.depth_camera, airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if not responses:
                raise Exception("No depth image returned from AirSim")
                
            response = responses[0]
            
            # Convert to depth map
            depth_img = airsim.list_to_2d_float_array(
                response.image_data_float, 
                response.width, 
                response.height
            )
            
            # Normalize and convert to uint8 for network input
            depth_img = np.array(depth_img)
            depth_img = np.clip(depth_img, 0, 100.0)  # Clip far distances 
            depth_img = (depth_img * 255.0 / 100.0).astype(np.uint8)  # Scale to 0-255
            
            # Resize to the dimensions expected by the CNN
            depth_resized = cv2.resize(depth_img, (self.image_width, self.image_height))
            
            # Add channel dimension and return in channel-first format
            depth_channels_first = depth_resized.reshape(1, self.image_height, self.image_width)
            
            return depth_channels_first.astype(np.uint8)
                
        except Exception as e:
            print(f"Error getting depth observation: {e}")
            # Return a blank observation as fallback
            return np.zeros((1, self.image_height, self.image_width), dtype=np.uint8)

    def _get_lidar_observation(self):
        """
        Get LiDAR point cloud data.
        """
        try:
            # Get LiDAR data
            lidar_data = self.client.getLidarData()
            
            if not lidar_data:
                raise Exception("No LiDAR data returned from AirSim")
            
            # Extract point cloud
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            
            # Handle variable number of points - pad or truncate
            if len(points) > self.lidar_points:
                # Randomly sample points to match desired count
                indices = np.random.choice(len(points), self.lidar_points, replace=False)
                points = points[indices]
            elif len(points) < self.lidar_points:
                # Pad with zeros
                padding = np.zeros((self.lidar_points - len(points), 3), dtype=np.float32)
                points = np.vstack((points, padding))
            
            return points
                
        except Exception as e:
            print(f"Error getting LiDAR observation: {e}")
            # Return empty point cloud as fallback
            return np.zeros((self.lidar_points, 3), dtype=np.float32)

    def _get_gps_observation(self):
        """
        Get GPS coordinates (position in global frame).
        """
        try:
            # Get GPS data
            gps_data = self.client.getGpsData(gps_name="Gps", vehicle_name="")
            
            # Extract position
            position = np.array([
                gps_data.gnss.geo_point.latitude,
                gps_data.gnss.geo_point.longitude,
                gps_data.gnss.geo_point.altitude
            ], dtype=np.float32)
            
            return position
                
        except Exception as e:
            # AirSim may not have GPS sensor implemented, use position instead
            try:
                position = self._get_position()
                return position.astype(np.float32)
            except Exception as e2:
                print(f"Error getting GPS observation: {e2}")
                # Return zeros as fallback
                return np.zeros(3, dtype=np.float32)

    def _get_imu_observation(self):
        """
        Get IMU data (orientation, angular velocity, linear acceleration).
        """
        try:
            # Get IMU data
            imu_data = self.client.getImuData(imu_name="Imu", vehicle_name="")
            
            # Extract orientation (quaternion)
            orientation = np.array([
                imu_data.orientation.x_val,
                imu_data.orientation.y_val,
                imu_data.orientation.z_val,
                imu_data.orientation.w_val
            ], dtype=np.float32)
            
            # Extract angular velocity
            angular_velocity = np.array([
                imu_data.angular_velocity.x_val,
                imu_data.angular_velocity.y_val,
                imu_data.angular_velocity.z_val
            ], dtype=np.float32)
            
            # Extract linear acceleration
            linear_acceleration = np.array([
                imu_data.linear_acceleration.x_val,
                imu_data.linear_acceleration.y_val,
                imu_data.linear_acceleration.z_val
            ], dtype=np.float32)
            
            # Combine all IMU data
            imu_data = np.concatenate((orientation, angular_velocity, linear_acceleration))
            
            return imu_data
                
        except Exception as e:
            # AirSim may not have IMU implemented, get orientation from state
            try:
                # Get state
                state = self.client.getMultirotorState()
                
                # Extract orientation
                orientation = np.array([
                    state.kinematics_estimated.orientation.x_val,
                    state.kinematics_estimated.orientation.y_val,
                    state.kinematics_estimated.orientation.z_val,
                    state.kinematics_estimated.orientation.w_val
                ], dtype=np.float32)
                
                # Extract angular velocity
                angular_velocity = np.array([
                    state.kinematics_estimated.angular_velocity.x_val,
                    state.kinematics_estimated.angular_velocity.y_val,
                    state.kinematics_estimated.angular_velocity.z_val
                ], dtype=np.float32)
                
                # Extract linear acceleration (or substitute with zeros)
                linear_acceleration = np.array([
                    state.kinematics_estimated.linear_acceleration.x_val,
                    state.kinematics_estimated.linear_acceleration.y_val,
                    state.kinematics_estimated.linear_acceleration.z_val
                ], dtype=np.float32)
                
                # Combine all IMU data
                imu_data = np.concatenate((orientation, angular_velocity, linear_acceleration))
                
                return imu_data
            except Exception as e2:
                print(f"Error getting IMU observation: {e2}")
                # Return zeros as fallback
                return np.zeros(10, dtype=np.float32)

    def _get_barometer_observation(self):
        """
        Get barometer data (altitude).
        """
        try:
            # Get barometer data
            barometer_data = self.client.getBarometerData(barometer_name="Barometer", vehicle_name="")
            
            # Extract altitude
            altitude = np.array([barometer_data.altitude], dtype=np.float32)
            
            return altitude
                
        except Exception as e:
            # AirSim may not have barometer implemented, use position z-value instead
            try:
                position = self._get_position()
                altitude = np.array([position[2]], dtype=np.float32)
                return altitude
            except Exception as e2:
                print(f"Error getting barometer observation: {e2}")
                # Return zeros as fallback
                return np.zeros(1, dtype=np.float32)

    def _get_distance_observation(self):
        """
        Get distance sensor readings in multiple directions.
        """
        try:
            # Simulate distance sensors in six directions using ray casting
            directions = [
                (1, 0, 0),   # Forward
                (-1, 0, 0),  # Backward
                (0, 1, 0),   # Right
                (0, -1, 0),  # Left
                (0, 0, 1),   # Up
                (0, 0, -1)   # Down
            ]
            
            distances = []
            max_distance = 100.0  # Maximum distance to check
            
            for direction in directions:
                # Get drone position
                drone_pose = self.client.simGetVehiclePose()
                drone_position = np.array([
                    drone_pose.position.x_val, 
                    drone_pose.position.y_val, 
                    drone_pose.position.z_val
                ])
                
                # Calculate ray direction in world frame
                # Note: This is simplified, ideally we'd apply the drone's rotation to the direction
                ray_direction = np.array(direction)
                
                # Cast ray
                hit_info = self.client.simCastRay(
                    airsim.Vector3r(drone_position[0], drone_position[1], drone_position[2]),
                    airsim.Vector3r(
                        drone_position[0] + ray_direction[0] * max_distance,
                        drone_position[1] + ray_direction[1] * max_distance,
                        drone_position[2] + ray_direction[2] * max_distance
                    )
                )
                
                # Check if ray hit anything
                if hit_info and hit_info.hit:
                    hit_position = np.array([hit_info.position.x_val, hit_info.position.y_val, hit_info.position.z_val])
                    distance = np.linalg.norm(hit_position - drone_position)
                    distances.append(distance)
                else:
                    distances.append(max_distance)  # No hit, use max distance
            
            return np.array(distances, dtype=np.float32)
                
        except Exception as e:
            # If ray casting fails, try using depth image for distances
            try:
                # Use depth image to estimate distances in different directions
                depth_img = self._get_raw_depth_image()
                
                if depth_img is not None:
                    h, w = depth_img.shape
                    center_h, center_w = h // 2, w // 2
                    
                    # Sample depth at center and edges
                    forward = np.median(depth_img[center_h-5:center_h+5, center_w-5:center_w+5])
                    backward = 100.0  # Can't see backward with front camera
                    right = np.median(depth_img[center_h-5:center_h+5, w-20:w-10])
                    left = np.median(depth_img[center_h-5:center_h+5, 10:20])
                    up = np.median(depth_img[10:20, center_w-5:center_w+5])
                    down = np.median(depth_img[h-20:h-10, center_w-5:center_w+5])
                    
                    distances = np.array([forward, backward, right, left, up, down], dtype=np.float32)
                    # Replace inf or NaN values
                    distances = np.nan_to_num(distances, nan=100.0, posinf=100.0, neginf=0.0)
                    return distances
                else:
                    raise Exception("No depth image available")
            except Exception as e2:
                print(f"Error getting distance observation: {e2}")
                # Return max distances as fallback
                return np.ones(6, dtype=np.float32) * 100.0

    def _get_raw_depth_image(self):
        """
        Get raw depth image for processing.
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.depth_camera, airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if responses:
                depth_response = responses[0]
                depth_img = airsim.list_to_2d_float_array(
                    depth_response.image_data_float, 
                    depth_response.width, 
                    depth_response.height
                )
                return np.array(depth_img)
            return None
        except Exception as e:
            print(f"Error getting raw depth image: {e}")
            return None

    def _update_obstacle_distances(self):
        """
        Update distances to obstacles using depth image and LiDAR for reward calculation.
        """
        try:
            min_distance = float('inf')
            
            # Use depth image
            depth_img = self._get_raw_depth_image()
            if depth_img is not None:
                # Filter out very large values (sky, etc.)
                valid_depths = depth_img[depth_img < 100]  # Filter out "infinity" values
                if valid_depths.size > 0:
                    min_depth = np.min(valid_depths)
                    min_distance = min(min_distance, min_depth)
            
            # Use LiDAR if available
            if self.use_lidar:
                lidar_data = self.client.getLidarData()
                if lidar_data and lidar_data.point_cloud:
                    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                    if points.shape[0] > 0:
                        # Calculate distances from all points to the origin (where the drone is)
                        distances = np.linalg.norm(points, axis=1)
                        if distances.size > 0:
                            lidar_min = np.min(distances)
                            min_distance = min(min_distance, lidar_min)
            
            # Update min distance if valid
            if min_distance < float('inf'):
                self.min_distance_to_obstacles = min(self.min_distance_to_obstacles, min_distance)
        except Exception as e:
            print(f"Error updating obstacle distances: {e}")

    def _get_position(self):
        """
        Get current drone position as numpy array.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])
    
    def _get_orientation(self):
        """
        Get current drone orientation as quaternion.
        """
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        return np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])

    def _calculate_reward(self):
        """
        Calculate reward based on:
        1. Progress toward current waypoint
        2. Collision avoidance
        3. Successful waypoint reaching
        4. Energy efficiency
        5. Obstacle proximity
        """
        reward = 0.0
        reward_info = {}
        
        # Get current position
        current_position = self._get_position()
        
        # Check for collisions
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            collision_reward = self.collision_penalty
            reward += collision_reward
            reward_info['collision'] = collision_reward
            self.collisions += 1
        
        # Calculate distance to current waypoint
        if self.current_waypoint_index < len(self.waypoints):
            current_waypoint = np.array(self.waypoints[self.current_waypoint_index])
            distance_to_waypoint = np.linalg.norm(current_position - current_waypoint)
            
            # Check if waypoint reached
            if distance_to_waypoint < self.target_reached_threshold:
                # Reward for reaching waypoint
                waypoint_reward = self.waypoint_reward
                reward += waypoint_reward
                reward_info['waypoint_reached'] = waypoint_reward
                
                # Update tracking metrics
                self.waypoints_reached += 1
                self.current_waypoint_index += 1
            else:
                # Calculate distance from last position to waypoint
                last_distance = np.linalg.norm(self.last_position - current_waypoint)
                
                # Reward for progress toward waypoint
                progress = last_distance - distance_to_waypoint
                progress_reward = progress * self.progress_reward_factor
                reward += progress_reward
                reward_info['progress'] = progress_reward
        
        # Penalize energy usage (based on control inputs, approximated by motion)
        motion = np.linalg.norm(current_position - self.last_position)
        energy_penalty = -motion * self.energy_penalty_factor
        reward += energy_penalty
        reward_info['energy'] = energy_penalty
        
        # Reward/penalty based on obstacle proximity
        if self.min_distance_to_obstacles < float('inf'):
            # Encourage keeping safe distance from obstacles, but not too far
            optimal_distance = 2.0  # Optimal distance to obstacles
            distance_diff = abs(self.min_distance_to_obstacles - optimal_distance)
            
            # Higher reward for staying near the optimal distance
            obstacle_reward = (1.0 / (1.0 + distance_diff)) * self.obstacle_distance_factor
            reward += obstacle_reward
            reward_info['obstacle_distance'] = obstacle_reward
        
        # Update last position
        self.last_position = current_position
        self.last_orientation = self._get_orientation()
        
        return reward, reward_info

    def _is_terminated(self):
        """
        Check if episode should terminate.
        """
        # Terminate if all waypoints reached
        if self.current_waypoint_index >= len(self.waypoints):
            return True
        
        # Terminate on collision if configured
        if self.config.get('terminate_on_collision', True):
            collision_info = self.client.simGetCollisionInfo()
            if collision_info.has_collided:
                return True
        
        return False

    def _is_truncated(self):
        """
        Check if episode should be truncated due to time limit.
        """
        return self.current_step >= self.max_steps

    def _plan_path(self):
        """
        Plan a path to the current waypoint using a simple algorithm.
        This could be enhanced with more sophisticated path planning algorithms.
        """
        # Get current position
        current_position = self._get_position()
        
        # Get current waypoint
        if self.current_waypoint_index < len(self.waypoints):
            target_waypoint = np.array(self.waypoints[self.current_waypoint_index])
            
            # For now, just plan a straight-line path with intermediate points
            num_points = 10
            planned_path = []
            
            for i in range(num_points + 1):
                t = i / num_points
                point = current_position * (1 - t) + target_waypoint * t
                planned_path.append(point)
            
            self.planned_path = planned_path
        else:
            self.planned_path = []

    def render(self, mode='human'):
        """
        Render environment visualization. In AirSim, this is often done externally,
        but we can capture and return images for debugging or recording.
        """
        if mode == 'human':
            # For human rendering, we don't need to do anything as AirSim
            # provides its own visualization
            return None
        
        elif mode == 'rgb_array':
            # Return RGB image for recording or debugging
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(0, airsim.ImageType.Scene)
                ])
                
                if responses:
                    response = responses[0]
                    img_rgba = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgba = img_rgba.reshape(response.height, response.width, 3)
                    return img_rgba
                else:
                    return np.zeros((84, 84, 3), dtype=np.uint8)
            except:
                return np.zeros((84, 84, 3), dtype=np.uint8)
        
        return None

    def close(self):
        """
        Clean up resources.
        """
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass

                
if __name__ == '__main__':
    # Example usage to test the environment (without PPO for now)
    env = AirSimForestEnv()
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    action = env.action_space.sample() # Sample random action
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step - observation shape:", obs.shape, "reward:", reward, "terminated:", terminated)
    env.close()

