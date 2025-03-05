import os
import time
import gymnasium as gym
from gymnasium import spaces
#from gymnasium.spaces.utils import is_image_space, flatten_space
import numpy as np
import airsim
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy



class AirSimForestEnv(gym.Env):
    """
    Gymnasium environment for Search and Rescue (SAR) drone navigation 
    in dense forest environments using AirSim.
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
        
        # Image capture setup - using multiple cameras for better perception
        self.front_camera = self.config.get('front_camera', "0")
        self.depth_camera = self.config.get('depth_camera', "0")  # Using same camera with different image type
        
        # Define action space: Roll, Pitch, Yaw Rate, Throttle - continuous
        # Scaled to be between -1 and 1 for easier learning
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space - RGB image with 3 channels
        self.image_height = 64
        self.image_width = 64
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, self.image_height, self.image_width),  # Channel-first format (C, H, W)
            dtype=np.uint8
        )

        self.observation_space.is_image = True

        # Target waypoints for navigation - these would be generated based on the mission
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
        
        # Tracking metrics
        self.collisions = 0
        self.waypoints_reached = 0
        self.min_distance_to_obstacles = float('inf')
        self.total_reward = 0.0
        
        # For simulation stability
        self.last_position = np.zeros(3)
        
        # Initialize obstacle detection
        self._update_obstacle_distances()

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
                float(roll) * 0.3,    # Scale down for stability
                float(pitch) * 0.3,   # Scale down for stability
                float(yaw_rate) * 0.3,  
                -3 + float(throttle),  # Base altitude of -3m with throttle adjustment
                0.1  # Duration - keep short for responsive control
            ).join()
            
            # Allow small time for physics to stabilize
            time.sleep(0.05)  
        except Exception as e:
            print(f"Control error: {e}")
        
        # Get observation
        observation = self._get_observation()
        
        # Update obstacle distances for reward calculation
        self._update_obstacle_distances()
        
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
            'reward_breakdown': reward_info
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
        
        # Get initial observation
        observation = self._get_observation()
        
        # Update obstacle distances
        self._update_obstacle_distances()
        
        # Reset position tracking
        self.last_position = self._get_position()
        
        info = {
            'reset': True,
            'initial_position': initial_position.tolist()
        }
        
        return observation, info

    def _get_observation(self):
        """
        Get RGB camera image and process it for the neural network.
        """
        try:
            # Request RGB image
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.front_camera, airsim.ImageType.Scene, False, False)
            ])
            
            if not responses:
                raise Exception("No image returned from AirSim")
                
            response = responses[0]
            
            # Convert to numpy array
            img_rgba = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgba = img_rgba.reshape(response.height, response.width, 3)  # RGB format
            
            # Resize to the dimensions expected by the CNN
            img_resized = cv2.resize(img_rgba, (self.image_width, self.image_height))
            
            img_channels_first = np.transpose(img_resized, (2, 0, 1))
            # Normalize pixel values to [0, 1]
            # img_normalized = img_resized.astype(np.float32) / 255.0

            # img_normalised = np.transpose(img_normalized, (2, 0, 1))

            # print(f"Observation shape: {img_channels_first.shape}, dtype: {img_channels_first.dtype}")
            # print(f"Observation shape: {img_normalised.shape}, dtype: {img_normalised.dtype}")
            
            return img_channels_first.astype(np.uint8)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            # Return a blank observation as fallback
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)

    def _update_obstacle_distances(self):
        """
        Update distances to obstacles using depth image for reward calculation.
        """
        try:
            # Get depth image
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
                
                # Filter out very large values (sky, etc.)
                depth_img = np.array(depth_img)
                valid_depths = depth_img[depth_img < 100]  # Filter out "infinity" values
                
                if valid_depths.size > 0:
                    min_depth = np.min(valid_depths)
                    self.min_distance_to_obstacles = min(self.min_distance_to_obstacles, min_depth)
        except Exception as e:
            print(f"Error updating obstacle distances: {e}")

    def _get_position(self):
        """
        Get current drone position as numpy array.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def _calculate_reward(self):
        """
        Calculate reward based on:
        1. Progress toward current waypoint
        2. Collision avoidance
        3. Successful waypoint reaching
        4. Energy efficiency
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
                waypoint_reward = self.waypoint_reward
                reward += waypoint_reward
                reward_info['waypoint_reached'] = waypoint_reward
                self.waypoints_reached += 1
                self.current_waypoint_index += 1
            else:
                # Calculate progress toward waypoint
                last_distance = np.linalg.norm(self.last_position - current_waypoint)
                progress = last_distance - distance_to_waypoint
                
                progress_reward = progress * self.progress_reward_factor
                reward += progress_reward
                reward_info['progress'] = progress_reward
        
        # Obstacle avoidance reward - encourage keeping distance from obstacles
        if self.min_distance_to_obstacles < float('inf'):
            # Higher reward for staying away from obstacles, with a safe threshold
            safe_distance = 5.0
            min_safe_distance = 1.0
            
            if self.min_distance_to_obstacles < min_safe_distance:
                # Penalty for getting too close
                obstacle_reward = -1.0 * (min_safe_distance - self.min_distance_to_obstacles) * 10.0
            elif self.min_distance_to_obstacles < safe_distance:
                # Small positive reward for maintaining reasonable distance
                obstacle_reward = (self.min_distance_to_obstacles - min_safe_distance) * 0.5
            else:
                obstacle_reward = 0.0  # No additional reward beyond safe distance
                
            reward += obstacle_reward
            reward_info['obstacle_distance'] = obstacle_reward
        
        # Energy efficiency reward - penalize excessive movement
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        velocity_mag = np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
        
        # Penalize high velocities to encourage smooth, efficient movement
        energy_penalty = -velocity_mag**2 * self.energy_penalty_factor
        reward += energy_penalty
        reward_info['energy'] = energy_penalty
        
        # Update position tracking
        self.last_position = current_position
        
        return reward, reward_info

    def _is_terminated(self):
        """
        Check if episode should terminate due to:
        1. Collision
        2. Completed all waypoints
        3. Reached final target
        """
        # Check for collision
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        
        # Check if all waypoints have been reached
        if self.current_waypoint_index >= len(self.waypoints):
            return True
        
        return False

    def _is_truncated(self):
        """
        Check if episode should be truncated due to:
        1. Maximum steps exceeded
        2. Drone stuck or not making progress
        """
        # Check for maximum steps
        if self.current_step >= self.max_steps:
            return True
            
        return False

    def render(self, mode="human"):
        """
        Render the environment.
        """
        if mode == "rgb_array":
            return self._get_observation()
        return super().render(mode=mode)

    def close(self):
        """
        Clean up resources.
        """
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == '__main__':
    # Example usage to test the environment (without PPO for now)
    env = AirSimForestEnv()
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    action = env.action_space.sample() # Sample random action
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step - observation shape:", obs.shape, "reward:", reward, "terminated:", terminated)
    env.close()

