import os
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy



class AirSimForestEnv(gym.Env):
    """
    Gymnasium environment for drone navigation in AirSim forest, using image input.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60} # Example metadata

    def __init__(self, ip_address='', config=None, client=None):
        super().__init__()
        self.config = config or {}

        if client:
            self.client = client
        else:
            # Original connection code
            ip_address = self.config.get('airsim_ip', '')
            self.client = airsim.MultirotorClient(ip=ip_address) if ip_address else airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


        
        # Image capture setup (you can customize camera name in settings.json)
        self.camera_name = self.config.get('camera_name', "0") # Assuming you are using Camera1 as defined in your settings.json
        self.image_type = airsim.ImageType.Scene # Or airsim.ImageType.DepthVis if you want to use depth images

        # Define action space (Roll, Pitch, Yaw Rate, Throttle - continuous)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) # Example action space

        # Define observation space (Image - example: grayscale 64x64 image)
        self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32) # Example grayscale image

        self.current_waypoint_index = 0
        self.waypoints = self.config.get('waypoints', [ # Example waypoints - adapt to your navigation task
            (7, 0, -3),
            (7, 23, -3),
            (0, 23, -3),
            (0, 0, -3)
        ])

    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, terminated, truncated, info.
        """
        roll, pitch, yaw_rate, throttle = action

        # Execute action in AirSim
        self.client.moveByRollPitchYawZAsync(roll, pitch, yaw_rate, -3, throttle, 0.1).join() # Example duration 0.1 sec
        # Wait for a small simulation step (important for SteppableClock)

        time.sleep(0.05) # Adjust sleep time as needed, smaller values for faster simulation


        # Get next observation (image)
        image_np = self._get_observation()

        # Calculate reward (replace with your reward function - basic example below)
        reward = self._calculate_reward()

        # Check for termination conditions (e.g., crash, reached goal, time limit)
        terminated = self._is_terminated()
        truncated = False # You can add truncation conditions if needed

        info = {} # Additional information (e.g., episode stats)

        return image_np, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """
        super().reset(seed=seed, options=options) # For Gymnasium compatibility
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.current_waypoint_index = 0 # Reset waypoint index

        # Initial drone position (optional - you can set a specific starting position)
        self.client.simSetVehiclePose(airsim.Pose(), True)

        # Get initial observation (image) after reset
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        """
        Get image observation from AirSim and preprocess it.
        """

        try:
            responses = self.client.simGetImages([airsim.ImageRequest(self.camera_name, self.image_type, pixels_as_float=False, compress=False)])
            response = responses[0]
            img_np = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_np = img_np.reshape(response.height, response.width, 3) # Assuming RGB
            gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) # Convert to grayscale
            resized_image = cv2.resize(img_np, (64, 64)) # Resize for CNN input
            normalized_image = resized_image / 255.0 # Normalize to 0-1
            return normalized_image.astype(np.float32).reshape(64, 64, 3) # Reshape to (64, 64, 1) for CNN input
        except Exception as e:
            print(f"Error getting observation: {e}")
            # Return a blank observation as fallback
            return np.zeros((64, 64, 1), dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculate reward based on drone's state and navigation task.
        (Replace with your actual reward function incorporating heuristics)
        """
        # Basic example reward - encourage forward motion and penalize altitude deviation
        state = self.client.getMultirotorState().kinematics_estimated
        forward_reward = state.linear_velocity.x_val * 0.1
        # forward_reward = state.linear_velocity.x_val # Example: reward for velocity in x direction
        # altitude_penalty = abs(state.position.z_val + 3) * 0.1 # Penalty for deviating from target altitude -3m

        # # Obstacle avoidance penalty (example - using depth, needs refinement)
        # depth_response = self.client.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthVis, pixels_as_float=True, compress=False)])[0]
        # depth_np = np.array(depth_response.image_data_float, dtype=np.float32)
        # depth_np = depth_np.reshape(depth_response.height, depth_response.width)
        # min_depth = np.min(depth_np) # Get minimum depth value (closest obstacle)
        # obstacle_penalty = 0
        # if min_depth < 2.0: # If obstacle closer than 2 meters (example threshold)
        #     obstacle_penalty = -1.0 # High penalty for getting too close

        # reward = forward_reward - altitude_penalty + obstacle_penalty # Combine rewards and penalties
        
        return forward_reward

    def _is_terminated(self):
        """
        Check for termination conditions (e.g., crash, reached goal).
        (Replace with your actual termination logic)
        """
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True # Terminate if crashed

        # Example waypoint completion (replace with your actual goal condition)
        if self.current_waypoint_index >= len(self.waypoints):
            return True # Terminate if all waypoints reached

        return False

    def render(self, mode="human"):
        """
        Render the environment (optional - for visualization).
        """
        if mode == "human":
            pass # You can add code to display a window if needed, but often not necessary in AirSim RL
        elif mode == "rgb_array":
            image_np = self._get_observation() # Reuse observation image for rendering
            return image_np
        else:
            super().render(mode=mode) # Handle other modes if needed

    def close(self):
        """
        Close the environment and release resources.
        """
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        # client.reset() # Resetting might be done in reset() already

if __name__ == '__main__':
    # Example usage to test the environment (without PPO for now)
    env = AirSimForestEnv()
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    action = env.action_space.sample() # Sample random action
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step - observation shape:", obs.shape, "reward:", reward, "terminated:", terminated)
    env.close()



# environment_name = 'DroneZone-V1'
# env = gym.make(environment_name)

# print(environment_name)

# episodes = 7
# for episodes in range(1, episdodes + 1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

