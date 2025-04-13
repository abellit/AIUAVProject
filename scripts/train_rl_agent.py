import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort: skip

import os
import sys
import numpy as np
import time
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Figure
from src.utils.training_manager import TrainingManager
from src.rl_agent.airsim_env import AirSimForestEnv
from src.rl_agent.feature_extractor import ForestNavMultiInputExtractor  # Import ForestNavCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor, VecEnv, SubprocVecEnv
from typing import List, Optional, Tuple, Union
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import PPO


class CustomVecTransposeImage(VecEnv):
    """
    Custom wrapper to ensure images are correctly formatted for PyTorch.
    """
    def __init__(self, venv):
        # Initialize with the same metadata and spaces as the wrapped env
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.metadata = venv.metadata
        
    def reset(self, **kwargs):
        # Handle different return types from the wrapped environment
        result = self.venv.reset(**kwargs)
        
        # Check if result is a tuple (newer API returning obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            # Store info for potential later use but return only observation
            self._last_info = info
            return obs, info  # Return both observation and info for compatibility with SB3
        else:
            # Older API just returning observation
            obs = result
            self._last_info = {}
            return obs
        
    def step_async(self, actions):
        self.venv.step_async(actions)
        
    def step_wait(self):
        # 
        result = self.venv.step_wait()
    
        # Check if result has 5 elements (newer API with truncated flag)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, done, truncated, info = result
            # For compatibility with SB3, combine terminated and truncated into a single done
            done = done | truncated  # Logical OR
            return obs, reward, done, info
        else:
            # Older API with 4 elements
            return result
        
    def close(self):
        self.venv.close()
        
    @property
    def envs(self):
        return self.venv.envs
        
    # Add the missing abstract methods
    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)
        
    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.venv.env_is_wrapped(wrapper_class, indices)
    
    
# class MetricsCallBack(BaseCallback):
#     """
#     Callback for logging episode statistics.
#     """

#     def __init__(self, verbose=0):
#         super(MetricsCallBack, self).__init__(verbose)
#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.training_losses = {'policy_loss': [], 'value_loss': [], 'entropy_loss': [], 'kl_div': []}
#         self.episode_num = 0
#         self.episode_reward = 0
#         self.last_timesteps = 0
#         self.last_episode_time = time.time()

#     def _on_step(self) -> bool:
        
#         for info in self.locals['infos']:
#             if 'episode' in info.keys():
#                 self.episode_rewards.append(info['episode']['r'])
#                 self.episode_lengths.append(info['episode']['l'])
#                 self.episode_num += 1
#                 self.episode_reward = 0
#                 self.last_timesteps = self.num_timesteps
#                 self.last_episode_time = time.time()

#         if hasattr(self.model, 'loss_dict'):
#             self.training_losses['policy_loss'].append(self.model.loss_dict['policy_loss'])
#             self.training_losses['value_loss'].append(self.model.loss_dict['value_loss'])
#             self.training_losses['entropy_loss'].append(self.model.loss_dict['entropy_loss'])
#             self.training_losses['kl_div'].append(self.model.loss_dict['kl_div'])

#         return True
    
#     def get_metrics(self):
#         """Track training metrics."""
#         metrics = {}

#         if self.episode_rewards:
#             metrics['mean_reward'] = np.mean(self.episode_rewards[-10:])
#             metrics['min_reward'] = np.min(self.episode_rewards[-10:])
#             metrics['max_reward'] = np.max(self.episode_rewards[-10:])
#             metrics['mean_episode_length'] = np.mean(self.episode_lengths[-10:])
#             metrics['total_episodes'] = self.episode_num

#             success_threshold = 0.75
#             recent_episodes = self.episode_rewards[-10:]
#             success_rate = sum([1 for r in recent_episodes if r > success_threshold]) / len(recent_episodes)
#             metrics['success_rate'] = success_rate

#         for key, values in self.training_losses.items():
#             if values:
#                 metrics[key] = np.mean(values[-100:])
        
#         return metrics

#     def plot_metrics(self, log_path):
#         """Plot training metrics."""
#         plt.figure(figsize=(16, 8))
#         plt.plot(self.episode_rewards, label='Episode Reward')
#         plt.title("Episode Rewards")
#         plt.xlabel("Episode")
#         plt.ylabel("Reward")
#         plt.grid()
#         plt.legend()
#         plt.show()
#         plt.savefig(f"{log_path}/reward_curve.png")
#         plt.close()
        

#         plt.figure(figsize=(16, 8))
#         window_size = min(10, len(self.episode_rewards))
#         if window_size > 1:
#             smoothed_rewards = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode='valid')
#             plt.plot(smoothed_rewards, label='Smoothed Episode Reward')
#             plt.title("Smoothed Episode Rewards")
#             plt.xlabel("Episode")
#             plt.ylabel("Reward")
#             plt.grid()
#             plt.legend()
#             plt.show()
#             plt.savefig(f"{log_path}/smoothed_reward_curve.png")
#             plt.close()



def train_sar_drone():
    """
    Train a Search and Rescue drone to navigate through dense forest environments.
    Using PPO with custom CNN feature extractor optimized for forest navigation.
    """
    print("Initializing Search and Rescue Drone Training...")
    
    # Initialize training manager
    manager = TrainingManager()
    config = manager.config
    
    # Get AirSim client through the manager
    client = manager.setup_airsim_client() if manager.is_colab else None
    
    # Create training environment
    def make_env(rank=0, config=None, client=None):
        def _init():
            env = AirSimForestEnv(config=config, client=client)
            return env
        return _init
    
    # Create vectorized environmenst
    print("Setting up environment...")

    """ env_config = {
        'max_steps': 2000,
        'use_rgb': True,
        'use_depth': True,
        'use_lidar': True,
        #'use_gps': True,
        'use_imu': True,
        #'use_barometer': True,
        #'use_distance_sensor': True,
        'randomize_start': True,
        'lidar_points': 1024,
        'lidar_range': 50.0,
        'collision_penalty': -100.0,
        'waypoint_reward': 50.0, 
        'progress_reward': 0.1,
        'energy_penalty': 0.01,
        'obstacle_distance_factor': 0.5
    } """

    num_envs = 2
    env = SubprocVecEnv([make_env(i, config, client) for i in range(num_envs)])
    #env = DummyVecEnv([make_env(config)])
    
    # Add Monitor wrapper to log episode statistics
    env = VecMonitor(env, os.path.join("./logs", "sar_drone_monitor"))
    
    # Add VecTransposeImage to convert image channels for CNN
    #env = VecTransposeImage(env)

    env = CustomVecTransposeImage(env)
    
    # Define policy kwargs with our custom feature extractor
    policy_kwargs = {
        "features_extractor_class": ForestNavMultiInputExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
    }
    
    # Define logger path
    log_path = "./logs/sar_training/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_path, exist_ok=True)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=os.path.join(log_path, "checkpoints"),
        name_prefix="sar_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Create evaluation environment
    #eval_env = DummyVecEnv([make_env(config)])
    eval_env = SubprocVecEnv([make_env(i, config, client) for i in range(num_envs)])
    #eval_env = VecTransposeImage(eval_env)

    #eval_metrics_env = None
    eval_env = CustomVecTransposeImage(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=os.path.join(log_path, "eval_results"),
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    #metrics_callback = MetricsCallBack()
    
    # Parameters
    timesteps_per_epoch = config.get('training', {}).get('timesteps_per_epoch', 1000)
    total_epochs = config.get('training', {}).get('total_epochs', 50)
    learning_rate = config.get('training', {}).get('learning_rate', 3e-4)
    
    # Initialize PPO model with custom parameters
    print("Creating PPO model with custom CNN...")
    model = PPO(
        MultiInputActorCriticPolicy,  # Custom policy with multiple inputs
        env,
        learning_rate=3e-5,
        n_steps=1024,  # Longer trajectories for stable learning
        batch_size=16,
        n_epochs=5,   # More update epochs per batch
        gamma=0.99,    # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.1,
        policy_kwargs=policy_kwargs,
        ent_coef=0.02,  # Entropy coefficient for exploration
        vf_coef=0.5,    # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        verbose=1
    )

    # replay_buffer = ReplayBuffer(
    #     buffer_size=config['training']['buffer_size'],
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     device="cpu",
    #     n_envs=num_envs
    # )

    def check_model_observation_space(model_path, env):
        """
        Check if a saved model's observation space matches the environment's Dict observation space.
        """
        try:
            saved_model = PPO.load(model_path)
            if saved_model.observation_space == env.observation_space:
                print(f"✅ Model {model_path} is compatible with Dict observation space.")
                return True
            else:
                print(f"❌ Model {model_path} has mismatched observation space:\n"
                    f"   Saved: {saved_model.observation_space}\n"
                    f"   Expected: {env.observation_space}")
                return False
        except Exception as e:
            print(f"⚠️ Failed to load model {model_path}: {e}")
            return False

    def find_latest_best_model(env):
        """
        Finds the most recent and compatible best model that matches the multi-input Dict observation space.
        """
        base_dir = "./logs/sar_training/"
        best_models = []

        # Walk through all directories in the base directory
        for root, dirs, files in os.walk(base_dir):
            if "best_model" in dirs:
                best_model_path = os.path.join(root, "best_model", "best_model.zip")
                if os.path.exists(best_model_path):
                    # Get last modified time
                    dir_time = os.path.getmtime(os.path.join(root, "best_model"))
                    best_models.append((best_model_path, dir_time))

        # Sort models by creation time (latest first)
        best_models.sort(key=lambda x: x[1], reverse=True)

        # Return the first compatible model
        for model_path, _ in best_models:
            if check_model_observation_space(model_path, env):
                return model_path

        print("🚨 No compatible model found. Starting training from scratch.")
        return None
    
    # Try to load previous checkpoint
    if manager.is_colab:
        # Specific handling for Colab environment
        print("Checking for model checkpoints in Colab environment...")
        try:
            loaded_model, start_epoch = manager.load_latest_checkpoint(model)
            if loaded_model:
                model = loaded_model
                print(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0
                print("No previous checkpoint found. Starting training from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            start_epoch = 0
    else:
        # Local environment
        print("Checking for model checkpoints in local environment...")
        checkpoint_path = os.path.join(log_path, "checkpoints")
        best_model_path = find_latest_best_model(env)
        
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model = PPO.load(best_model_path, env=env)
            start_epoch = 0  # We don't know which epoch it was from
        else:
            start_epoch = 0
            print("No previous best model found. Starting training from scratch.")
    # Training loop
    print(f"Starting training for {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")


        print("Attempting to reset AirSim client before epoch...")
        if client:
            try:
                client.reset()  # This should reset the simulation state
                if epoch == 0 or connection_is_broken:
                    client.confirmConnection()
                    client.enableApiControl(True)
                    client.armDisarm(True)
                    connection_is_broken = False
                print("AirSim client reset successfully.")
            except Exception as e:
                print(f"Error resetting AirSim client: {e}")
                connection_is_broken = True
        else:
            print("AirSim client is None, skipping reset.")
        
        # Train for one epoch
        model.learn(
            total_timesteps=timesteps_per_epoch,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False
        )
        
    #     # Log training metrics
    #     metrics = metrics_callback.get_metrics()
    #     print(f"\nEpoch {epoch+1} Metrics:")
    #     print(f"{'='*30}")

    #     # Display metrics in a nice format
    #     if 'mean_reward' in metrics:
    #         print(f"Mean Reward (Accuracy Proxy): {metrics['mean_reward']:.2f}")
    #     if 'mean_episode_length' in metrics:
    #         print(f"Mean Episode Length: {metrics['mean_episode_length']:.2f}")
    #     if 'success_rate' in metrics:    
    #         print(f"Success Rate: {metrics['success_rate']*100:.2f}%")
        

    #     # Display training losses
    #     print(f"\nTraining Losses:")
    #     print(f"{'-'*30}")
    #     if 'policy_loss' in metrics:
    #         print(f"Policy Loss: {metrics['policy_loss']:.6f}")
    #     if 'value_loss' in metrics:
    #         print(f"Value Loss: {metrics['value_loss']:.6f}")
    #     if 'entropy_loss' in metrics:
    #         print(f"Entropy Loss: {metrics['entropy_loss']:.6f}")   
    #     if 'kl_div' in metrics:
    #         print(f"KL Divergence: {metrics['kl_div']:.6f}")

        
    #     #Plot \training metrics
    #     metrics_callback.plot_metrics(log_path)


    #     # Save model after each epoch
    #     epoch_save_path = os.path.join(log_path, f"model_epoch_{epoch+1}")
    #     model.save(epoch_save_path)
    #     print(f"Model saved to {epoch_save_path}")


    #     # Save detailed metrics to JSON
    #     with open(os.path.join(log_path, f"metrics_epoch_{epoch+1}.json"), "w") as f:
    #         json.dump(metrics, f, indent=2)
        
    #     #Log training progress
    #     with open(os.path.join(log_path, "training_log.txt"), "a") as f:
    #         f.write(f"Completed epoch {epoch+1}/{total_epochs}\n")
    #         f.write(f"Timestamp: {datetime.now()}\n")

    #         for key, value in metrics.items():
    #             f.write(f"{key}: {value}\n")

    #         f.write("---\n")

    # print("\nPerforming final evaluation...")
    # mean_reward, std_reward = evaluate_policy(
    #     model, 
    #     eval_env, 
    #     n_eval_episodes=10,
    #     deterministic=True
    # )

        if (epoch + 1) % 5 == 0:
            print("Evaluating model...")
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=5,
                deterministic=True
            )
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # if eval_metrics_env is None:
        #     eval_metrics_env = AirSimForestEnv(config=config, client=client)
        # else:
        #     eval_metrics_env.reset()

        # if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
        
        #     n_eval_episodes = 2
        #     episode_metrics = []

        #     for episode in range(n_eval_episodes):
        #         obs = eval_metrics_env.reset()
        #         done = False
        #         episode_reward = 0

        #         while not done:
        #             action, _ = model.predict(obs, deterministic=True)
        #             obs, reward, done, info = eval_metrics_env.step(action)
        #             episode_reward += reward

        #         # Get success metrics from the environment
        #         success_metrics = eval_metrics_env.calculate_success_metrics()
        #         success_metrics['episode_reward'] = episode_reward
        #         episode_metrics.append(success_metrics)

    print("Training completed!")
    model.save(os.path.join(log_path, "final_model"))
    
    # Final evaluation using evaluate_policy
    print("Final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=5,
        deterministic=True
    )
    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # avg_metrics = {
    #     'waypoint_success_rate': np.mean([m['waypoint_success_rate'] for m in episode_metrics]),
    #     'obstacle_success_rate': np.mean([m['obstacle_success_rate'] for m in episode_metrics]),
    #     'overall_success_rate': np.mean([m['overall_success_rate'] for m in episode_metrics]),
    #     'waypoints_reached': np.mean([m['waypoints_reached'] for m in episode_metrics]),
    #     'total_waypoints': np.mean([m['total_waypoints'] for m in episode_metrics]),
    #     'obstacles_encountered': np.mean([m['obstacles_encountered'] for m in episode_metrics]),
    #     'obstacles_avoided': np.mean([m['obstacles_avoided'] for m in episode_metrics]),
    #     'episode_reward': np.mean([m['episode_reward'] for m in episode_metrics])
    # }
    
    # # Display detailed success metrics in an attractive terminal format
    # print("\n" + "="*80)
    # print(f"{'SEARCH AND RESCUE DRONE TRAINING COMPLETED':^80}")
    # print("="*80)
    
    # print(f"\n{'FINAL SUCCESS METRICS':^80}")
    # print("-"*80)
    # print(f"Waypoint Navigation Success Rate:    {avg_metrics['waypoint_success_rate']:>7.2f}%")
    # print(f"Obstacle Avoidance Success Rate:     {avg_metrics['obstacle_success_rate']:>7.2f}%")
    # print(f"Overall Mission Success Rate:        {avg_metrics['overall_success_rate']:>7.2f}%")
    # print(f"Average Episode Reward:              {avg_metrics['episode_reward']:>7.2f}")
    # print("-"*80)
    
    # print(f"\n{'DETAILED STATISTICS':^80}")
    # print("-"*80)
    # print(f"Average Waypoints Reached:           {avg_metrics['waypoints_reached']:>7.2f} / {avg_metrics['total_waypoints']:>7.2f}")
    # print(f"Average Obstacles Encountered:       {avg_metrics['obstacles_encountered']:>7.2f}")
    # print(f"Average Obstacles Avoided:           {avg_metrics['obstacles_avoided']:>7.2f}")
    # print("-"*80)
    
    # # Save final metrics to JSON
    # final_metrics_path = os.path.join(log_path, "final_success_metrics.json")
    # with open(final_metrics_path, "w") as f:
    #     json.dump(avg_metrics, f, indent=2)
    
    # print(f"\nFinal metrics saved to: {final_metrics_path}")
    # print("\nTraining and evaluation completed!")
    
    # Close environments
    env.close()
    eval_env.close()
    
    # return avg_metrics


   

# def evaluate_sar_drone(model_path):
#     """
#     Evaluate a trained Search and Rescue drone model.
#     """

#     print(f"Evaluating model from {model_path}...")

#     # Initialize training manager
#     manager = TrainingManager()
#     config = manager.config

#     # Get AirSim client through the manager
#     client = manager.setup_airsim_client() if manager.is_colab else None

#     # Create evaluation environment

#     env = DummyVecEnv([lambda: AirSimForestEnv(config=config, client=client)])
#     #env = VecTransposeImage(env)
#     env = CustomVecTransposeImage(env)

#     # Load the model
#     model = PPO.load(model_path, env=env)

#     # Evaluate the model
#     episode_rewards = []
#     episode_successes = []
#     n_eval_episodes = 10

#     obs, _ = env.reset(), None

#     for i in range(n_eval_episodes):
#         print(f"Evaluating episode {i+1}/{n_eval_episodes}...")
#         episode_reward = 0
#         done = False

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             episode_reward += reward[0]
        
#         episode_rewards.append(episode_reward)
#         episode_success = episode_reward > 0.75
#         episode_successes.append(episode_success)

#         print(f"Episode {i+1} reward: {episode_reward}, Success: {episode_success}")  

#         obs, _ = env.reset(), None

#     mean_reward = np.mean(episode_rewards)  
#     std_reward = np.std(episode_rewards)
#     success_rate = np.mean(episode_successes)

#     print(f"\nEvaluation Results:")
#     print(f"{'='*30}")
#     print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#     print(f"Success Rate: {success_rate*100:.2f}%")
#     print(f"Total Individual Episode Rewards: {episode_rewards}")
#     print("Evaluation complete.")

#     env.close()

#     return mean_reward, success_rate



# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # isort: skip
# import numpy as np
# import json
# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage, SubprocVecEnv, VecEnv
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from stable_baselines3.common.policies import MultiInputActorCriticPolicy
# from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from src.utils.training_manager import TrainingManager
# from src.rl_agent.airsim_env import AirSimForestEnv
# from src.rl_agent.feature_extractor import ForestNavMultiInputExtractor


# class CustomVecTransposeImage(VecEnv):
#     """
#     Custom wrapper to ensure images are correctly formatted for PyTorch.
#     """
#     def __init__(self, venv):
#         # Initialize with the same metadata and spaces as the wrapped env
#         self.venv = venv
#         self.num_envs = venv.num_envs
#         self.observation_space = venv.observation_space
#         self.action_space = venv.action_space
#         self.metadata = venv.metadata
        
#     def reset(self, **kwargs):
#         # Handle different return types from the wrapped environment
#         result = self.venv.reset(**kwargs)
        
#         # Check if result is a tuple (newer API returning obs, info)
#         if isinstance(result, tuple) and len(result) == 2:
#             obs, info = result
#             # Store info for potential later use but return only observation
#             self._last_info = info
#             return obs, info  # Return both observation and info for compatibility with SB3
#         else:
#             # Older API just returning observation
#             obs = result
#             self._last_info = {}
#             return obs
        
#     def step_async(self, actions):
#         self.venv.step_async(actions)
        
#     def step_wait(self):
#         # 
#         result = self.venv.step_wait()
    
#         # Check if result has 5 elements (newer API with truncated flag)
#         if isinstance(result, tuple) and len(result) == 5:
#             obs, reward, done, truncated, info = result
#             # For compatibility with SB3, combine terminated and truncated into a single done
#             done = done | truncated  # Logical OR
#             return obs, reward, done, info
#         else:
#             # Older API with 4 elements
#             return result
        
#     def close(self):
#         self.venv.close()
        
#     @property
#     def envs(self):
#         return self.venv.envs
        
#     # Add the missing abstract methods
#     def get_attr(self, attr_name, indices=None):
#         return self.venv.get_attr(attr_name, indices)
        
#     def set_attr(self, attr_name, value, indices=None):
#         return self.venv.set_attr(attr_name, value, indices)
        
#     def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
#         return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
        
#     def env_is_wrapped(self, wrapper_class, indices=None):
#         return self.venv.env_is_wrapped(wrapper_class, indices)

# def make_env(config, rank=0):
#     def _init():
#         env = AirSimForestEnv(config=config)
#         env = Monitor(env)
#         return env
#     return _init

# def train_sar_drone():
#     print("Initializing Search and Rescue Drone Training...")
#     manager = TrainingManager()
#     config = manager.config
#     client = manager.setup_airsim_client() if manager.is_colab else None

#     num_envs = 1
#     # env = SubprocVecEnv([make_env(config, i) for i in range(num_envs)])
#     env = DummyVecEnv([make_env(config, i) for i in range(num_envs)])
#     env = VecMonitor(env, os.path.join("./logs", "sar_drone_monitor"))
#     #env = VecTransposeImage(env)
#     env = CustomVecTransposeImage(env)

#     policy_kwargs = {
#         "features_extractor_class": ForestNavMultiInputExtractor,
#         "features_extractor_kwargs": {"features_dim": 256},
#         "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
#     }
    
#     log_path = "./logs/sar_training/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#     os.makedirs(log_path, exist_ok=True)
    
#     checkpoint_callback = CheckpointCallback(
#         save_freq=1000, save_path=os.path.join(log_path, "checkpoints"), name_prefix="sar_model"
#     )
#     # eval_env = SubprocVecEnv([make_env(config, i) for i in range(num_envs)])
#     eval_env = DummyVecEnv([make_env(config, i) for i in range(num_envs)])
#     #eval_env = VecTransposeImage(eval_env)
#     eval_env = CustomVecTransposeImage(eval_env)

#     eval_callback = EvalCallback(
#         eval_env, best_model_save_path=os.path.join(log_path, "best_model"), log_path=os.path.join(log_path, "eval_results"), eval_freq=1000
#     )
    
#     print("Creating PPO model with custom CNN...")
#     model = PPO(
#         MultiInputActorCriticPolicy, env, learning_rate=config['training']['learning_rate'],
#         n_steps=2048, batch_size=config['training']['batch_size'], n_epochs=5, gamma=0.99,
#         gae_lambda=0.95, clip_range=0.1, policy_kwargs=policy_kwargs, ent_coef=0.02,
#         vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=os.path.join(log_path, "tensorboard"), verbose=1
#     )
    
#     # Initialize Replay Buffer
#     # replay_buffer = ReplayBuffer(
#     #     buffer_size=config['training']['buffer_size'],
#     #     observation_space=env.observation_space,
#     #     action_space=env.action_space,
#     #     device="cpu",
#     #     n_envs=num_envs
#     # )
    
#     print("Starting training...")
#     total_epochs = config['training']['total_epochs']
#     timesteps_per_epoch = config['training']['timesteps_per_epoch']
    
#     for epoch in range(total_epochs):
#         print(f"Epoch {epoch+1}/{total_epochs}")

#         print("Attempting to reset AirSim client before epoch...")
#         if client:
#             try:
#                 client.reset()  # This should reset the simulation state
#                 if epoch == 0 or connection_is_broken:
#                     client.confirmConnection()
#                     client.enableApiControl(True)
#                     client.armDisarm(True)
#                     connection_is_broken = False
#                 print("AirSim client reset successfully.")
#             except Exception as e:
#                 print(f"Error resetting AirSim client: {e}")
#                 connection_is_broken = True
#         else:
#             print("AirSim client is None, skipping reset.")
        
#         model.learn(
#             total_timesteps=timesteps_per_epoch, 
#             callback=[checkpoint_callback, eval_callback], 
#             reset_num_timesteps=False
#         )
        
#         # Evaluate model every 5 epochs using evaluate_policy
#         if (epoch + 1) % 5 == 0:
#             print("Evaluating model...")
#             mean_reward, std_reward = evaluate_policy(
#                 model, 
#                 eval_env, 
#                 n_eval_episodes=5,
#                 deterministic=True
#             )
#             print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
#     print("Training completed!")
#     model.save(os.path.join(log_path, "final_model"))
    
#     # Final evaluation using evaluate_policy
#     print("Final evaluation...")
#     mean_reward, std_reward = evaluate_policy(
#         model, 
#         eval_env, 
#         n_eval_episodes=5,
#         deterministic=True
#     )
#     print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
#     env.close()
#     eval_env.close()

if __name__ == "__main__":
    train_sar_drone()

# if __name__ == "__main__":
#     train_sar_drone()
    # import argparse

    # parser = argparse.ArgumentParser(description="Train or evaluate SAR drone model.")
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help="Whether to train or evaluate the model.")
    # parser.add_argument('--model_path', type=str, help="Path to model for evaluation.")
    # args = parser.parse_args()
    

    # if args.mode == 'train':
    #     train_sar_drone()
    # elif args.mode == 'eval' and args.model_path:
    #     evaluate_sar_drone(args.model_path)
    # else:
    #     print("Please provide a valid mode and model path for evaluation")










































# def train_ppo_cnn():
#     """
#     Trains the PPO agent with a custom CNN policy in the AirSim environment, managed by TrainingManager for configuration and checkpointing
#     """

#     manager = TrainingManager()
#     config = manager.config
#     # Get AirSim client through the manager
#     client = manager.setup_airsim_client() if manager.is_colab else None

#     # Pass the client to the environment
#     env = AirSimForestEnv(config=config, client=client)

#     model = PPO(ActorCriticCnnPolicy, env,
#                 batch_size=config['training']['batch_size'],
#                 learning_rate=config['training']['learning_rate'],
#                 verbose=1,
#                 tensorboard_log="./ppo_cnn_tensorboard/"
#                 )

#     loaded_model, start_episode = manager.load_latest_checkpoint(
#         PPO(ActorCriticCnnPolicy, env))
#     if loaded_model:
#         model = loaded_model
#         print(f"Resuming training from episode {start_episode}")
#     else:
#         start_episode = 0
#         print("Starting training from scratch.")

#     max_episodes = config['training']['max_episodes']
#     for episode in range(start_episode + 1, max_episodes + 1):
#         print(f"Starting episode {episode}/{max_episodes}")
#         model.learn(total_timesteps=10000, reset_num_timesteps=False)
#         metrics = {'episode_reward_mean': 0}
#         manager.save_checkpoint(model.policy, metrics, episode)

#     print("PPO CNN training finished.")
#     env.close()


# if __name__ == "__main__":
#     train_ppo_cnn()
