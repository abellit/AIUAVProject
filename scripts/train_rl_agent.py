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

from src.utils.training_manager import TrainingManager
from src.rl_agent.airsim_env import AirSimForestEnv
from src.rl_agent.feature_extractor import ForestNavCNN  # Import ForestNavCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import VecEnv
from typing import List, Optional, Tuple, Union
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
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
            return obs  # Only return observation for compatibility with SB3
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
    def make_env(rank=0):
        def _init():
            env = AirSimForestEnv(config=config, client=client)
            return env
        return _init
    
    # Create vectorized environment
    print("Setting up environment...")
    env = DummyVecEnv([make_env()])
    
    # Add Monitor wrapper to log episode statistics
    env = VecMonitor(env, os.path.join("./logs", "sar_drone_monitor"))
    
    # Add VecTransposeImage to convert image channels for CNN
    #env = VecTransposeImage(env)

    env = CustomVecTransposeImage(env)
    
    # Define policy kwargs with our custom feature extractor
    policy_kwargs = {
        "features_extractor_class": ForestNavCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
    }
    
    # Define logger path
    log_path = "./logs/sar_training/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_path, exist_ok=True)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_path, "checkpoints"),
        name_prefix="sar_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env()])
    #eval_env = VecTransposeImage(eval_env)

    eval_env = CustomVecTransposeImage(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=os.path.join(log_path, "eval_results"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Parameters
    timesteps_per_epoch = config.get('training', {}).get('timesteps_per_epoch', 10000)
    total_epochs = config.get('training', {}).get('total_epochs', 100)
    learning_rate = config.get('training', {}).get('learning_rate', 3e-4)
    
    # Initialize PPO model with custom parameters
    print("Creating PPO model with custom CNN...")
    model = PPO(
        ActorCriticCnnPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=1024,  # Longer trajectories for stable learning
        batch_size=64,
        n_epochs=10,   # More update epochs per batch
        gamma=0.99,    # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        verbose=1
    )
    
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
        best_model_path = os.path.join(log_path, "best_model", "best_model.zip")
        
        if os.path.exists(best_model_path):
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
        
        # Train for one epoch
        model.learn(
            total_timesteps=timesteps_per_epoch,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False
        )
        
        # Save model after each epoch
        epoch_save_path = os.path.join(log_path, f"model_epoch_{epoch+1}")
        model.save(epoch_save_path)
        print(f"Model saved to {epoch_save_path}")
        
        # Log training progress
        with open(os.path.join(log_path, "training_log.txt"), "a") as f:
            f.write(f"Completed epoch {epoch+1}/{total_epochs}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("---\n")
    
    print("Training completed!")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_sar_drone()



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
