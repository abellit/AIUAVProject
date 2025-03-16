import os
import json
import datetime
import numpy as np
from pathlib import Path
import time


class TrainingManager:
    """
    Manages the transition between local development and Colab training environments.
    Handles configuration, model checkpointing, and environment detection.
    """
    def __init__(self):
        self.is_colab = self._check_colab()
        self.config = self._load_config()
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Curriculum learning parameters
        self.current_curriculum_phase = 0
        self.curriculum_thresholds = [0.3, 0.5, 0.7, 0.8]
        
        # Success window for curriculum advancement
        self.success_history = []
        self.success_window_size = 20  # Number of episodes to consider for success rate

        self.metrics_history = {
            'mean_reward': [],
            'success_rate': [],
            'episode_length': [],
            'collision_rate': [],
            'distance_travelled': [],
            'time_elapsed': []
        }
        
        # Setup directories immediately
        self.setup_directories()

    def setup_directories(self):
        """Create directories for training and analysis"""
        base_path = self.get_experiment_path()

        # Create directories
        for subdir in ['models', 'logs', 'tensorboard', 'config', 'visualizations']:
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    
    def get_experiment_path(self):
        """Get the base path for the current experiment"""
        if self.is_colab:
            # In Colab, we'll use a drive path if available
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                base_path = f"/content/drive/MyDrive/airsim_experiments/{self.experiment_id}"
            except ImportError:
                base_path = f"/content/airsim_experiments/{self.experiment_id}"
        else:
            # Local environment
            base_path = os.path.join(self.config.get('paths', {}).get('experiments', './experiments'), self.experiment_id)
        
        return base_path

    def setup_airsim_client(self):
        """Create and return the appropriate AirSim client based on environment"""
        try:
            import airsim
            # Use the IP from config if available, otherwise use default connection
            ip_address = self.config.get('airsim_ip', '')
            if ip_address:
                print(f"Connecting to AirSim at {ip_address}")
                client = airsim.MultirotorClient(ip=ip_address)
            else:
                print("Connecting to AirSim on localhost")
                client = airsim.MultirotorClient()
                
            # Verify connection
            client.confirmConnection()
            print("AirSim connection confirmed")
            return client
        except Exception as e:
            print(f"Error connecting to AirSim: {e}")
            raise

    def _check_colab(self):
        """Detect if we're running in Colab environment without importing google.colab."""
        # Check for typical Colab environment indicators
        try:
            return os.path.exists('/content') and 'COLAB_GPU' in os.environ
        except:
            return False
    
    def _load_config(self):
        """Load appropriate configuration based on environment."""
        default_config = {
            'paths': {
                'model_weights': './models',
                'logs': './logs',
                'experiments': './experiments'
            },
            'training': {
                'total_timesteps': 1000000,
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99
            },
            'environment': {
                'max_steps': 1000,
                'collision_penalty': -100.0,
                'waypoint_reward': 200.0,
                'progress_reward': 1.0,
                'energy_penalty': 0.005,
                'randomize_start': True
            },
            'features': {
                'feature_dim': 256
            }
        }
        
        try:
            # Attempt to load config from file
            if self.is_colab:
                config_path = './configs/cloud_config.json'
            else:
                # Try to find config in the current directory structure
                possible_paths = [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'local_config.json')),
                    os.path.abspath(os.path.join(os.getcwd(), 'configs', 'local_config.json')),
                    os.path.abspath('./configs/local_config.json')
                ]
                
                config_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
            
            if config_path and os.path.exists(config_path):
                print(f"Loading config from: {config_path}")
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge loaded config with default config
                self._deep_update(default_config, loaded_config)
                return default_config
            else:
                print(f"Config file not found. Using default configuration.")
                return default_config
                
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return default_config
    
    def _deep_update(self, d, u):
        """Deep update of nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def get_model_path(self):
        """Get the appropriate path for saving/loading models."""
        base_path = self.config.get('paths', {}).get('model_weights', './models')
        if self.is_colab:
            try:
                # Try to use Google Drive if available
                from google.colab import drive
                if not os.path.exists('/content/drive'):
                    drive.mount('/content/drive')
                return f"/content/drive/MyDrive/airsim_models/{self.experiment_id}"
            except ImportError:
                return f"/content/airsim_models/{self.experiment_id}"
        return os.path.join(base_path, self.experiment_id)
    
    def save_checkpoint(self, model, metrics, episode):
        """Save model checkpoint with metadata."""
        checkpoint_dir = Path(self.get_model_path())
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / f"checkpoint_{episode}"
        model.save(str(model_path))
        
        # Save training metadata
        metadata = {
            'episode': episode,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'environment': 'colab' if self.is_colab else 'local',
            'curriculum_phase': self.current_curriculum_phase
        }
        
        with open(checkpoint_dir / f"metadata_{episode}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved checkpoint at episode {episode}")
    
    def load_latest_checkpoint(self, model=None):
        """Load the latest checkpoint for the model."""
        checkpoint_dir = Path(self.get_model_path())
        
        # Go up one level to check for any experiment directories
        if not checkpoint_dir.exists():
            parent_dir = checkpoint_dir.parent
            if not parent_dir.exists():
                print("No checkpoints found - starting from scratch")
                return None, 0
                
            # Find latest experiment if current doesn't exist
            experiments = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
            if not experiments:
                print("No experiments found - starting from scratch")
                return None, 0
                
            checkpoint_dir = experiments[-1]
            print(f"Using latest experiment directory: {checkpoint_dir}")
        
        # Find latest checkpoint
        checkpoints = sorted([d for d in checkpoint_dir.glob("checkpoint_*") if d.is_dir()])
        if not checkpoints:
            print("No checkpoints found in experiment directory")
            return None, 0
            
        latest_checkpoint = checkpoints[-1]
        try:
            episode = int(latest_checkpoint.name.split('_')[1])
        except (IndexError, ValueError):
            episode = 0
            
        print(f"Loading checkpoint from episode {episode}")
        
        # Look for metadata to restore curriculum phase
        metadata_path = checkpoint_dir / f"metadata_{episode}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.current_curriculum_phase = metadata.get('curriculum_phase', 0)
                    print(f"Restored curriculum phase: {self.current_curriculum_phase}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # If model is provided, load the weights
        if model is not None:
            try:
                model.load(str(latest_checkpoint))
                print(f"Model weights loaded successfully")
                return model, episode
            except Exception as e:
                print(f"Error loading model: {e}")
                return None, episode
        
        return latest_checkpoint, episode
    
    def update_curriculum(self, env, success):
        """Update curriculum based on success rate"""
        # Add latest success result to history
        self.success_history.append(1 if success else 0)
        
        # Keep only the most recent results
        if len(self.success_history) > self.success_window_size:
            self.success_history.pop(0)
            
        # Calculate success rate
        if len(self.success_history) >= self.success_window_size:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Check if we should advance to next phase
            if (self.current_curriculum_phase < len(self.curriculum_thresholds) and 
                success_rate >= self.curriculum_thresholds[self.current_curriculum_phase]):
                self.current_curriculum_phase += 1
                print(f"\n--- ADVANCING TO CURRICULUM PHASE {self.current_curriculum_phase} ---")
                print(f"Success rate: {success_rate:.2f}")
                
                # Update environment curriculum
                if hasattr(env, 'update_curriculum'):
                    env.update_curriculum(success_rate)
    
    def log_training_stats(self, stats, episode):
        """Log training statistics to file and update metrics history"""
        log_dir = Path(self.get_experiment_path()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "training_log.csv"
        
        # Create header if file doesn't exist
        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write("episode,timestamp,mean_reward,success_rate,collision_rate,episode_length,curriculum_phase\n")
        
        # Log stats
        timestamp = datetime.datetime.now().isoformat()
        with open(log_file, 'a') as f:
            f.write(f"{episode},{timestamp},{stats['mean_reward']:.2f},{stats['success_rate']:.2f},"
                   f"{stats['collision_rate']:.2f},{stats['episode_length']:.2f},{self.current_curriculum_phase}\n")
        
        # Update metrics history
        for key in stats:
            if key in self.metrics_history:
                self.metrics_history[key].append(stats[key])
                
        return stats
    
    def evaluate_policy(self, model, env, n_eval_episodes=10):
        """Evaluate the policy and return performance metrics"""
        print(f"\nEvaluating policy over {n_eval_episodes} episodes...")
        rewards = []
        successes = 0
        collisions = 0
        episode_lengths = []
        
        for i in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Track success and collision
                if info.get('success', False):
                    successes += 1
                if info.get('collisions', 0) > 0:
                    collisions += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(step_count)
            print(f"Episode {i+1}/{n_eval_episodes}: Reward={episode_reward:.2f}, Steps={step_count}")
        
        # Calculate metrics
        stats = {
            'mean_reward': np.mean(rewards),
            'success_rate': successes / n_eval_episodes,
            'collision_rate': collisions / n_eval_episodes,
            'episode_length': np.mean(episode_lengths)
        }
        
        print(f"Evaluation results:")
        print(f"  Mean reward: {stats['mean_reward']:.2f}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Collision rate: {stats['collision_rate']:.2%}")
        print(f"  Mean episode length: {stats['episode_length']:.2f}")
        
        return self.log_training_stats(stats, model.num_timesteps)