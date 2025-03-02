import os
import json
import datetime
from pathlib import Path


class TrainingManager:
    """
    Manages the transition between local development and Colab training environments.
    Handles configuration, model checkpointing, and environment detection.
    """
    def __init__(self):
        print("TrainingManager initialized - UPDATED VERSION")
        self.is_colab = self._check_colab()
        self.config = self._load_config()
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    def setup_airsim_client(self):
        """Create and return the appropriate AirSim client based on environment"""
        if self.is_colab:
            # In Colab, we need special handling for AirSim
            import airsim
            # You might need to specify IP if running AirSim remotely
            client = airsim.MultirotorClient(ip=self.config.get('airsim_ip', ''))
            return client
        else:
            # Local environment - normal connection
            import airsim
            return airsim.MultirotorClient()

    def _check_colab(self):
        """Detect if we're running in Colab environment without importing google.colab."""
        # Check for typical Colab environment indicators
        return os.path.exists('/content') and 'COLAB_GPU' in os.environ
    
    def _load_config(self):
        """Load appropriate configuration based on environment."""
        if self.is_colab:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'cloud_config.json'))
        else:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'local_config.json'))
        
        
        # Debugging: Print working directory
        print("Current working directory (training_manager):", os.getcwd())
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
        """Load appropriate configuration based on environment."""
        # Since we know the Docker container structure, use absolute paths
        # config_filename = 'cloud_config.json' if self.is_colab else 'local_config.json'
        # config_path = f'/app/configs/{config_filename}'
        
        # if not os.path.exists(config_path):
        #     raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # with open(config_path, 'r') as f:
        #     return json.load(f)

    def get_model_path(self):
        """Get the appropriate path for saving/loading models."""
        base_path = self.config['paths']['model_weights']
        if self.is_colab:
            # In Colab, we'll save to Google Drive
            from google.cloud import drive
            drive.mount('/content/drive')
            return f"/content/drive/MyDrive/{base_path}"
        return base_path
    
    def save_checkpoint(self, model, metrics, episode):
        """Save model checkpoint with metadata."""
        checkpoint_dir = Path(self.get_model_path()) / self.experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_dir / f"checkpoint_{episode}.h5"
        model.save_weights(str(model_path))
        
        # Save training metadata
        metadata = {
            'episode': episode,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'environment': 'colab' if self.is_colab else 'local'
        }
        
        with open(checkpoint_dir / f"metadata_{episode}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_latest_checkpoint(self, model):
        """Load the latest checkpoint for the model."""
        checkpoint_dir = Path(self.get_model_path())
        if not checkpoint_dir.exists():
            return None, 0
        
        # Find latest experiment
        experiments = sorted(checkpoint_dir.glob("*"))
        if not experiments:
            return None, 0
            
        latest_exp = experiments[-1]
        checkpoints = sorted(latest_exp.glob("checkpoint_*.h5"))
        if not checkpoints:
            return None, 0
            
        latest_checkpoint = checkpoints[-1]
        episode = int(latest_checkpoint.stem.split('_')[1])
        
        model.load_weights(str(latest_checkpoint))
        return model, episode