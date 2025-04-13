import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# class ForestNavCNN(BaseFeaturesExtractor):
#     """
#     Custom CNN policy network for forest navigation using a residual architecture.
#     This network extracts features from image observations for reinforcement learning.
#     """
#     def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
#         super(ForestNavCNN, self).__init__(observation_space, features_dim)

#         # Get the number of input channels from observation space
#         input_channels = observation_space.shape[0]
        
#         self.cnn_layers = nn.Sequential(
#             # Initial convolutional layer
#             nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             # Residual blocks
#             ResidualBlock(32, 64),
#             nn.MaxPool2d(kernel_size=2),

#             ResidualBlock(64, 64),
#             nn.MaxPool2d(kernel_size=2),

#             ResidualBlock(64, 128),

#             # Final Convolution layer
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Fixed output channels to match BatchNorm
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )

#         # Calculate flattened size
#         with torch.no_grad():
#             sample = torch.as_tensor(observation_space.sample()[None]).float()
#             n_flatten = self.cnn_layers(sample).shape[1]

#         self.fc_layers = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )

#         # Initialize weights
#         self.apply(self.init_weights)
        
#         print(f"ForestNavCNN initialized: input_channels={input_channels}, flattened_size={n_flatten}, features_dim={features_dim}")

#     def init_weights(self, module):
#         """Initialize weights for the CNN layers using Xavier uniform initialization."""
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0.0)

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the CNN feature extractor.
        
#         Args:
#             observations: Input tensor of shape [batch_size, channels, height, width]
            
#         Returns:
#             Extracted features tensor
#         """
#         # Convert uint8 [0-255] to float32 [0-1] if needed
#         if observations.dtype == torch.uint8:
#             observations = observations.float() / 255.0
        
#         # Process through CNN layers
#         features = self.cnn_layers(observations)
        
#         # Process through fully connected layers
#         return self.fc_layers(features)


# class ResidualBlock(nn.Module):
#     """
#     Residual block for the CNN policy network.
#     Implements skip connections to help with gradient flow during training.
#     """
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()  # Fixed typo in super() call

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
        
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         """
#         Forward pass through the residual block.
        
#         Args:
#             x: Input tensor
            
#         Returns:
#             Output tensor after applying residual connection
#         """
#         # Main path
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))

#         # Shortcut connection (skip connection)
#         out += self.shortcut(x)

#         return self.relu(out)


# class SpatialAttention(nn.Module):
#     """
#     Spatial attention mechanism for the CNN policy network.
#     This helps the network focus on important spatial regions in the input.
#     """
#     def __init__(self, in_channels):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         """
#         Apply spatial attention to input features.
        
#         Args:
#             x: Input tensor
            
#         Returns:
#             Tensor with applied spatial attention
#         """
#         attention = self.sigmoid(self.conv(x))
#         return attention * x





class ForestNavMultiInputExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the AirSimForestEnv that processes multiple sensor inputs:
    - RGB images
    - Depth images
    - LiDAR point clouds
    - GPS/position data
    - IMU data
    - Barometer data
    - Distance sensor data
    
    Each input type has its own processing network, and outputs are concatenated.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Initialize parent with the combined features dimension
        super(ForestNavMultiInputExtractor, self).__init__(observation_space, features_dim)
        
        # Extract spaces from the Dict observation space
        self.observation_spaces = observation_space.spaces
        self.has_rgb = 'rgb' in self.observation_spaces
        self.has_depth = 'depth' in self.observation_spaces
        self.has_lidar = 'lidar' in self.observation_spaces
        self.has_gps = 'gps' in self.observation_spaces
        self.has_imu = 'imu' in self.observation_spaces
        self.has_barometer = 'barometer' in self.observation_spaces
        self.has_distance = 'distance' in self.observation_spaces
        
        # Network components for each sensor type
        self.extractors = {}
        self.feature_dims = {}
        
        # Define feature dimensions for each input
        if self.has_rgb:
            rgb_dim = 128
            self.feature_dims['rgb'] = rgb_dim
            self.extractors['rgb'] = self._build_cnn_extractor(
                self.observation_spaces['rgb'], 
                rgb_dim
            )
            
        if self.has_depth:
            depth_dim = 64
            self.feature_dims['depth'] = depth_dim
            self.extractors['depth'] = self._build_cnn_extractor(
                self.observation_spaces['depth'], 
                depth_dim
            )
            
        if self.has_lidar:
            lidar_dim = 128
            self.feature_dims['lidar'] = lidar_dim
            self.extractors['lidar'] = self._build_lidar_extractor(
                self.observation_spaces['lidar'], 
                lidar_dim
            )
            
        # if self.has_gps:
        #     gps_dim = 16
        #     self.feature_dims['gps'] = gps_dim
        #     self.extractors['gps'] = self._build_vector_extractor(
        #         self.observation_spaces['gps'], 
        #         gps_dim
        #     )
            
        if self.has_imu:
            imu_dim = 32
            self.feature_dims['imu'] = imu_dim
            self.extractors['imu'] = self._build_vector_extractor(
                self.observation_spaces['imu'], 
                imu_dim
            )
            
        # if self.has_barometer:
        #     barometer_dim = 8
        #     self.feature_dims['barometer'] = barometer_dim
        #     self.extractors['barometer'] = self._build_vector_extractor(
        #         self.observation_spaces['barometer'], 
        #         barometer_dim
        #     )
            
        if self.has_distance:
            distance_dim = 24
            self.feature_dims['distance'] = distance_dim
            self.extractors['distance'] = self._build_vector_extractor(
                self.observation_spaces['distance'], 
                distance_dim
            )
        
        # Calculate total features dimension from all extractors
        total_concat_size = sum(self.feature_dims.values())
        
        # Final feature combination network
        self.combination_layer = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )
        
        print(f"ForestNavMultiInputExtractor initialized with:")
        print(f"  Feature dimensions: {self.feature_dims}")
        print(f"  Total concat size: {total_concat_size}")
        print(f"  Final features dim: {features_dim}")
    
    def _build_cnn_extractor(self, observation_space, features_dim):
        """
        Build a CNN for processing image inputs (RGB or depth)
        """
        n_input_channels = observation_space.shape[0]
        
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = cnn(sample).shape[1]
        
        linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        return nn.Sequential(cnn, linear)
    
    def _build_lidar_extractor(self, observation_space, features_dim):
        """
        Build a network for processing LiDAR point cloud data
        """
        # Calculate input size (flattened LiDAR points)
        n_points = observation_space.shape[0]
        n_features_per_point = observation_space.shape[1]  # Typically 3 for x,y,z
        input_dim = n_points * n_features_per_point

        max_points = 1024
        if n_points > max_points:
            self.downsample_lidar = True
            input_dim = max_points * n_features_per_point
        else:
            self.downsample_lidar = False
        
        # Use 1D convolutions to process points as a sequence
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def _build_vector_extractor(self, observation_space, features_dim):
        """
        Build a network for processing vector inputs (GPS, IMU, etc.)
        """
        input_dim = int(np.prod(observation_space.shape))
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
    
    def _maybe_downsample_lidar(self, lider_data):
        """
        Downsample LiDAR data point cloud if needed
        """

        if not hasattr(self, 'downsample_lidar') or not self.downsample_lidar:
            return lider_data
        
        batch_size = lider_data.shape[0]
        n_points = lider_data.shape[1]
        n_features = lider_data.shape[2]
        max_points = 1024

        if n_points <= max_points:
            return lider_data
        
        indices = torch.linspace(0, n_points - 1, max_points).long()
        return lider_data[:, indices, :]
    
    def forward(self, observations):
        """
        Process all enabled sensor inputs and combine their features
        """
        encoded_tensor_list = []
        
        try:
            # Process each observation type with its corresponding network
            if self.has_rgb and 'rgb' in observations:
                rgb_observations = observations['rgb']
                
                # Ensure RGB data is valid before processing
                if rgb_observations.numel() > 0:
                    # Convert uint8 [0-255] to float32 [0-1] if needed
                    if rgb_observations.dtype == torch.uint8:
                        rgb_observations = rgb_observations.float() / 255.0
                    encoded_rgb = self.extractors['rgb'](rgb_observations)
                    encoded_tensor_list.append(encoded_rgb)
                else:
                    # Handle empty RGB tensor by creating a zero tensor of expected shape
                    batch_size = next(iter(observations.values())).shape[0]
                    encoded_rgb = torch.zeros((batch_size, self.feature_dims['rgb']), 
                                            device=next(iter(observations.values())).device)
                    encoded_tensor_list.append(encoded_rgb)
            
            if self.has_depth and 'depth' in observations:
                depth_observations = observations['depth']
                
                # Ensure depth data is valid
                if depth_observations.numel() > 0:
                    # Convert uint8 [0-255] to float32 [0-1] if needed
                    if depth_observations.dtype == torch.uint8:
                        depth_observations = depth_observations.float() / 255.0
                    encoded_depth = self.extractors['depth'](depth_observations)
                    encoded_tensor_list.append(encoded_depth)
                else:
                    # Handle empty depth tensor
                    batch_size = next(iter(observations.values())).shape[0]
                    encoded_depth = torch.zeros((batch_size, self.feature_dims['depth']), 
                                              device=next(iter(observations.values())).device)
                    encoded_tensor_list.append(encoded_depth)
            
            if self.has_lidar and 'lidar' in observations:
                lidar_data = observations['lidar']
                # Downsample LiDAR data if configured
                lidar_data = self._maybe_downsample_lidar(lidar_data)
                encoded_lidar = self.extractors['lidar'](lidar_data)
                encoded_tensor_list.append(encoded_lidar)
            
            if self.has_imu and 'imu' in observations:
                encoded_imu = self.extractors['imu'](observations['imu'])
                encoded_tensor_list.append(encoded_imu)
            
            if self.has_distance and 'distance' in observations:
                encoded_distance = self.extractors['distance'](observations['distance'])
                encoded_tensor_list.append(encoded_distance)
            
            # Concatenate all encoded tensors
            combined_features = torch.cat(encoded_tensor_list, dim=1)
            
            # Final combination layer
            return self.combination_layer(combined_features)
            
        except Exception as e:
            print(f"Error in feature extractor forward pass: {e}")
            # Return emergency fallback output
            batch_size = 1
            for obs in observations.values():
                if obs is not None and hasattr(obs, 'shape') and len(obs.shape) > 0:
                    batch_size = obs.shape[0]
                    break
            
            # Create zero tensor of expected output shape
            device = next(self.parameters()).device
            return torch.zeros((batch_size, self.features_dim), device=device)



# Original CNN extractor updated to handle different image types
class ForestNavCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for image observations (RGB or depth).
    This is a simpler version if you're only using one image type.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(ForestNavCNN, self).__init__(observation_space, features_dim)
        
        # Get the number of input channels
        if len(observation_space.shape) == 3:
            # Image observation (channel, height, width)
            n_input_channels = observation_space.shape[0]
        else:
            # Fallback to 3 channels if shape is unexpected
            n_input_channels = 3
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
        print(f"ForestNavCNN initialized: input_channels={n_input_channels}, flattened_size={n_flatten}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert uint8 [0-255] to float32 [0-1] if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        
        # Process through CNN and linear layers
        return self.linear(self.cnn(observations))