import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
class ForestNavCNN(BaseFeaturesExtractor):
    """
    Custom CNN policy network using a pre-trained Faster R-CNN backbone as feature extractor.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int =256): # Add backbone_layer_name
        super(ForestNavCNN, self).__init__(observation_space, features_dim)

        input_shape = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )


        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]   
        # self.cnn = tf.models.Sequential([
        #     tf.layers.Conv2D(32, kernel_size=3, strides=4, padding='same', activation='relu', input_shape=input_shape),
        #     tf.layers.MaxPooling2D(pool_size=2, strides=2),

        #     tf.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        #     tf.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        #     tf.layers.MaxPooling2D(pool_size=2, strides=2),

        #     tf.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        #     tf.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        #     tf.layers.MaxPooling2D(pool_size=2, strides=2),

        #     tf.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu'),
        #     tf.layers.GlobalAveragePooling2D()
        # ])

        self.linear = nn.Sequential(  
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def build_tf_model(self, n_input_channels, features_dim):
        self.tf_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, padding='same', activation='relu', input_shape=(64, 64, n_input_channels)),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(features_dim, activation='relu')
        ])

        self.tf_model.compile(optimizer='adam', loss='mse')

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
         # Convert uint8 [0-255] to float32 [0-1] if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        
        # Process through CNN and linear layers
        return self.linear(self.cnn(observations))

    def tf_process(self, observations_np):
        if isinstance(observations_np, torch.Tensor):
            observations_np = observations_np.permute(0, 2, 3, 1).cpu().numpy()
        elif isinstance(observations_np, np.ndarray) and observations_np.shape[1] <= 3:
            observations_np = observations_np.permute(0, 2, 3, 1)

        return self.tf_model(observations_np)
    


#  class PretrainedFasterRCNNFeatureExtractor(tf.keras.Model): # Renamed class for clarity
#     """
#     Defines the pre-trained Faster R-CNN backbone feature extractor and trainable MLP layers on top.
#     """
#     def __init__(self, features_dim: int, pretrained_model_path: str, backbone_layer_name: str): # Add backbone_layer_name
#         super(PretrainedFasterRCNNFeatureExtractor, self).__init__()
#         self.features_dim = features_dim

#         # 1. Load the pre-trained Faster R-CNN model
#         pretrained_cnn = tf.keras.models.load_model(pretrained_model_path)

#         # 2. Create feature extractor model from the backbone layer
#         self.feature_extractor = tf.keras.Model(
#             inputs=pretrained_cnn.input,
#             outputs=pretrained_cnn.get_layer(backbone_layer_name).output # Use backbone_layer_name here
#         )

#         # 3. Freeze the weights of the pre-trained feature extractor layers
#         for layer in self.feature_extractor.layers:
#             layer.trainable = False

#         # 4. Define new, trainable layers on top of the feature extractor
#         self.flatten = tf.keras.layers.Flatten()
#         self.fc1 = tf.keras.layers.Dense(256, activation='relu')
#         self.fc_final = tf.keras.layers.Dense(features_dim, activation=None) # Output layer to features_dim

#     def call(self, observations):
#         """Forward pass: pre-trained Faster R-CNN feature extraction + trainable MLP."""
#         pretrained_features = self.feature_extractor(observations) # Get features from Faster R-CNN backbone
#         flattened_features = self.flatten(pretrained_features)
#         intermediate_representation = self.fc1(flattened_features)
#         features = self.fc_final(intermediate_representation)
#         return features

