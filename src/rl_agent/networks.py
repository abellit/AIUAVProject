import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.policy import CnnPolicy
import tensorflow as tf


# class CustomCNN(CnnPolicy):
#     """
#     Custom CNN policy network using a pre-trained Faster R-CNN backbone as feature extractor.
#     """
#     def __init__(self, *args, pretrained_model_path=None, backbone_layer_name=None, **kwargs): # Add backbone_layer_name
#         super(CustomCNN, self).__init__(*args, **kwargs)
#         self.pretrained_model_path = pretrained_model_path
#         self.backbone_layer_name = backbone_layer_name # Store backbone layer name

#     def _build_mlp_extractor(self, **kwargs) -> tf.keras.Model:
#         """
#         Load pre-trained Faster R-CNN backbone and define MLP on top for PPO.
#         """
#         return PretrainedFasterRCNNFeatureExtractor(self.features_dim,
#                                                       pretrained_model_path=self.pretrained_model_path,
#                                                       backbone_layer_name=self.backbone_layer_name) # Pass layer name

# class PretrainedFasterRCNNFeatureExtractor(tf.keras.Model): # Renamed class for clarity
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

# if __name__ == '__main__': # Example usage
#     # Example usage for testing the CNN definition
#     env = DummyVecEnv([lambda: gym.make('CartPole-v1')]) # Replace with your AirSim env
#     env = VecTransposeImage(env) # If your environment returns images as channels_last, use this

#     # **Important:** Replace with actual paths and layer name
#     pretrained_model_path = '../model/mask_rcnn_coco.h5' # <--- REPLACE THIS
#     backbone_layer_name = 'resnet_block5_conv3' # <--- REPLACE THIS with the correct backbone layer name

#     model = PPO(CustomCNN, env,
#                 policy_kwargs={'pretrained_model_path': pretrained_model_path,
#                                'backbone_layer_name': backbone_layer_name}, # Pass path and layer name
#                 verbose=1, tensorboard_log="./ppo_cnn_pretrained_tensorboard/") # Use CustomCNN policy
#     model.learn(total_timesteps=10000) # Train for example timesteps