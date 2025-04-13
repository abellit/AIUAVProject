import airsim
import cv2
import numpy as np
import keyboard
import random
import time
import os
import json

# Initialize AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Control parameters
VELOCITY = 50.0
YAW_RATE = 23.0  # Degrees per second
ACCELERATION = 2.5

# Ensure directories exist
def ensure_directories(directories):
    """Create necessary directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Define data storage directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '../..', 'data')
image_dir = os.path.join(base_dir, 'images')
#depth_dir = os.path.join(base_dir, 'depth')
lidar_dir = os.path.join(base_dir, 'lidar')
imu_dir = os.path.join(base_dir, 'imu')
label_dir = os.path.join(base_dir, 'labels')

train_image_dir = os.path.join(image_dir, 'train')
val_image_dir = os.path.join(image_dir, 'val')
train_label_dir = os.path.join(label_dir, 'train')
val_label_dir = os.path.join(label_dir, 'val')


# Create directories
ensure_directories([train_image_dir, val_image_dir, train_label_dir, val_label_dir, lidar_dir, imu_dir])

# Function to generate label for the image
def generate_label(image_name, cvat_annotations):
    """
    This functino will generate the label for the image. 
    For simplicity, we are assuming 'obstacle' class with ID 0.
    You can change the logic here based on your actual detection needs.
    """
    label_lines = []
    for annotation in cvat_annotations:
        class_id = annotation[0]  # The class ID for the object
        x_center = annotation[1]
        y_center = annotation[2]
        width = annotation[3]
        height = annotation[4]
        
        label_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return "\n".join(label_lines)

# Function to capture images
def capture_images():
    responses = client.simGetImages([
        # Front center RGB image
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
        # Downward RGB image
        #airsim.ImageRequest("downward", airsim.ImageType.Scene, False, False)
    ])
    timestamp = int(time.time() * 1000)

    # Process Front center RGB and Downward images
    if responses[0] and responses[0].height > 0:
        img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        # Decide whether to save in train or val folder (80% train, 20% val)
        if random.random() < 0.8:
            img_path = os.path.join(train_image_dir, f"rgb_{timestamp}.png")
            label_path = os.path.join(train_label_dir, f"rgb_{timestamp}.txt")
        else:
            img_path = os.path.join(val_image_dir, f"rgb_{timestamp}.png")
            label_path = os.path.join(val_label_dir, f"rgb_{timestamp}.txt")
        
        cv2.imwrite(img_path, img_rgb)
        with open(label_path, "w") as label_file:
            label = generate_label(f"rgb_{timestamp}", [])
            label_file.write(label)
        
    # if responses[1] and responses[1].height > 0:
    #     img_downward_rgb = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height, responses[1].width, 3)
    #     # Decide whether to save in train or val folder (80% train, 20% val)
    #     if random.random() < 0.8:
    #         img_path = os.path.join(train_image_dir, f"downward_rgb_{timestamp}.png")
    #         label_path = os.path.join(train_label_dir, f"downward_rgb_{timestamp}.txt")
    #     else:
    #         img_path = os.path.join(val_image_dir, f"downward_rgb_{timestamp}.png")
    #         label_path = os.path.join(val_label_dir, f"downward_rgb_{timestamp}.txt")
        
    #     cv2.imwrite(img_path, img_downward_rgb)
    #     with open(label_path, "w") as label_file:
    #         label = generate_label(f"downward_rgb_{timestamp}", [])
    #         label_file.write(label)

# Function to capture LiDAR and IMU data (optional for YOLO)
def capture_lidar_imu():
    lidar_data = client.getLidarData()
    imu_data = client.getImuData()
    timestamp = int(time.time() * 1000)

    # Save LiDAR data if needed (not directly used for YOLO)
    if lidar_data.point_cloud and len(lidar_data.point_cloud) > 0:
        np.save(os.path.join(lidar_dir, f"lidar_{timestamp}.npy"), np.array(lidar_data.point_cloud))

    # Save IMU data if needed (not directly used for YOLO)
    imu_dict = {
        "timestamp": timestamp,
        "angular_velocity": [imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val],
        "linear_acceleration": [imu_data.linear_acceleration.x_val, imu_data.linear_acceleration.y_val, imu_data.linear_acceleration.z_val],
        "orientation": [imu_data.orientation.x_val, imu_data.orientation.y_val, imu_data.orientation.z_val, imu_data.orientation.w_val]
    }
    with open(os.path.join(imu_dir, f"imu_data_{timestamp}.json"), "w") as f:
        json.dump(imu_dict, f)       

# Keyboard control loop
running = True
print("Start controlling the drone and collect data!")

# Define waypoints as (x, y, z) positions
waypoints = [
    (0, 0, -5), 
    (360, 0, -4), 
    (0, 70, -4), 
    (360, 70, -4),
    (0, 140, -3),
    (360, 140, -4),
    (0, 210, -4),
    (360, 210, -4),
    (0, 280, -5),
]

data_dir = "airsim_data"
os.makedirs(data_dir, exist_ok=True)

# Function to collect sensor data
def collect_data(index):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # RGB image
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)  # Depth image
    ])
    
    imu_data = client.getImuData()
    lidar_data = client.getLidarData()
    
    if responses[0].height > 0 and responses[0].width > 0:
        rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        depth_image = np.frombuffer(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width)
        
        airsim.write_png(os.path.join(data_dir, f"rgb_{index}.png"), rgb_image)
        np.save(os.path.join(data_dir, f"depth_{index}.npy"), depth_image)
    
    np.save(os.path.join(data_dir, f"imu_{index}.npy"), [imu_data.orientation.x_val, imu_data.orientation.y_val, imu_data.orientation.z_val])
    np.save(os.path.join(data_dir, f"lidar_{index}.npy"), lidar_data.point_cloud)

# Move to waypoints and collect data continuously
index = 0
for waypoint in waypoints:
    client.moveToPositionAsync(waypoint[0], waypoint[1], waypoint[2], 5).join()
    start_time = time.time()
    while time.time() - start_time < 2:  # Collect data for 2 seconds per waypoint
        capture_images()
        capture_lidar_imu()
        index += 1
        time.sleep(0.5)  # Collect data every 0.5 seconds

print("Data collection complete.")
