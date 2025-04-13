import airsim
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO

pre_trained_model = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\model\TrainedWeights\best_yolov10n_obstacle.pt'

def get_image(client):
    """
    Get the image from the camera of the specified vehicle.
    """
    try:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        
        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        
        # Check if image is valid
        if img_rgb.size == 0 or img_rgb.shape[0] == 0 or img_rgb.shape[1] == 0:
            print("Warning: Received empty image")
            return None
            
        return img_rgb
    except Exception as e:
        print(f"Error getting image: {e}")
        return None

def get_lidar_data(client):
    lidar_data = client.getLidarData()
    if len(lidar_data.point_cloud) < 3:
        return None
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    return points

def get_distance_sensor_data(client):
    """
    Retrieves distance sensor data from the drone.
    """
    try:
        distance_data = client.getDistanceSensorData(vehicle_name="Drone", distance_sensor_name="Distance")
        if distance_data is not None and distance_data.distance != float('inf'):
            return distance_data.distance  # in meters
        else:
            return None
    except Exception as e:
        print(f"Error getting distance sensor data: {e}")
        return None

def yolo_inference(model, image):
    """
    Perform YOLO inference on the image.
    """
    results = model(image)
    return results[0]

def process_detections(detections, lidar_points):
    detected_objects = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = detection
        label = detections.names[int(cls)]
        depth = np.mean(lidar_points[:, 0]) if lidar_points is not None else None
        detected_objects.append({
            "label": label,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "depth": depth
        })
    return detected_objects

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    model = YOLO(pre_trained_model)  # Load the trained YOLO model
    model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference

    try:

        while True:
            try:

                # Get the image from the camera
                image = get_image(client)
                image = image.copy()  # Make a copy of the image to avoid modifying the original
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                
                if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                    print("Warning: Received empty image")
                    time.sleep(0.1)
                    continue
                # Get the LiDAR data
                lidar_points = get_lidar_data(client)

                # Get distance sensor reading
                distance = get_distance_sensor_data(client)
                if distance is not None:
                    print(f"Distance Sensor Reading: {distance:.2f} meters")
                else:
                    print("Distance Sensor Reading: Not available")

                # Perform YOLO inference
                detections = yolo_inference(model, image)

                # Process the detections and LiDAR data
                detected_objects = process_detections(detections, lidar_points)

                # Print the detected objects
                for obj in detected_objects:
                    print(f"Detected {obj['label']} at depth: {obj['depth']} meters")
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, obj['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Detected Objects", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)

            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(0.5)
        
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()