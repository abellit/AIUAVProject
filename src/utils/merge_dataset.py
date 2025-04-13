import kagglehub

# Download latest version
path = kagglehub.dataset_download("sentinel3734/tree-detection-lidar-rgb")

print("Path to dataset files:", path)