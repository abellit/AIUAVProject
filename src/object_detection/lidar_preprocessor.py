import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

class LidarProcessor:
    def __init__(self, voxel_size=0.5, fps_samples=1024):
        self.voxel_size = voxel_size
        self.fps_samples = fps_samples
        
    def process_lidar_data(self, points):
        """
        Process raw LiDAR point cloud with voxelization and FPS
        
        Args:
            points: Nx3 numpy array of point cloud data
            
        Returns:
            processed_points: Downsampled and processed point cloud
            voxel_grid: Voxelized representation of the point cloud
        """
        if len(points) < 3:
            return np.array([]), None
            
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Voxelization - downsample using voxel grid
        voxel_grid = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Convert back to numpy array
        voxel_points = np.asarray(voxel_grid.points)
        
        # Apply Farthest Point Sampling if we have enough points
        if len(voxel_points) > self.fps_samples:
            fps_points = self.farthest_point_sampling(voxel_points, self.fps_samples)
        else:
            fps_points = voxel_points
            
        return fps_points, voxel_grid
    
    def farthest_point_sampling(self, points, npoint):
        """
        Farthest point sampling algorithm to select representative points
        
        Args:
            points: Nx3 numpy array of points
            npoint: Number of points to sample
            
        Returns:
            sampled_points: npoint x 3 array of sampled points
        """
        N, D = points.shape
        centroids = np.zeros((npoint, D))
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(npoint):
            centroids[i] = points[farthest]
            dist = np.sum((points - centroids[i]) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
            
        return centroids
    
    def save_point_cloud(self, points, filename):
        """Save point cloud to file"""
        np.save(filename, points)
        
    def load_point_cloud(self, filename):
        """Load point cloud from file"""
        return np.load(filename)
    
    def visualize_point_cloud(self, points):
        """Visualize point cloud using Open3D"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
        
    def create_voxel_grid_feature(self, points, grid_size=32):
        """
        Create a 3D voxel grid feature representation
        
        Args:
            points: Nx3 numpy array of points
            grid_size: Size of the voxel grid (grid_size x grid_size x grid_size)
            
        Returns:
            voxel_grid: 3D numpy array representing the voxel grid occupancy
        """
        # Normalize points to [0, 1]
        if len(points) == 0:
            return np.zeros((grid_size, grid_size, grid_size))
            
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        points_normalized = (points - min_bound) / (max_bound - min_bound + 1e-8)
        
        # Create empty voxel grid
        voxel_grid = np.zeros((grid_size, grid_size, grid_size))
        
        # Fill voxel grid
        for point in points_normalized:
            x = min(int(point[0] * (grid_size-1)), grid_size-1)
            y = min(int(point[1] * (grid_size-1)), grid_size-1)
            z = min(int(point[2] * (grid_size-1)), grid_size-1)
            voxel_grid[x, y, z] = 1
            
        return voxel_grid