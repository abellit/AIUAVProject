import numpy as np
from rrt_star import RRTStar

class PathPlanner:
    def __init__(self):
        pass
    
    def plan_path(self, start, goal, obstacles):
        """Use RRT* to find an optimized path avoiding obstacles"""
        start_2d = (start[0], start[1])
        goal_2d = (goal[0], goal[1])
        
        # Create buffer around obstacles for safety
        buffered_obstacles = self._create_obstacle_buffer(obstacles)
        
        try:
            rrt_star = RRTStar(
                start_2d, 
                goal_2d, 
                buffered_obstacles, 
                step_size=3, 
                max_iter=1000, 
                search_radius=10
            )
            path = rrt_star.plan()
            
            if not path:
                print("Warning: No path found by RRT*")
                return []
                
            # Add current height to the path
            path_3d = [(p[0], p[1], start[2]) for p in path]
            return path_3d
        except Exception as e:
            print(f"Path planning error: {e}")
            return []
    
    def _create_obstacle_buffer(self, obstacles):
        """Create safety buffer around obstacles"""
        buffered_obstacles = set()
        for obs in obstacles:
            for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                for dy in range(-2, 3):
                    buffered_obstacles.add((obs[0] + dx, obs[1] + dy))
        return buffered_obstacles
    
    def smooth_path(self, path):
        """Apply path smoothing to reduce jerkiness"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            smoothed.append((
                (path[i-1][0] + path[i][0] + path[i+1][0]) / 3,
                (path[i-1][1] + path[i][1] + path[i+1][1]) / 3,
                path[i][2]  # Keep original height
            ))
        smoothed.append(path[-1])
        return smoothed