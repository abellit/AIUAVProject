import numpy as np
import math
import random
from queue import PriorityQueue

class Node:
    """Class representing a node in the RRT* tree"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # Cost from start to this node

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def distance(self, other):
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class RRTStar:
    """RRT* path planning algorithm implementation"""
    def __init__(self, start, goal, obstacles, step_size=3.0, max_iter=7000, 
                 search_radius=10.0, goal_sample_rate=0.1, map_limits=(-100, 100)):
        """
        Initialize RRT* planner
        
        Parameters:
        -----------
        start: tuple (x, y)
            Starting position
        goal: tuple (x, y)
            Goal position
        obstacles: set of tuples (x, y)
            Set of obstacle coordinates
        step_size: float
            Maximum step size between nodes
        max_iter: int
            Maximum number of iterations
        search_radius: float
            Radius to search for nearby nodes for rewiring
        goal_sample_rate: float
            Probability of sampling the goal position
        map_limits: tuple (min, max)
            Boundaries of the map
        """
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.min_x, self.max_x = map_limits
        self.min_y, self.max_y = map_limits
        
        # List to store all nodes in the tree
        self.nodes = [self.start_node]
        
        # Adjust search radius based on expected map size
        map_size = max(abs(self.max_x - self.min_x), abs(self.max_y - self.min_y))
        self.search_radius = min(self.search_radius, map_size * 0.1)
        
        # Collision margin for obstacle avoidance
        self.collision_margin = 1.0

    def random_sample(self):
        """Generate a random sample in the map space with goal biasing"""
        if random.random() < self.goal_sample_rate:
            return Node(self.goal_node.x, self.goal_node.y)
        else:
            # Random sample within map boundaries
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            return Node(x, y)

    def nearest_node(self, sampled_node):
        """Find the nearest node in the tree to the sampled node"""
        distances = [node.distance(sampled_node) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_node, to_node):
        """Steer from one node towards another with maximum step size"""
        dist = from_node.distance(to_node)
        
        if dist <= self.step_size:
            return to_node
        else:
            # Calculate the direction vector
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            
            # Create a new node in the direction with step_size distance
            new_x = from_node.x + self.step_size * math.cos(theta)
            new_y = from_node.y + self.step_size * math.sin(theta)
            
            new_node = Node(new_x, new_y)
            return new_node

    def is_collision_free(self, from_node, to_node):
        """Check if the path between two nodes is collision-free"""
        # For efficiency, first check if either endpoint is in collision
        if self.is_point_in_collision(to_node.x, to_node.y) or \
           self.is_point_in_collision(from_node.x, from_node.y):
            return False
        
        # Check along the path with discrete steps
        dist = from_node.distance(to_node)
        steps = max(int(dist / self.collision_margin), 5)  # At least 5 checks along the path
        
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            if self.is_point_in_collision(x, y):
                return False
                
        return True

    def is_point_in_collision(self, x, y):
        """Check if a point is in collision with any obstacle"""
        # Round coordinates to match obstacle grid
        point = (round(x), round(y))
        
        # Check if point is in obstacles set
        if point in self.obstacles:
            return True
            
        # Check within collision margin
        for dx in range(-int(self.collision_margin), int(self.collision_margin) + 1):
            for dy in range(-int(self.collision_margin), int(self.collision_margin) + 1):
                if (point[0] + dx, point[1] + dy) in self.obstacles:
                    return True
                    
        return False

    def near_nodes(self, node):
        """Find nodes that are within search_radius of the given node"""
        near_nodes = []
        
        for existing_node in self.nodes:
            if existing_node.distance(node) <= self.search_radius:
                near_nodes.append(existing_node)
                
        return near_nodes

    def calculate_cost(self, from_node, to_node):
        """Calculate the cost of the path from one node to another"""
        if not self.is_collision_free(from_node, to_node):
            return float('inf')
            
        return from_node.cost + from_node.distance(to_node)

    def choose_parent(self, new_node, near_nodes):
        """Choose the best parent for new_node from near_nodes"""
        if not near_nodes:
            return new_node
            
        min_cost = float('inf')
        best_parent = None
        
        for near_node in near_nodes:
            potential_cost = self.calculate_cost(near_node, new_node)
            
            if potential_cost < min_cost:
                min_cost = potential_cost
                best_parent = near_node
                
        if best_parent:
            new_node.parent = best_parent
            new_node.cost = min_cost
            
        return new_node

    def rewire(self, new_node, near_nodes):
        """Rewire the tree to optimize paths through the new node"""
        for near_node in near_nodes:
            # Calculate cost through new_node
            potential_cost = new_node.cost + new_node.distance(near_node)
            
            # If path through new_node is better, rewire
            if potential_cost < near_node.cost and self.is_collision_free(new_node, near_node):
                near_node.parent = new_node
                near_node.cost = potential_cost
                
                # Recursively update costs for all descendants
                self.update_descendants_cost(near_node)

    def update_descendants_cost(self, node):
        """Update costs for all descendants after rewiring"""
        # Find all immediate children of this node
        children = [n for n in self.nodes if n.parent == node]
        
        for child in children:
            # Update cost for child
            child.cost = node.cost + node.distance(child)
            # Recursively update descendants
            self.update_descendants_cost(child)

    def is_goal_reached(self, node):
        """Check if the goal is reached within a threshold"""
        return node.distance(self.goal_node) <= self.step_size

    def extract_path(self):
        """Extract the path from start to goal"""
        # Find the node closest to the goal
        goal_candidates = []
        for node in self.nodes:
            if self.is_goal_reached(node):
                goal_candidates.append((node.distance(self.goal_node), node))
                
        if not goal_candidates:
            return []  # No path found
            
        # Sort by distance to goal
        goal_candidates.sort(key=lambda x: x[0])
        closest_node = goal_candidates[0][1]
        
        # Backtrack to create the path
        path = []
        current = closest_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
            
        # Reverse to get start-to-goal order
        path.reverse()
        
        # Add exact goal position if not already in path
        if path and math.sqrt((path[-1][0] - self.goal_node.x)**2 + 
                              (path[-1][1] - self.goal_node.y)**2) > 0.1:
            path.append((self.goal_node.x, self.goal_node.y))
            
        return path

    def plan(self):
        """Main RRT* planning function"""
        for i in range(self.max_iter):
            # Sample a random node
            random_node = self.random_sample()
            
            # Find the nearest node in the tree
            nearest = self.nearest_node(random_node)
            
            # Steer towards the random node with max step size
            new_node = self.steer(nearest, random_node)
            
            # Check if path is collision-free
            if self.is_collision_free(nearest, new_node):
                # Find nearby nodes
                near_nodes = self.near_nodes(new_node)
                
                # Choose best parent
                new_node = self.choose_parent(new_node, near_nodes)
                
                # Add the node to the tree
                self.nodes.append(new_node)
                
                # Rewire the tree
                self.rewire(new_node, near_nodes)
                
                # Check if we've reached the goal
                if i % 10 == 0 and self.is_goal_reached(new_node):  # Check periodically to save computation
                    path = self.extract_path()
                    if path:
                        return self.post_process_path(path)
                        
        # If max iterations reached, try to extract a path anyway
        return self.post_process_path(self.extract_path())

    def post_process_path(self, path):
        """Post-process the path for smoothness and simplification"""
        if not path or len(path) <= 2:
            return path
            
        # Path shortcutting - try to connect non-adjacent nodes if collision-free
        i = 0
        while i < len(path) - 2:
            # Try to connect current node to nodes further ahead
            for j in range(len(path) - 1, i + 1, -1):
                node_i = Node(path[i][0], path[i][1])
                node_j = Node(path[j][0], path[j][1])
                
                # If direct path is collision-free, remove intermediate nodes
                if self.is_collision_free(node_i, node_j):
                    path = path[:i+1] + path[j:]
                    break
            i += 1
            
        # Simple path smoothing
        if len(path) >= 3:
            smoothed_path = [path[0]]
            for i in range(1, len(path) - 1):
                # Apply a simple moving average
                smoothed_x = (path[i-1][0] + path[i][0] + path[i+1][0]) / 3
                smoothed_y = (path[i-1][1] + path[i][1] + path[i+1][1]) / 3
                
                # Check if smoothed point is collision-free
                if not self.is_point_in_collision(smoothed_x, smoothed_y):
                    smoothed_path.append((smoothed_x, smoothed_y))
                else:
                    smoothed_path.append(path[i])
            smoothed_path.append(path[-1])
            return smoothed_path
            
        return path

# For testing the RRT* implementation independently
if __name__ == "__main__":
    # Example usage
    start = (0, 0)
    goal = (50, 50)
    
    # Create some sample obstacles
    obstacles = set()
    for i in range(20, 40):
        obstacles.add((i, 25))
        obstacles.add((25, i))
    
    # Plan path
    rrt_star = RRTStar(start, goal, obstacles)
    path = rrt_star.plan()
    
    print(f"Path found with {len(path)} waypoints")
    for point in path:
        print(point)
    
    # Visualization code would go here if needed
    # This could be done with matplotlib or another visualization library