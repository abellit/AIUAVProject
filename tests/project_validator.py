import time
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.drone_control.main import EnhancedDroneNavigationSystem

nav = EnhancedDroneNavigationSystem()
start_time = time.time()  # Added missing start_time definition

# tests/quick_validation.py
def test_essential_functionality():
    # 1. Basic Navigation
    nav.takeoff()  # Added missing takeoff function call
    
    assert nav.client.getMultirotorState().landed_state == 1  # Verify takeoff
    
    # 2. Obstacle Detection
    obstacles = nav.detect_obstacles()
    assert len(obstacles) > 0  # Verify environment perception
    
    # 3. Path Planning
    nav.path = nav.plan_path((0,0,0), (50,50,-5), obstacles)
    assert len(nav.path) > 3  # Verify path generation
    
    # 4. Goal Reaching
    success = nav.navigate_in_forest((50,50))
    assert success == True  # Verify mission completion


def collect_core_metrics():
    path = nav.path  # Added missing path reference
    return {
        "navigation_time": time.time() - start_time,
        "path_length": sum(np.linalg.norm(np.array(p1)-np.array(p2)) for p1,p2 in zip(path[:-1], path[1:])),
        "collisions": nav.sensor_sync.collision_count,
        "replan_events": nav.path_replanning_count,
        "avg_processing_time": np.mean(nav.frame_processing_times)
    }