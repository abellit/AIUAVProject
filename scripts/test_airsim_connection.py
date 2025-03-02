# test_airsim_connection.py
import airsim
import os
import time

def test_connection(host_ip='host.docker.internal'):
    """
    Test connection to AirSim running on host machine.
    
    Args:
        host_ip (str): IP address of the host machine running AirSim
    """
    try:
        # Connect to AirSim using host IP
        client = airsim.MultirotorClient(ip=host_ip)
        print(f"Attempting to connect to AirSim at {host_ip}")
        
        # Test connection
        client.confirmConnection()
        print("Successfully connected to AirSim!")
        
        # Get drone state
        state = client.getMultirotorState()
        print("\nDrone State:")
        print(f"Position: x={state.kinematics_estimated.position.x_val:.2f}, "
              f"y={state.kinematics_estimated.position.y_val:.2f}, "
              f"z={state.kinematics_estimated.position.z_val:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\nError connecting to AirSim: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Unreal Engine is running with AirSim")
        print("2. Check if AirSim settings.json is configured correctly")
        print("3. Verify the host IP address is correct")
        print("4. Ensure port 41451 is accessible")
        return False

if __name__ == "__main__":
    # Try to connect using host.docker.internal first
    if not test_connection():
        # If that fails, try localhost
        print("\nRetrying with localhost...")
        if not test_connection('localhost'):
            print("\nFinal attempt with host network IP...")
            # You might need to replace this with your actual host IP
            test_connection('192.168.192.209')