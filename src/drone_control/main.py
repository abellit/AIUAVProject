# import airsim
# import time
# import sys
# import os
# from datetime import datetime


# class ImageCapture:
#     """Handles image capture and storage for the drone."""

#     def __init__(self):
#         # Create a directory structure for storing images
#         self.base_dir = os.path.join(
#             os.path.dirname(os.path.abspath(__file__)), 'data')
#         self.image_dir = os.path.join(self.base_dir, 'images')
#         self.depth_dir = os.path.join(self.base_dir, 'depth')
#         self._ensure_directories()

#     def _ensure_directories(self):
#         """Create necessary directories if they don't exist."""
#         for directory in [self.image_dir, self.depth_dir]:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
#                 print(f"Created directory: {directory}")

    # def capture_and_save_images(self, client):
    #     """Capture and save both regular and depth images."""
    #     try:
    #         # Generate timestamp for unique filenames
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #         # Request images from AirSim
    #         responses = client.simGetImages([
    #             airsim.ImageRequest(
    #                 "0", airsim.ImageType.Scene),  # Regular camera
    #             # Depth visualization
    #             airsim.ImageRequest("1", airsim.ImageType.DepthVis)
    #         ])

    #         print(f"Retrieved {len(responses)} images")

    #         # Save each image with appropriate naming
    #         for idx, response in enumerate(responses):
    #             if idx == 0:  # Regular image
    #                 filename = os.path.join(
    #                     self.image_dir, f'scene_{timestamp}.png')
    #             else:  # Depth image
    #                 filename = os.path.join(
    #                     self.depth_dir, f'depth_{timestamp}.png')

    #             # Save the image
    #             airsim.write_file(filename, response.image_data_uint8)
    #             print(f"Saved image to: {filename}")

    #         return True

    #     except Exception as e:
    #         print(f"Error capturing images: {str(e)}")
    #         return False


# def wait_for_connection(client, timeout=30):
#     """
#     Wait for the vehicle to establish a stable connection.
#     Returns True if connection is established, False if timeout occurs.
#     """
#     print("Waiting for vehicle connection...")
#     start_time = time.time()

#     while time.time() - start_time < timeout:
#         try:
#             # Try to get vehicle state
#             state = client.getMultirotorState()
#             print("Vehicle connection established!")
#             return True
#         except:
#             print("Waiting for vehicle to respond...")
#             time.sleep(2)

#     return False


# def initialize_drone():
#     """
#     Initialize the drone with robust connection handling and state verification.
#     """
#     # First, print important setup information
#     print("\nPrerequisites check:")
#     print("1. PX4 SITL should be running")
#     print("2. Unreal Engine simulator should be running")
#     print("3. Network ports (4560, 14540) should be available")

#     try:
#         # Create the AirSim client
#         client = airsim.MultirotorClient()
#         client.confirmConnection()
#         print("\nInitial connection to AirSim established...")

#         # Wait for vehicle to be ready
#         if not wait_for_connection(client):
#             print("Failed to establish stable vehicle connection!")
#             return None

#         print("\nStarting initialization sequence...")

#         # Wait for initial setup
#         time.sleep(3)

#         try:
#             # Try to get initial state
#             state = client.getMultirotorState()
#             print(f"\nInitial vehicle state:")
#             print(f"Position: x={state.kinematics_estimated.position.x_val:.2f}, "
#                   f"y={state.kinematics_estimated.position.y_val:.2f}, "
#                   f"z={state.kinematics_estimated.position.z_val:.2f}")

#             # Enable API control with retry
#             max_retries = 3
#             for attempt in range(max_retries):
#                 try:
#                     client.enableApiControl(True)
#                     print("API Control enabled successfully!")
#                     break
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         raise Exception(
#                             "Failed to enable API control after multiple attempts")
#                     print(
#                         f"Attempt {attempt + 1} to enable API control failed, retrying...")
#                     time.sleep(2)

#             # Final connection verification
#             if client.isApiControlEnabled():
#                 print("\nDrone initialized successfully!")
#                 return client
#             else:
#                 print("\nFailed to enable API control")
#                 return None

#         except Exception as e:
#             print(f"\nError during vehicle initialization: {str(e)}")
#             return None

#     except Exception as e:
#         print(f"\nError during connection: {str(e)}")
#         return None


# def wait_for_gps(client, timeout=30):
#     """
#     Wait for the drone to receive a valid GPS signal.

#     Args:
#         client: AirSim client object
#         timeout: Maximum time to wait in seconds

#     Returns:
#         bool: True if GPS is initialized, False if timeout occurred
#     """
#     print("Waiting for GPS initialization...")
#     start_time = time.time()

#     while time.time() - start_time < timeout:
#         gps_data = client.getGpsData()
#         # In PX4, a valid GPS fix is indicated by a non-zero position
#         if (abs(gps_data.gnss.geo_point.latitude) > 0.0001 and
#                 abs(gps_data.gnss.geo_point.longitude) > 0.0001):
#             print("GPS initialization complete!")
#             return True
#         time.sleep(1)
#         print("Still waiting for GPS...", end='\r')

#     print("\nGPS initialization timeout!")
#     return False


# def execute_flight_pattern(client, image_capturer):
#     """Execute a square flight pattern with image capture at waypoints."""
#     try:
#         # Wait for GPS before attempting takeoff
#         if not wait_for_gps(client):
#             print("Cannot proceed with flight - GPS not initialized")
#             return False

#         # Take off
#         print("\nInitiating takeoff sequence...")
#         client.takeoffAsync().join()
#         time.sleep(5)
#         print("Takeoff complete")

#         # Capture images at starting position
#         print("\nCapturing initial position images...")
#         image_capturer.capture_and_save_images(client)

#         # Define waypoints for square pattern
#         waypoints = [
#             (7, 0, -5),    # Forward
#             (7, 23, -5),   # Right
#             (0, 23, -5),   # Backward
#             (0, 0, -5)     # Return to start
#         ]

#         # Execute flight pattern with image capture at each waypoint
#         for idx, (x, y, z) in enumerate(waypoints):
#             print(f"\nMoving to waypoint {idx + 1}...")
#             client.moveToPositionAsync(x, y, z, 10).join()
#             time.sleep(2)  # Stabilization time

#             print(f"Capturing images at waypoint {idx + 1}...")
#             image_capturer.capture_and_save_images(client)

#         # Final landing sequence
#         print("\nInitiating landing sequence...")
#         client.landAsync().join()
#         print("Landing complete")

#         return True

#     except Exception as e:
#         print(f"Error during flight pattern: {str(e)}")
#         # Emergency landing
#         client.landAsync().join()
#         return False


# def main():
#     """Main execution function with enhanced error handling."""
#     try:
#         print("Starting drone connection sequence...")
#         client = initialize_drone()  # Your existing initialize_drone function

#         if client:
#             print("\nConnection and initialization successful!")

#             # Initialize image capture system
#             image_capturer = ImageCapture()

#             while True:
#                 print("\n1. Start drone flight mode")
#                 print("2. Exit drone flight mode")

#                 try:
#                     user_choice = int(input("\nSelect an option: "))

#                     if user_choice == 1:
#                         execute_flight_pattern(client, image_capturer)
#                     elif user_choice == 2:
#                         print("\nExiting flight mode...")
#                         client.armDisarm(False)
#                         break
#                     else:
#                         print("\nInvalid option. Please select 1 or 2.")

#                 except ValueError:
#                     print("\nPlease enter a valid number.")

#         else:
#             print("\nFailed to initialize drone. Please check your simulation setup.")

#     except KeyboardInterrupt:
#         print("\n\nOperation interrupted by user.")
#     except Exception as e:
#         print(f"\nUnexpected error: {str(e)}")
#     finally:
#         if 'client' in locals():
#             client.armDisarm(False)
#             print("\nDrone disarmed. Exiting program.")


# if __name__ == "__main__":
#     main()

import airsim
import time
import sys
import os
from datetime import datetime
import asyncio

class ImageCapture:
    """Handles image capture and storage for the drone."""

    def __init__(self):
        # Create a directory structure for storing images relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_dir, '../..', 'data')
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.depth_dir = os.path.join(self.base_dir, 'depth')
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.image_dir, self.depth_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def capture_and_save_images(self, client):
        """Capture and save both regular and depth images."""
        try:
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Request images from AirSim using camera names from settings.json
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),  # Main camera
                airsim.ImageRequest("1", airsim.ImageType.DepthVis)  # Depth camera
            ])

            print(f"Retrieved {len(responses)} images")

            # Save each image with appropriate naming
            for idx, response in enumerate(responses):
                if idx == 0:  # Regular image
                    filename = os.path.join(self.image_dir, f'scene_{timestamp}.png')
                else:  # Depth image
                    filename = os.path.join(self.depth_dir, f'depth_{timestamp}.png')

                # Save the image
                airsim.write_file(filename, response.image_data_uint8)
                print(f"Saved image to: {filename}")

            return True

        except Exception as e:
            print(f"Error capturing images: {str(e)}")
            return False

def initialize_drone(ip_address=''):
    """
    Initialize the drone with robust connection handling and state verification.
    
    Args:
        ip_address (str): Optional IP address for the AirSim server
    """
    print("\nPrerequisites check:")
    print("1. Unreal Engine simulator should be running")
    print("2. AirSim plugin should be enabled")
    print(f"3. Attempting connection to AirSim{f' at {ip_address}' if ip_address else ''}")

    try:
        # Create the AirSim client with optional IP address
        client = airsim.MultirotorClient(ip=ip_address) if ip_address else airsim.MultirotorClient()
        client.confirmConnection()
        print("\nInitial connection to AirSim established...")

        # Get initial state
        state = client.getMultirotorState()
        print(f"\nInitial vehicle state:")
        print(f"Position: x={state.kinematics_estimated.position.x_val:.2f}, "
              f"y={state.kinematics_estimated.position.y_val:.2f}, "
              f"z={state.kinematics_estimated.position.z_val:.2f}")

        # Enable API control
        client.enableApiControl(True)
        print("API Control enabled successfully!")

        # Arm the drone
        client.armDisarm(True)
        print("Drone armed successfully!")

        return client

    except Exception as e:
        print(f"\nError during connection: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if Unreal Engine is running")
        print("2. Verify AirSim settings.json is configured correctly")
        print("3. Try connecting with explicit IP (host.docker.internal or your host IP)")
        return None

async def capture_continuous_images(client, image_capturer, stop_event):
    """Continuously capture images every 0.2 seconds until stop_event is set."""
    while not stop_event.is_set():
        image_capturer.capture_and_save_images(client)
        await asyncio.sleep(0.2)  # Capture every 0.2 seconds

async def execute_flight_pattern(client, image_capturer):
    """Execute a square flight pattern with continuous image capture."""
    stop_event = asyncio.Event()  # Event to stop image capture thread
    capture_task = asyncio.create_task(capture_continuous_images(client, image_capturer, stop_event))
    
    try:
        print("\nInitiating takeoff sequence...")
        await client.takeoffAsync()
        await asyncio.sleep(5)  # Give more time for stabilization
        print("Takeoff complete")

        await client.hoverAsync()
        await asyncio.sleep(2)

        print("\nStarting continuous image capture...")
#      capture_thread.start()  # Start continuous image capture

        waypoints = [
            (7, 0, -3),
            (7, 23, -3),
            (0, 23, -3),
            (0, 0, -3)
        ]

        for idx, (x, y, z) in enumerate(waypoints):
            print(f"\nMoving to waypoint {idx + 1}...")
            await client.moveToPositionAsync(x, y, z, 2)
            await asyncio.sleep(3)  # More stabilization time

        print("\nReturning to home position...")
        await client.goHomeAsync()
        await asyncio.sleep(3)

        print("\nInitiating landing sequence...")
        await client.landAsync()
        print("Landing complete")

    except Exception as e:
        print(f"Error during flight pattern: {str(e)}")
        try:
            await client.landAsync()
        except:
            print("Emergency landing failed, attempting to disarm")
            client.armDisarm(False)
    
    finally:
        print("\nStopping continuous image capture...")
        stop_event.set()  # Stop image capture thread
        await capture_task
        print("Image capture stopped.")

    return True

async def main():
    """Main execution function with enhanced error handling."""
    client = None
    try:
        print("Starting drone connection sequence...")
        
        # Try different connection methods
        connection_attempts = [
            ('', 'local connection'),
            ('host.docker.internal', 'Docker host'),
            ('172.17.0.1', 'Docker bridge network')
        ]

        for ip, description in connection_attempts:
            print(f"\nAttempting {description}...")
            client = initialize_drone(ip)
            if client:
                break
            time.sleep(2)

        if client:
            print("\nConnection and initialization successful!")

            # Initialize image capture system
            image_capturer = ImageCapture()

            while True:
                print("\n1. Start drone flight mode")
                print("2. Exit drone flight mode")

                try:
                    user_choice = int(input("\nSelect an option: "))

                    if user_choice == 1:
                        await execute_flight_pattern(client, image_capturer)
                    elif user_choice == 2:
                        print("\nExiting flight mode...")
                        break
                    else:
                        print("\nInvalid option. Please select 1 or 2.")

                except ValueError:
                    print("\nPlease enter a valid number.")

        else:
            print("\nFailed to initialize drone. Please check your simulation setup.")

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        if client:
            try:
                client.armDisarm(False)
                print("\nDrone disarmed. Exiting program.")
            except:
                print("\nFailed to disarm drone. Please check simulator status.")

if __name__ == "__main__":
    asyncio.run(main())
