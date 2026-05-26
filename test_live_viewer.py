import sys
import numpy as np

# Use glfw backend for maximum compatibility on ARM/Linux
import os
os.environ["VISPY_BACKEND"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

from lidar_mapping.sensors.vlp16 import VLP16Driver
from lidar_mapping.visualization import LivePointCloudViewer

def main():
    print("Connecting to VLP-16...")
    driver = VLP16Driver(host="0.0.0.0", port=2368)
    driver.start()
    
    viewer = LivePointCloudViewer(title="VLP-16 Live Feed (Orange Pi 5)", width=1280, height=720)
    
    def fetch_data():
        # Fast-forward to the latest frame to keep latency minimal
        frame = None
        while driver.frames_available() > 0:
            frame = driver.get_frame(timeout=0.01)
            
        if frame is None:
            # Dropdown to block if queue was empty
            frame = driver.get_frame(timeout=0.01)
            
        if frame is not None:
            return frame.to_numpy()
        return None

    viewer.set_callback(fetch_data)
    
    try:
        print("Starting interactive GUI. Close the window to exit.")
        viewer.start()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        driver.stop()
        print("Driver stopped.")

if __name__ == "__main__":
    main()
