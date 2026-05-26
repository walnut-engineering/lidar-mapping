import sys
import os
import time
import numpy as np
import cv2

# Ensure GL compatibility
os.environ["VISPY_BACKEND"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

from lidar_mapping.sensors.vlp16 import VLP16Driver
from lidar_mapping.sensors.imu import WitMotionDriver
from lidar_mapping.visualization.viewer import LivePointCloudViewer
# Assuming CameraCapture is in camera.py (let's use cv2.VideoCapture for absolute simplicity first if needed)
# Actually, let's use the camera module if possible, or simple cv2.

def quaternion_to_rotation_matrix(q):
    """(w, x, y, z) to 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

def main():
    print("Starting IMU...")
    imu = WitMotionDriver(port="/dev/ttyS1", baudrate=230400)
    imu.start()
    
    print("Starting VLP-16...")
    lidar = VLP16Driver(host="0.0.0.0", port=2368)
    lidar.start()
    
    print("Starting Camera...")
    # fallback to standard cv2 if the camera module isn't active
    cam = cv2.VideoCapture(0) # or the appropriate /dev/video index
    if not cam.isOpened():
        print("Warning: /dev/video0 not found, camera fusion skipped.")
    
    # Let devices spin up
    time.sleep(1.0)
    
    viewer = LivePointCloudViewer(title="Full Fusion Test", width=1280, height=720)
    
    last_rot = np.eye(3)
    
    def fetch_data():
        nonlocal last_rot
        
        frame_rgb = None
        # 1. Update Camera
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                # Resize and swap BGR to RGB for VisPy
                frame_small = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                
        # 2. Get latest IMU pose
        imu_reading = imu.get_latest_reading()
        if imu_reading is not None:
            # We have an updated rotation!
            last_rot = quaternion_to_rotation_matrix(imu_reading.quaternion)
            
        # 3. Get latest Lidar frame
        l_frame = None
        while lidar.frames_available() > 0:
            l_frame = lidar.get_frame(timeout=0.01)
        if l_frame is None:
            l_frame = lidar.get_frame(timeout=0.01)
            
        # 4. Transform and return
        if l_frame is not None:
            pts = l_frame.to_numpy() # Nx4 (x,y,z,i)
            xyz = pts[:, :3]
            
            # For this test, simply rotate the points by the IMU rotation matrix.
            # This demonstrates the tripod spinning logic!
            # R maps IMU body frame -> World frame. (Assuming Lidar and IMU axes are roughly aligned)
            xyz_rotated = xyz @ last_rot.T
            
            pts[:, :3] = xyz_rotated
            return (pts, frame_rgb)
            
        return (None, frame_rgb)

    viewer.set_callback(fetch_data)
    
    try:
        print("Starting interactive GUI. Rotate the tripod to see world points stabilize!")
        viewer.start()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        imu.stop()
        lidar.stop()
        if cam.isOpened():
            cam.release()
        cv2.destroyAllWindows()
        print("All drivers stopped.")

if __name__ == "__main__":
    main()
