import time
from lidar_mapping.sensors.vlp16 import VLP16Driver

def test_vlp16():
    driver = VLP16Driver(host="0.0.0.0", port=2368)
    driver.start()
    print("VLP-16 driver started. Listening for packets on port 2368...")
    
    try:
        start_time = time.time()
        while time.time() - start_time < 5.0:
            frame = driver.get_frame(timeout=1.0)
            if frame is not None:
                points = frame.to_numpy()
                print(f"Received frame with {len(points)} points.")
            else:
                print("Waiting for frames...")
        
        print(f"Driver stats - Packets received: {driver.packets_received}, Frames completed: {driver.frames_completed}")
    finally:
        driver.stop()
        print("VLP-16 driver stopped.")

if __name__ == "__main__":
    test_vlp16()
