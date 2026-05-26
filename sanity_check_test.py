from lidar_mapping.sensors.imu import WitMotionDriver
import time
d = WitMotionDriver()
d.start()
time.sleep(1)
print(d.get_latest_reading())
d.stop()
