# lidar-mapping
VLP-16 based mapping software testing

## Prerequisites

- Python 3.10+
- pip for Python 3

Install runtime dependencies:

python3 -m pip install .

Install with dev tests:

python3 -m pip install ".[dev]"

Quick non-hardware verification:

python3 -m pytest -q tests/test_imu.py

Hardware diagnostics (camera + IMU + LiDAR UDP ingress):

python3 -m apps.diagnose_sensors --camera-indices 0,2 --imu-port /dev/ttyS1 --imu-baud 230400

If LiDAR shows udp_packets=0, check Ethernet link/IP configuration and the sensor destination IP/port.

### Safe eth0 setup when Wi-Fi is on the same subnet

When Wi-Fi is already on 192.168.1.x, do not add a full 192.168.1.0/24 route on eth0 over SSH.
Use host-route mode instead:

sudo bash apps/lidar_net_safe.sh up

Shortcut from repo root:

sudo bash lidar_net_safe.sh up

This assigns a /32 address on eth0 and routes only the LiDAR IP to eth0.
It avoids stealing your normal Wi-Fi default/subnet routes.

Tear down after testing:

sudo bash apps/lidar_net_safe.sh down

Or:

sudo bash lidar_net_safe.sh down

Then run diagnostics again:

python3 -m apps.diagnose_sensors --camera-indices 0,2 --imu-port /dev/ttyS1 --imu-baud 230400

Run the fusion app with two cameras:

python3 -m apps.run_stationary --camera-indices 0,1 --primary-camera 0

For outward-angled cameras, pass per-camera yaw offsets (note `=` form for negative values):

python3 -m apps.run_stationary --camera-indices 0,2 --camera-yaws-deg=-10,10 --primary-camera 0

Rotation-phase validation while fusion is running:

python3 -m apps.run_rotation_test --duration 10 --countdown 1 --host 127.0.0.1 --port 8765

Rotation test snapshots are written to the project folder by default:

rotation_test_captures/before_camera.png
rotation_test_captures/after_camera.png
rotation_test_captures/before_top_down.png
rotation_test_captures/after_top_down.png
