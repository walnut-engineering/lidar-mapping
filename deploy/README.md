# Raspberry Pi 5 deployment

Files for installing the lidar-mapping kit on a Raspberry Pi 5 with a 7"
touchscreen, Velodyne VLP-16, WitMotion WTGAHRS2, and Pi Camera.

## Quick install

Copy the source tree to the Pi (e.g. via `git clone` or `rsync`) and run:

```bash
sudo ./deploy/install.sh
sudo systemctl enable --now lidar-mapping.service
```

The dashboard starts on the local display and the unit restarts on failure.
Tail logs with:

```bash
journalctl -u lidar-mapping -f
```

## What the installer does

- Installs apt packages (Python, Tk, OpenBLAS, libGL, rsync).
- Syncs the source tree to `/opt/lidar-mapping`.
- Creates a virtualenv at `/opt/lidar-mapping/.venv` and installs the project
  with `imu-serial` and `camera` extras.
- Writes a default `/etc/lidar-mapping/kit.toml` (first install only — edit
  freely; it will not be overwritten by re-runs).
- Installs `lidar-mapping.service` to `/etc/systemd/system/`.

Re-running the script upgrades the venv and source tree without touching the
config.

## Hardware setup

### VLP-16

The Velodyne emits UDP on ports **2368** (data) and **8308** (position) by
default. Plug the sensor into the Pi (or a switch) and give the Pi a static
IP on the same subnet (Velodyne default: `192.168.1.201`, sensor → host):

```bash
# /etc/dhcpcd.conf or NetworkManager equivalent
interface eth0
static ip_address=192.168.1.100/24
```

Verify packets arrive with `sudo tcpdump -i eth0 udp port 2368`.

### WitMotion WTGAHRS2

Connects via USB-serial; appears as `/dev/ttyUSB0`. Add the service user to
the `dialout` group (the systemd unit already does so via
`SupplementaryGroups=`). Adjust `imu.port` / `imu.baud` in `kit.toml` if
needed.

### Pi Camera

Enable in `raspi-config` → Interface Options → Camera, then set
`camera.enabled = true` in `kit.toml`. The `video` group is already added to
the service.

### 7" touchscreen

The official 800×480 DSI display works out of the box. The dashboard is sized
for it and runs fullscreen when launched via the unit (`--fullscreen`).

## Running without the service

For development or one-off captures:

```bash
sudo -u pi /opt/lidar-mapping/.venv/bin/lidar-ui --simulate            # no HW
sudo -u pi /opt/lidar-mapping/.venv/bin/lidar-ui --config /etc/lidar-mapping/kit.toml
```

Other CLIs installed by the project:

- `lidar-record` — capture raw VLP-16 + IMU streams to disk
- `lidar-playback` — replay a recording into the mapper
- `lidar-map` — build a map from a recording and write PLY

## Uninstall

```bash
sudo systemctl disable --now lidar-mapping.service
sudo rm /etc/systemd/system/lidar-mapping.service
sudo rm -rf /opt/lidar-mapping /etc/lidar-mapping /var/lib/lidar-mapping
sudo systemctl daemon-reload
```
