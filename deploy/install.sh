#!/usr/bin/env bash
# Install / update the lidar-mapping kit on a Raspberry Pi 5.
#
# Idempotent: re-running upgrades the venv and reloads systemd.
#
# Usage (as root):
#     sudo ./deploy/install.sh [--no-service] [--source DIR]
#
# Options:
#     --no-service   Skip systemd unit installation (dev installs).
#     --source DIR   Source tree to install from (default: repo root).
#
# After install:
#     sudo systemctl enable --now lidar-mapping.service
#     journalctl -u lidar-mapping -f

set -euo pipefail

INSTALL_DIR="/opt/lidar-mapping"
CONFIG_DIR="/etc/lidar-mapping"
DATA_DIR="/var/lib/lidar-mapping"
MAPS_DIR="/home/pi/maps"
SERVICE_USER="pi"
SERVICE_GROUP="pi"
PY_BIN="${PY_BIN:-python3}"

INSTALL_SERVICE=1
SOURCE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-service) INSTALL_SERVICE=0; shift ;;
        --source)     SOURCE_DIR="$2"; shift 2 ;;
        -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$SOURCE_DIR" ]]; then
    SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)." >&2
    exit 1
fi

echo "==> Source:  $SOURCE_DIR"
echo "==> Install: $INSTALL_DIR"

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
echo "==> Installing system dependencies via apt"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-tk \
    libatlas-base-dev libopenblas-dev \
    libgl1 libusb-1.0-0 \
    git rsync

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
echo "==> Creating directories"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$INSTALL_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$DATA_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$MAPS_DIR"
install -d -m 0755 "$CONFIG_DIR"

echo "==> Syncing source tree"
rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    --exclude '*.pyc' --exclude '.pytest_cache' \
    "$SOURCE_DIR/" "$INSTALL_DIR/"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
VENV="$INSTALL_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
    echo "==> Creating virtualenv at $VENV"
    sudo -u "$SERVICE_USER" "$PY_BIN" -m venv "$VENV"
fi

echo "==> Upgrading pip and installing project"
sudo -u "$SERVICE_USER" "$VENV/bin/pip" install --upgrade pip wheel
sudo -u "$SERVICE_USER" "$VENV/bin/pip" install \
    "$INSTALL_DIR[imu-serial,camera]"

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
if [[ ! -f "$CONFIG_DIR/kit.toml" ]]; then
    echo "==> Writing default config $CONFIG_DIR/kit.toml"
    cat > "$CONFIG_DIR/kit.toml" <<'TOML'
# /etc/lidar-mapping/kit.toml — edit to match your hardware
[lidar]
enabled = true
host = "0.0.0.0"
data_port = 2368
position_port = 8308

[imu]
enabled = true
driver = "witmotion"
port = "/dev/ttyUSB0"
baud = 115200
rate_hz = 100

[camera]
enabled = false

[mapper]
voxel_size = 0.1
min_range = 0.5
max_range = 100.0
z_min = -3.0
z_max = 20.0
remove_ground = true

[ui]
fullscreen = true
TOML
fi

# ---------------------------------------------------------------------------
# systemd unit
# ---------------------------------------------------------------------------
if [[ "$INSTALL_SERVICE" -eq 1 ]]; then
    echo "==> Installing systemd unit"
    install -m 0644 \
        "$SOURCE_DIR/deploy/lidar-mapping.service" \
        /etc/systemd/system/lidar-mapping.service
    systemctl daemon-reload
    echo
    echo "Enable and start with:"
    echo "    sudo systemctl enable --now lidar-mapping.service"
fi

echo
echo "==> Done."
echo "Test in simulation (no hardware required):"
echo "    sudo -u $SERVICE_USER $VENV/bin/lidar-ui --simulate"
