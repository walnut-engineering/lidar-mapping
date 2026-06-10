#!/usr/bin/env bash
set -euo pipefail

# Safe direct-LiDAR networking while Wi-Fi stays on the same /24.
# It uses a /32 address on eth0 plus a host route to the LiDAR only,
# so the main 192.168.1.0/24 route remains on wlan0.
#
# Usage:
#   sudo bash apps/lidar_net_safe.sh up
#   sudo bash apps/lidar_net_safe.sh down
#
# Optional env overrides:
#   IFACE=eth0
#   HOST_IP=192.168.1.100
#   LIDAR_IP=192.168.1.201

IFACE="${IFACE:-eth0}"
HOST_IP="${HOST_IP:-192.168.1.100}"
LIDAR_IP="${LIDAR_IP:-192.168.1.201}"

cmd="${1:-}"
if [[ "$cmd" != "up" && "$cmd" != "down" ]]; then
  echo "usage: sudo bash apps/lidar_net_safe.sh [up|down]"
  exit 2
fi

if [[ "$cmd" == "up" ]]; then
  ip link set "$IFACE" up

  # Remove stale host route/address from previous runs if present.
  ip route del "$LIDAR_IP"/32 dev "$IFACE" 2>/dev/null || true
  ip addr del "$HOST_IP"/32 dev "$IFACE" 2>/dev/null || true

  # Add /32 host address and route only for the LiDAR peer.
  ip addr add "$HOST_IP"/32 dev "$IFACE"
  ip route add "$LIDAR_IP"/32 dev "$IFACE" src "$HOST_IP"

  echo "Configured $IFACE for LiDAR host-route mode"
  echo "  host ip : $HOST_IP/32"
  echo "  lidar ip: $LIDAR_IP/32 via $IFACE"
  echo
  ip -4 addr show dev "$IFACE"
  ip route get "$LIDAR_IP"
  exit 0
fi

# down
ip route del "$LIDAR_IP"/32 dev "$IFACE" 2>/dev/null || true
ip addr del "$HOST_IP"/32 dev "$IFACE" 2>/dev/null || true

echo "Removed LiDAR host-route config from $IFACE"
ip -4 addr show dev "$IFACE"
