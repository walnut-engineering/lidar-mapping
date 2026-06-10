#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper so running from repo root works:
#   sudo bash lidar_net_safe.sh up
#   sudo bash lidar_net_safe.sh down

exec bash "$(dirname "$0")/apps/lidar_net_safe.sh" "$@"
