#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${LEROBOT_PYTHON:-python}" src/robot_setup.py calibrate
