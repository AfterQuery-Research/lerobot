#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .env

episode="${1:-0}"
if [[ $# -gt 0 ]]; then
    shift
fi

uv run python -m lerobot.scripts.replay_yam_relative_filtered \
    --episode="$episode" \
    --robot-id="$ROBOT_ID" \
    --left-adapter-serial="$LEFT_CAN" \
    --right-adapter-serial="$RIGHT_CAN" \
    "$@"
