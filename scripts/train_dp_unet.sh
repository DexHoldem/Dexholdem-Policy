#!/bin/bash
# Public release alias for DP_unet.
#
# This keeps the older train_dp_light.sh entry point working while exposing
# the experiment name used in the paper/release tables.
#
# Usage:
#   bash scripts/train_dp_unet.sh DATA_PATH SAVE_PATH GPU

EXP_NAME=${EXP_NAME:-"dp_unet"}
exec "$(dirname "$0")/train_dp_light.sh" "$@"
