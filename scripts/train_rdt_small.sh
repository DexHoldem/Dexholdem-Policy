#!/bin/bash
# Public release alias for RDT_small.
#
# This keeps the older train_rdt.sh entry point working while exposing the
# experiment name used in the paper/release tables.
#
# Usage:
#   bash scripts/train_rdt_small.sh DATA_PATH SAVE_PATH GPU [FEATURE_DIR]

EXP_NAME=${EXP_NAME:-"rdt_small"}
exec "$(dirname "$0")/train_rdt.sh" "$@"
