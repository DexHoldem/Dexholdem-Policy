#!/bin/bash
# Train BAKU — native Dexas adaptation of the BAKU action-token policy.
#
# Usage:
#   bash scripts/train_baku.sh DATA_PATH SAVE_PATH GPU
#
# Examples:
#   bash scripts/train_baku.sh data/easy_mode checkpoints/baku 0
#   bash scripts/train_baku.sh data/easy_mode/0 checkpoints/baku_0 0
#
# Environment variables (override defaults):
#   BATCH_SIZE=256  EPOCHS=100  LR=1e-4  NUM_WORKERS=8
#   EXP_NAME=baku  TRAIN_SUFFIX=_train  TEST_SUFFIX=_test
#   REPR_TYPE=img-depth-pos
#   INSTR_MODE=integer  INSTRUCTIONS_FILE=workflow/instructions.json
#   USE_FILM=1
#   CACHE_ON_GPU=0  LAZY_LOADING=1

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}
FEATURE_DIR=""

# Tunable via env vars
EXP_NAME=${EXP_NAME:-"baku"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-100}
LR=${LR:-"1e-4"}
NUM_WORKERS=${NUM_WORKERS:-16}
REPR_TYPE=${REPR_TYPE:-"img-depth-pos"}
INSTR_MODE=${INSTR_MODE:-"integer"}
INSTRUCTIONS_FILE=${INSTRUCTIONS_FILE:-"workflow/instructions.json"}
USE_FILM=${USE_FILM:-1}
CACHE_ON_GPU=${CACHE_ON_GPU:-0}
SAVE_FREQ=${SAVE_FREQ:-10}
EVAL_FREQ=${EVAL_FREQ:-10}

source "$(dirname "$0")/_common.sh"

TIMESTAMP=$(date +%m%d_%H%M)

echo ""
echo "============================================================"
echo "  BAKU  (native action-token Transformer)"
echo "============================================================"
echo "  Mode:        $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:         $GPU"
echo "  Batch size:  $BATCH_SIZE"
echo "  Repr:        $REPR_TYPE"
echo "  Train files: $TOTAL_TRAIN"
echo "  Val files:   $TOTAL_VAL"
echo "  FiLM:        $([ "$USE_FILM" = "1" ] && echo "on" || echo "off")"
echo "  GPU cache:   $([ "$CACHE_ON_GPU" = "1" ] && echo "on" || echo "off")"
echo "  Save path:   $SAVE_PATH"
echo "============================================================"
echo ""

mkdir -p "$SAVE_PATH"

CMD=(python train.py
    --model baku
    --gpu "$GPU"

    # Encoder
    --rgb_encoder resnet18
    --depth_encoder resnet18
    --fuse_rgbd
    --representation_type "$REPR_TYPE"
    --camera_indices "012"
    --rgb_per_cam_output 128
    --depth_per_cam_output 32
    --pos_output_size 128

    # Temporal windows
    --obs_horizon 1 --action_horizon 32 --pred_horizon 64

    # Training
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight_decay 1e-5
    --num_workers "$NUM_WORKERS"
    --use_amp

    # Checkpointing
    --save_path "$SAVE_PATH"
    --save_freq "$SAVE_FREQ"
    --eval_freq "$EVAL_FREQ"

    # BAKU-specific
    --baku_hidden_size 256
    --baku_depth 8
    --baku_num_heads 4
    --baku_dropout 0.1

    # WandB
    --use_wandb
    --wandb_project "TexasPoker_BAKU"
    --wandb_entity "winniechen2002"
    --wandb_exp_name "${EXP_NAME}_${TIMESTAMP}"
)

if [ "$CACHE_ON_GPU" = "1" ]; then
    CMD+=( --cache_on_gpu )
fi
if [ "${LAZY_LOADING:-1}" = "1" ]; then
    CMD+=( --lazy_loading )
fi
if [ "${COMPILE:-0}" = "1" ]; then
    CMD+=( --compile )
fi
if [ "$USE_FILM" = "1" ]; then
    CMD+=( --baku_use_film )
else
    CMD+=( --no-baku_use_film )
fi

if $MULTITASK; then
    CMD+=(
        --multitask
        --train_paths "$TRAIN_PATH"
        --val_paths   "$VAL_PATH"
        --num_instructions 14
        --instruction_embed_dim 64
    )
    if [ "$INSTR_MODE" = "text" ]; then
        CMD+=( --use_text_instruction --text_encoder "clip"
               --instructions_file "$INSTRUCTIONS_FILE" )
    elif [ "$INSTR_MODE" = "integer" ]; then
        CMD+=( --use_instruction )
    fi
else
    CMD+=( --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" )
    if [ "$INSTR_MODE" = "text" ]; then
        CMD+=( --use_text_instruction --text_encoder "clip"
               --instructions_file "$INSTRUCTIONS_FILE"
               --num_instructions 14 --instruction_embed_dim 64 )
    elif [ "$INSTR_MODE" = "integer" ]; then
        CMD+=( --use_instruction --num_instructions 14 --instruction_embed_dim 64 )
    fi
fi

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
