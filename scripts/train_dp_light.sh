#!/bin/bash
# Train DiffusionPolicy lightweight (ResNet18 + UNet backbone).
# Same encoder/modality parameters as train_act.sh for fair comparison.
#
# Usage:
#   bash scripts/train_dp_light.sh DATA_PATH SAVE_PATH GPU
#
# Examples:
#   bash scripts/train_dp_light.sh data/easy_mode checkpoints/dp_light 2
#
# Environment variables (override defaults):
#   BATCH_SIZE=128  EPOCHS=100  LR=1e-4  NUM_WORKERS=8
#   EXP_NAME=dp_light  TRAIN_SUFFIX=_train  TEST_SUFFIX=_test
#   REPR_TYPE=img-depth-pos
#   INSTR_MODE=""  INSTRUCTIONS_FILE=workflow/instructions.json
#   CACHE_ON_GPU=0
#   LAZY_LOADING=1  (load images on-demand; requires uncompressed NPZ)
#   USE_WANDB=1 WANDB_PROJECT=TexasPoker_DP WANDB_ENTITY=<entity>

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU"}

# Tunable via env vars - SAME defaults as train_act.sh
EXP_NAME=${EXP_NAME:-"dp_light"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-100}
LR=${LR:-"1e-4"}
NUM_WORKERS=${NUM_WORKERS:-32}
REPR_TYPE=${REPR_TYPE:-"img-depth-pos"}
INSTR_MODE=${INSTR_MODE:-"integer"}
INSTRUCTIONS_FILE=${INSTRUCTIONS_FILE:-"workflow/instructions.json"}
CACHE_ON_GPU=${CACHE_ON_GPU:-0}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"TexasPoker_DP"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
FEATURE_DIR=""

# ---- Shared: conda + data path detection ------------------------------------
source "$(dirname "$0")/_common.sh"

TIMESTAMP=$(date +%m%d_%H%M)

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DiffusionPolicy  (ResNet18 + UNet, lightweight)"
echo "============================================================"
echo "  Mode:        $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:         $GPU"
echo "  Batch size:  $BATCH_SIZE"
echo "  Repr:        $REPR_TYPE"
echo "  Train files: $TOTAL_TRAIN"
echo "  Val files:   $TOTAL_VAL"
echo "  GPU cache:   $([ "$CACHE_ON_GPU" = "1" ] && echo "on" || echo "off")"
echo "  AMP:         on (bf16)"
echo "  Save path:   $SAVE_PATH"
echo "============================================================"
echo ""

mkdir -p "$SAVE_PATH"

# ---- Build python command ----------------------------------------------------
# Encoder params match train_act.sh exactly for fair comparison.
CMD=(python train.py
    --model diffusion_policy
    --gpu "$GPU"

    # Encoder - ResNet18 RGBD fused (4-channel, halves backbone passes)
    --rgb_encoder resnet18
    --depth_encoder resnet18
    --fuse_rgbd
    --diffusion_model_type unet
    --representation_type "$REPR_TYPE"
    --camera_indices "012"
    --rgb_per_cam_output 96
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
    --lr_schedule cosine
    --warmup_steps 500

    # Checkpointing
    --save_path "$SAVE_PATH"
    --save_freq 5 --eval_freq 5

    # Diffusion
    --num_diffusion_iters 100

    # Episode isolation
    --isolate_episodes

)

if [ "$USE_WANDB" = "1" ]; then
    CMD+=( --use_wandb --wandb_project "$WANDB_PROJECT" )
    if [ -n "$WANDB_ENTITY" ]; then
        CMD+=( --wandb_entity "$WANDB_ENTITY" )
    fi
    CMD+=( --wandb_exp_name "${EXP_NAME}_${TIMESTAMP}" )
fi

# GPU caching
if [ "$CACHE_ON_GPU" = "1" ]; then
    CMD+=( --cache_on_gpu )
fi
if [ "${LAZY_LOADING:-1}" = "1" ]; then
    CMD+=( --lazy_loading )
fi

# torch.compile (optimizes compute-heavy model components)
if [ "${COMPILE:-0}" = "1" ]; then
    CMD+=( --compile )
fi
if [ -n "${RESUME:-}" ]; then
    CMD+=( --resume "$RESUME" )
fi
if [ "${PIN_MEMORY:-1}" = "0" ]; then
    CMD+=( --no-pin_memory )
fi

# ---- Data paths (mode-dependent) ---------------------------------------------
if $MULTITASK; then
    CMD+=(
        --multitask
        --train_paths "$TRAIN_PATH"
        --val_paths   "$VAL_PATH"
        --num_instructions 32
        --instruction_embed_dim 128
    )
    if [ "$INSTR_MODE" = "text" ]; then
        CMD+=(--use_text_instruction --text_encoder "clip"
              --instructions_file "$INSTRUCTIONS_FILE")
    else
        CMD+=( --use_instruction )
    fi
else
    CMD+=(--train_path "$TRAIN_PATH" --val_path "$VAL_PATH")
    if [ "$INSTR_MODE" = "text" ]; then
        CMD+=(--use_text_instruction --text_encoder "clip"
              --instructions_file "$INSTRUCTIONS_FILE"
              --num_instructions 32 --instruction_embed_dim 128)
    elif [ "$INSTR_MODE" = "integer" ]; then
        CMD+=(--use_instruction --num_instructions 32 --instruction_embed_dim 128)
    fi
fi

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
