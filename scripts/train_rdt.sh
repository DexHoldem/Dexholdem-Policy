#!/bin/bash
# Train RDT-small (~170M, RDT(small) ablation) from scratch - SigLIP + T5 + ACI.
# Architecture matches HF robotics-diffusion-transformer/rdt-170m:
#   hidden_size=1024, depth=14, num_heads=32.
#
# Usage:
#   bash scripts/train_rdt.sh DATA_PATH SAVE_PATH GPU [FEATURE_DIR]
#
# Examples:
#   # With precomputed SigLIP features (recommended)
#   bash scripts/train_rdt.sh data/easy_mode checkpoints/rdt_small 5 data/siglip_features
#
#   # Without precomputed features (on-the-fly, slower)
#   bash scripts/train_rdt.sh data/easy_mode checkpoints/rdt_small 5
#
#   # Single-task
#   bash scripts/train_rdt.sh data/easy_mode/0 checkpoints/rdt_small_0 5
#
# Precompute SigLIP features first (run once):
#   python workflow/precompute_features.py \
#       --encoder siglip_so400m --data_dir data/easy_mode \
#       --feature_dir data/siglip_features --gpu 0
#
# Environment variables (override defaults):
#   BATCH_SIZE=128  EPOCHS=100  LR=1e-4  NUM_WORKERS=8
#   EXP_NAME=rdt_small  TRAIN_SUFFIX=_train  TEST_SUFFIX=_test
#   INSTRUCTIONS_FILE=workflow/instructions.json
#   CACHE_ON_GPU=1  (auto-enabled when FEATURE_DIR is set)
#   LAZY_LOADING=1  (load images on-demand; requires uncompressed NPZ)
#   USE_WANDB=1 WANDB_PROJECT=TexasPoker_RDT WANDB_ENTITY=<entity>

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
FEATURE_DIR=${4:-""}

# Tunable via env vars
EXP_NAME=${EXP_NAME:-"rdt_small"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EPOCHS=${EPOCHS:-100}
LR=${LR:-"1e-4"}
NUM_WORKERS=${NUM_WORKERS:-32}
SAVE_FREQ=${SAVE_FREQ:-10}
EVAL_FREQ=${EVAL_FREQ:-10}
INSTRUCTIONS_FILE=${INSTRUCTIONS_FILE:-"workflow/instructions.json"}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"TexasPoker_RDT"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

# GPU caching - off by default (SigLIP patch tokens are too large to cache)
CACHE_ON_GPU=${CACHE_ON_GPU:-0}
GRAD_CKPT=${GRAD_CKPT:-0}

# ---- Shared: conda + data path detection ------------------------------------
source "$(dirname "$0")/_common.sh"

TIMESTAMP=$(date +%m%d_%H%M)

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  RDT-small ~170M  (SigLIP-SO400M + T5 + ACI Transformer, from scratch)"
echo "============================================================"
echo "  Mode:         $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:          $GPU"
echo "  Batch size:   $BATCH_SIZE (x$GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective)"
echo "  Train files:  $TOTAL_TRAIN"
echo "  Val files:    $TOTAL_VAL"
if [ -n "$FEATURE_DIR" ]; then
    echo "  SigLIP feats: $FEATURE_DIR  (patch tokens)"
else
    echo "  SigLIP feats: on-the-fly  (frozen backbone, runs each batch)"
fi
echo "  GPU cache:    $([ "$CACHE_ON_GPU" = "1" ] && echo "on" || echo "off")"
echo "  AMP:          on (bf16)"
echo "  Instructions: $INSTRUCTIONS_FILE"
echo "  Save path:    $SAVE_PATH"
echo "============================================================"
echo ""

mkdir -p "$SAVE_PATH"

# ---- Build python command ----------------------------------------------------
# If GPU contains a comma, use torchrun for DDP multi-GPU.
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    # Extract first GPU for CUDA_VISIBLE_DEVICES offset
    export CUDA_VISIBLE_DEVICES="$GPU"
    LAUNCHER=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
    # With torchrun, --gpu is ignored (LOCAL_RANK is used instead)
    GPU_ARG=""
else
    LAUNCHER=(python)
    GPU_ARG="--gpu $GPU"
fi

CMD=("${LAUNCHER[@]}" train.py
    --model rdt
    $GPU_ARG

    # Obs encoder - SigLIP-SO400M (frozen)
    --rgb_encoder siglip_so400m
    --freeze_rgb_encoder
    --representation_type "img-pos"
    --camera_indices "012"
    --rgb_per_cam_output 128
    --pos_output_size 128

    # Temporal windows (obs_horizon=2 matches official RDT-1B img_history_size)
    --obs_horizon 2 --action_horizon 32 --pred_horizon 64

    # Training
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight_decay 1e-5
    --num_workers "$NUM_WORKERS"
    --use_amp

    # Checkpointing
    --save_path "$SAVE_PATH"
    --save_freq "$SAVE_FREQ" --eval_freq "$EVAL_FREQ"

    # Episode isolation
    --isolate_episodes

    # RDT-specific (RDT-small / 170M architecture, from scratch)
    --rdt_text_encoder t5_xxl
    --rdt_token_max_len 120
    --rdt_hidden_size 1024
    --rdt_depth 14
    --rdt_num_heads 32
    --rdt_ff_dim 1024
    --rdt_dropout 0.0
    --rdt_num_diffusion_iters 1000
    --rdt_num_inference_iters 5
    --rdt_inference_scheduler dpmsolver
    --rdt_prediction_type sample
    --rdt_cond_mask_prob 0.0
    --rdt_siglip_resolution "${SIGLIP_RES:-384}"
    --rdt_siglip_pool_patches "${SIGLIP_POOL:-0}"
    --instructions_file "$INSTRUCTIONS_FILE"

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
# Gradient checkpointing
if [ "$GRAD_CKPT" = "1" ]; then
    CMD+=( --gradient_checkpointing )
fi
# torch.compile (targets RDT transformer only, not SigLIP)
if [ "${COMPILE:-0}" = "1" ]; then
    CMD+=( --compile )
fi
if [ -n "${RESUME:-}" ]; then
    CMD+=( --resume "$RESUME" )
fi
if [ "${LAZY_LOADING:-1}" = "1" ]; then
    CMD+=( --lazy_loading )
fi

# ---- Data paths (mode-dependent) ---------------------------------------------
# RDT handles text conditioning internally (T5 lookup on batch["instruction"]).
# Do NOT pass --use_instruction - that would double-condition.
if $MULTITASK; then
    CMD+=(
        --multitask
        --train_paths "$TRAIN_PATH"
        --val_paths   "$VAL_PATH"
        --num_instructions 32
    )
    if [ -n "$FEATURE_DIR" ]; then
        CMD+=(
            --feature_dirs     "$FEATURE_TRAIN_PATHS"
            --val_feature_dirs "$FEATURE_VAL_PATHS"
            --precompute_rgb_features
        )
    fi
else
    CMD+=(--train_path "$TRAIN_PATH" --val_path "$VAL_PATH" --num_instructions 32)
    if [ -n "$FEATURE_DIR" ]; then
        CMD+=(
            --feature_dir     "$FEATURE_TRAIN_PATHS"
            --val_feature_dir "$FEATURE_VAL_PATHS"
            --precompute_rgb_features
        )
    fi
fi

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
