#!/bin/bash
# Finetune RDT from the official pretrained RDT-1B checkpoint.
#
# This loads matching weights from the official RDT-1B (HuggingFace:
# robotics-diffusion-transformer/rdt-1b) and finetunes on our
# Shadow Hand + UR arm data. Mismatched layers (state_adaptor,
# output_head final layer) are trained from scratch.
#
# Usage:
#   bash scripts/finetune_rdt.sh DATA_PATH SAVE_PATH GPU PRETRAINED_CKPT [FEATURE_DIR]
#
# Examples:
#   # Download the checkpoint first:
#   #   pip install huggingface_hub
#   #   python -c "from huggingface_hub import snapshot_download; snapshot_download('robotics-diffusion-transformer/rdt-1b', local_dir='checkpoints/rdt-1b-pretrained')"
#
#   # Finetune with precomputed SigLIP features (recommended)
#   bash scripts/finetune_rdt.sh data/easy_mode checkpoints/rdt_ft 0 \
#       checkpoints/rdt-1b-pretrained data/siglip_features
#
#   # Finetune without precomputed features (on-the-fly, slower)
#   bash scripts/finetune_rdt.sh data/easy_mode checkpoints/rdt_ft 0 \
#       checkpoints/rdt-1b-pretrained
#
# Environment variables (override defaults):
#   BATCH_SIZE=32   EPOCHS=100  LR=5e-5  NUM_WORKERS=32
#   EXP_NAME=rdt_ft  GRAD_ACCUM=4
#   INSTRUCTIONS_FILE=workflow/instructions.json
#   FREEZE_BACKBONE=0  (set to 1 to freeze the ACI decoder layers)
#   USE_WANDB=1 WANDB_PROJECT=TexasPoker_RDT WANDB_ENTITY=<entity>

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU PRETRAINED_CKPT [FEATURE_DIR]"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU PRETRAINED_CKPT [FEATURE_DIR]"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU PRETRAINED_CKPT [FEATURE_DIR]"}
PRETRAINED_CKPT=${4:?"Usage: $0 DATA_PATH SAVE_PATH GPU PRETRAINED_CKPT [FEATURE_DIR]"}
FEATURE_DIR=${5:-""}

# Tunable via env vars - lower LR and batch size for finetuning
EXP_NAME=${EXP_NAME:-"rdt_ft"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EPOCHS=${EPOCHS:-100}
LR=${LR:-"5e-5"}
NUM_WORKERS=${NUM_WORKERS:-32}
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
echo "  RDT Finetune  (from official RDT-1B pretrained)"
echo "============================================================"
echo "  Mode:         $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:          $GPU"
echo "  Batch size:   $BATCH_SIZE (x$GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective)"
echo "  LR:           $LR"
echo "  Train files:  $TOTAL_TRAIN"
echo "  Val files:    $TOTAL_VAL"
echo "  Pretrained:   $PRETRAINED_CKPT"
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
    export CUDA_VISIBLE_DEVICES="$GPU"
    LAUNCHER=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
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

    # Training - lower LR for finetuning
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight_decay 1e-5
    --num_workers "$NUM_WORKERS"
    --use_amp

    # LR schedule - cosine warmup is good for finetuning
    --lr_schedule cosine
    --warmup_steps 200

    # Checkpointing
    --save_path "$SAVE_PATH"
    --save_freq 1 --eval_freq 1

    # Pretrained checkpoint
    --pretrained_ckpt "$PRETRAINED_CKPT"

    # Episode isolation
    --isolate_episodes

    # RDT-specific (must match official RDT-1B architecture)
    --rdt_text_encoder t5_xxl
    --rdt_token_max_len 120
    --rdt_hidden_size 2048
    --rdt_depth 28
    --rdt_num_heads 32
    --rdt_ff_dim 2048
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
# torch.compile
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
