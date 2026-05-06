#!/bin/bash
# Train DiffusionPolicy with the original Transformer backbone (Chi et al.).
#
# Uses the DiffusionPolicy Transformer architecture (Chi et al.)
# sized to match DP (train_dp.sh):
#   hidden=768, depth=12, heads=12, causal_attn, n_cond_layers=4
# with a trainable ResNet18 RGB encoder by default.
#
# Usage:
#   bash scripts/train_dp_transformer.sh DATA_PATH SAVE_PATH GPU [FEATURE_DIR]
#
# Examples:
#   bash scripts/train_dp_transformer.sh data/easy_mode checkpoints/dp_transformer_resnet 0
#   RGB_ENCODER=dinov2_vitl14 bash scripts/train_dp_transformer.sh data/easy_mode checkpoints/dp_transformer_dino 0 data/vitl14_features
#
# Environment variables (override defaults):
#   BATCH_SIZE=128  EPOCHS=300  LR=1e-4  NUM_WORKERS=32
#   EXP_NAME=dp_transformer_resnet  TRAIN_SUFFIX=_train  TEST_SUFFIX=_test
#   RGB_ENCODER=resnet18  (set RGB_ENCODER=dinov2_vitl14 for frozen DinoV2)
#   INSTR_MODE=integer  (one-hot instruction embedding)
#   CACHE_ON_GPU=0  LAZY_LOADING=1  COMPILE=0  RESUME=...
#   HIDDEN=768  DEPTH=12  HEADS=12  N_COND_LAYERS=4
#   USE_WANDB=1 WANDB_PROJECT=TexasPoker_DP WANDB_ENTITY=<entity>

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}

# Tunable via env vars
EXP_NAME=${EXP_NAME:-"dp_transformer_resnet"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-300}
LR=${LR:-"1e-4"}
NUM_WORKERS=${NUM_WORKERS:-32}
INSTR_MODE=${INSTR_MODE:-"integer"}
RGB_ENCODER=${RGB_ENCODER:-"resnet18"}
if [ "$RGB_ENCODER" = "resnet18" ]; then
    FEATURE_DIR=${4:-${FEATURE_DIR:-""}}
else
    FEATURE_DIR=${4:-${FEATURE_DIR:-""}}
fi
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"TexasPoker_DP"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

# Transformer defaults (same size as train_dp.sh)
HIDDEN=${HIDDEN:-768}
DEPTH=${DEPTH:-12}
HEADS=${HEADS:-12}
N_COND_LAYERS=${N_COND_LAYERS:-4}

# GPU caching
CACHE_ON_GPU=${CACHE_ON_GPU:-0}

# ---- Shared: conda + data path detection ------------------------------------
source "$(dirname "$0")/_common.sh"

TIMESTAMP=$(date +%m%d_%H%M)

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DiffusionPolicy Transformer (dp_transformer_resnet)"
echo "============================================================"
echo "  Mode:        $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:         $GPU"
echo "  RGB encoder: $RGB_ENCODER"
echo "  Batch size:  $BATCH_SIZE"
echo "  Transformer: hidden=$HIDDEN depth=$DEPTH heads=$HEADS cond_layers=$N_COND_LAYERS"
echo "  Train files: $TOTAL_TRAIN"
echo "  Val files:   $TOTAL_VAL"
echo "  Features:    $([ -n "$FEATURE_DIR" ] && echo "$FEATURE_DIR" || echo "on-the-fly")"
echo "  GPU cache:   $([ "$CACHE_ON_GPU" = "1" ] && echo "on" || echo "off")"
echo "  AMP:         off (fp32)"
echo "  Save path:   $SAVE_PATH"
echo "============================================================"
echo ""

mkdir -p "$SAVE_PATH"

# ---- Build python command ----------------------------------------------------
CMD=(python train.py
    --model diffusion_policy
    --gpu "$GPU"

    # Encoder
    --rgb_encoder "$RGB_ENCODER"
    --depth_encoder resnet18
    --diffusion_model_type transformer

    # Modalities
    --representation_type "img-depth-pos"
    --camera_indices "012"
    --rgb_per_cam_output 96
    --depth_per_cam_output 32
    --pos_output_size 128

    # Temporal windows
    --obs_horizon 1 --action_horizon 32 --pred_horizon 64

    # DiffusionPolicy Transformer config (768/12/12, same as DP)
    --transformer_hidden_size "$HIDDEN"
    --transformer_depth "$DEPTH"
    --transformer_num_heads "$HEADS"
    --transformer_causal_attn
    --transformer_n_cond_layers "$N_COND_LAYERS"

    # Training
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight_decay 1e-5
    --num_workers "$NUM_WORKERS"
    --no-use_amp
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

if [ "$RGB_ENCODER" != "resnet18" ]; then
    CMD+=( --freeze_rgb_encoder )
fi

if [ "$RGB_ENCODER" = "resnet18" ] && [ -n "$FEATURE_DIR" ]; then
    echo "ERROR: RGB_ENCODER=resnet18 must not use FEATURE_DIR/precomputed RGB features."
    echo "       Run with no 4th argument and unset FEATURE_DIR."
    exit 1
fi

# GPU caching
if [ "$CACHE_ON_GPU" = "1" ]; then
    CMD+=( --cache_on_gpu )
fi
if [ "${LAZY_LOADING:-1}" = "1" ]; then
    CMD+=( --lazy_loading )
fi

# torch.compile
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
        CMD+=(--use_text_instruction --text_encoder "$TEXT_ENCODER"
              --instructions_file "$INSTRUCTIONS_FILE")
    else
        CMD+=( --use_instruction )
    fi
    if [ -n "$FEATURE_DIR" ]; then
        CMD+=(
            --feature_dirs     "$FEATURE_TRAIN_PATHS"
            --val_feature_dirs "$FEATURE_VAL_PATHS"
            --precompute_rgb_features
        )
    fi
else
    CMD+=(--train_path "$TRAIN_PATH" --val_path "$VAL_PATH")
    if [ "$INSTR_MODE" = "text" ]; then
        CMD+=(--use_text_instruction --text_encoder "$TEXT_ENCODER"
              --instructions_file "$INSTRUCTIONS_FILE"
              --num_instructions 32 --instruction_embed_dim 128)
    elif [ "$INSTR_MODE" = "integer" ]; then
        CMD+=(--use_instruction --num_instructions 32 --instruction_embed_dim 128)
    fi
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
