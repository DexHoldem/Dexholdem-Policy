#!/bin/bash
# Train DiffusionPolicy WITH joint efforts — ablation variant.
#
# Same as train_dp.sh but adds joint efforts (24-d) to the observation,
# so the policy can condition on hand joint torques.
#
# Usage:
#   bash scripts/train_dp_eff.sh DATA_PATH SAVE_PATH GPU [FEATURE_DIR]
#
# Examples:
#   bash scripts/train_dp_eff.sh data/easy_mode checkpoints/dp_eff 1 data/vitl14_features
#
# Environment variables (override defaults):
#   BATCH_SIZE=128  EPOCHS=300  LR=1e-4  NUM_WORKERS=8
#   EXP_NAME=dp_eff  TRAIN_SUFFIX=_train  TEST_SUFFIX=_test
#   INSTR_MODE=integer  (one-hot instruction embedding for DinoV2/ResNet models)
#   CACHE_ON_GPU=1  (auto-enabled when FEATURE_DIR is set)
#   LAZY_LOADING=1  (load images on-demand; requires uncompressed NPZ)

DATA_BASE_PATH=${1:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
SAVE_PATH=${2:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
GPU=${3:?"Usage: $0 DATA_PATH SAVE_PATH GPU [FEATURE_DIR]"}
FEATURE_DIR=${4:-"data/vitl14_features"}

# Tunable via env vars
EXP_NAME=${EXP_NAME:-"dp_eff"}
TRAIN_SUFFIX=${TRAIN_SUFFIX:-"_train"}
TEST_SUFFIX=${TEST_SUFFIX:-"_test"}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-100}
LR=${LR:-"1e-4"}
NUM_WORKERS=${NUM_WORKERS:-32}
INSTR_MODE=${INSTR_MODE:-"integer"}

# Auto-enable GPU caching when using precomputed features (small data)
if [ -n "$FEATURE_DIR" ]; then
    CACHE_ON_GPU=${CACHE_ON_GPU:-0}
else
    CACHE_ON_GPU=${CACHE_ON_GPU:-0}
fi

# ---- Shared: conda + data path detection ------------------------------------
source "$(dirname "$0")/_common.sh"

TIMESTAMP=$(date +%m%d_%H%M)

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DiffusionPolicy + Efforts  (DinoV2 ViT-L/14 + Transformer)"
echo "============================================================"
echo "  Mode:        $(if $MULTITASK; then echo multi-task; else echo single-task; fi)"
echo "  GPU:         $GPU"
echo "  Batch size:  $BATCH_SIZE"
echo "  Train files: $TOTAL_TRAIN"
echo "  Val files:   $TOTAL_VAL"
echo "  Features:    $([ -n "$FEATURE_DIR" ] && echo "$FEATURE_DIR" || echo "on-the-fly")"
echo "  GPU cache:   $([ "$CACHE_ON_GPU" = "1" ] && echo "on" || echo "off")"
echo "  AMP:         off (fp32, match TexasPoker)"
echo "  Save path:   $SAVE_PATH"
echo "============================================================"
echo ""

mkdir -p "$SAVE_PATH"

# ---- Build python command ----------------------------------------------------
CMD=(python train.py
    --model diffusion_policy
    --gpu "$GPU"

    # Encoder
    --rgb_encoder dinov2_vitl14
    --depth_encoder resnet18
    --freeze_rgb_encoder
    --diffusion_model_type transformer

    # Modalities (pos + efforts)
    --representation_type "img-depth-pos-efforts"
    --camera_indices "012"
    --rgb_per_cam_output 96
    --depth_per_cam_output 32
    --pos_output_size 128

    # Temporal windows
    --obs_horizon 1 --action_horizon 32 --pred_horizon 64

    # Transformer (match TexasPoker: 768/12/12)
    --transformer_hidden_size 768
    --transformer_depth 12
    --transformer_num_heads 12
    --transformer_causal_attn
    --transformer_n_cond_layers 4
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

    # WandB
    --use_wandb
    --wandb_project "TexasPoker_DP"
    --wandb_entity "winniechen2002"
    --wandb_exp_name "${EXP_NAME}_${TIMESTAMP}"
)

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
