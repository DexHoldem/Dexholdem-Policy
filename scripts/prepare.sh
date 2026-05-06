#!/bin/bash
# Full data preparation pipeline: organize raw data + (optionally) precompute features.
#
# Step 1 - Organize raw per-operation NPZ files into the easy_mode training layout.
# Step 2 - (Optional) Extract DinoV2 + SigLIP features using ALL available GPUs.
#
# With 4 GPUs and both encoders, each encoder is sharded across 2 GPUs (4 workers total).
# With 2 GPUs, one encoder per GPU (2 workers).
# With 1 GPU or only one encoder, runs sequentially.
#
# Usage:
#   bash scripts/prepare.sh [SOURCE_DIR] [TARGET_DIR] [EVAL_COUNT] [NUM_GPUS] \
#        [DINO_FEATURE_DIR] [SIGLIP_FEATURE_DIR] [GPU_OFFSET] [--symlink]
#
# Arguments:
#   SOURCE_DIR          Raw data root                   (default: data/TexasPokerRobot)
#   TARGET_DIR          Organized output root           (default: data/easy_mode)
#   EVAL_COUNT          Test files held out per task    (default: 5)
#   NUM_GPUS            Number of GPUs to use           (default: 4)
#   DINO_FEATURE_DIR    Where to write DinoV2 NPZs      (default: empty = skip)
#   SIGLIP_FEATURE_DIR  Where to write SigLIP NPZs      (default: empty = skip)
#   GPU_OFFSET          First GPU index                 (default: 0, so GPUs 0..NUM_GPUS-1)
#   --symlink           Symlink instead of copy         (skips image resize, saves disk)
#
# Examples:
#   # Organize only
#   bash scripts/prepare.sh
#
#   # Both encoders on 4 GPUs (fastest: 2 shards x 2 encoders)
#   bash scripts/prepare.sh data/TexasPokerRobot data/easy_mode 5 4 \
#       data/vitl14_features data/siglip_features
#
#   # Both encoders on 2 GPUs (1 encoder per GPU)
#   bash scripts/prepare.sh data/TexasPokerRobot data/easy_mode 5 2 \
#       data/vitl14_features data/siglip_features
#
#   # DinoV2 only, sharded across 4 GPUs
#   bash scripts/prepare.sh data/TexasPokerRobot data/easy_mode 5 4 \
#       data/vitl14_features
#
#   # Use GPUs 4-7 instead of 0-3
#   bash scripts/prepare.sh data/TexasPokerRobot data/easy_mode 5 4 \
#       data/vitl14_features data/siglip_features 4

set -e

SYMLINK_FLAG=""
POSITIONAL_ARGS=()

# Accept --symlink anywhere in the argument list
for arg in "$@"; do
    if [ "$arg" = "--symlink" ]; then
        SYMLINK_FLAG="--symlink"
    else
        POSITIONAL_ARGS+=("$arg")
    fi
done

SOURCE_DIR=${POSITIONAL_ARGS[0]:-"data/TexasPokerRobot"}
TARGET_DIR=${POSITIONAL_ARGS[1]:-"data/easy_mode"}
EVAL_COUNT=${POSITIONAL_ARGS[2]:-5}
NUM_GPUS=${POSITIONAL_ARGS[3]:-4}
DINO_FEATURE_DIR=${POSITIONAL_ARGS[4]:-""}    # DinoV2 ViT-L/14 features
SIGLIP_FEATURE_DIR=${POSITIONAL_ARGS[5]:-""}  # SigLIP-SO400M features
GPU_OFFSET=${POSITIONAL_ARGS[6]:-0}           # First GPU index

# ---- Conda ------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate texas
if [[ "$CONDA_DEFAULT_ENV" != "texas" ]]; then
    echo "ERROR: Could not activate the 'texas' conda environment."
    exit 1
fi

# ---- Validate source --------------------------------------------------------
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# ---- Plan GPU allocation ----------------------------------------------------
NEED_DINO=false; NEED_SIGLIP=false
[ -n "$DINO_FEATURE_DIR" ]   && NEED_DINO=true
[ -n "$SIGLIP_FEATURE_DIR" ] && NEED_SIGLIP=true

# Determine how many shards per encoder
DINO_SHARDS=0; SIGLIP_SHARDS=0
if $NEED_DINO && $NEED_SIGLIP; then
    # Split GPUs evenly between the two encoders
    DINO_SHARDS=$((NUM_GPUS / 2))
    SIGLIP_SHARDS=$((NUM_GPUS - DINO_SHARDS))
    # Minimum 1 shard each
    [ $DINO_SHARDS -lt 1 ]   && DINO_SHARDS=1
    [ $SIGLIP_SHARDS -lt 1 ] && SIGLIP_SHARDS=1
elif $NEED_DINO; then
    DINO_SHARDS=$NUM_GPUS
elif $NEED_SIGLIP; then
    SIGLIP_SHARDS=$NUM_GPUS
fi

TOTAL_WORKERS=$((DINO_SHARDS + SIGLIP_SHARDS))

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Data preparation"
echo "============================================================"
echo "  Source        : $SOURCE_DIR"
echo "  Target        : $TARGET_DIR"
echo "  Eval count    : $EVAL_COUNT files per task"
if [ -n "$SYMLINK_FLAG" ]; then
    echo "  Copy mode     : symlink"
else
    echo "  Copy mode     : copy (original resolution 640x480)"
fi
if $NEED_DINO; then
    DINO_GPU_START=$GPU_OFFSET
    DINO_GPU_END=$((GPU_OFFSET + DINO_SHARDS - 1))
    echo "  DinoV2 feats  : $DINO_FEATURE_DIR  (${DINO_SHARDS} shards, GPU ${DINO_GPU_START}-${DINO_GPU_END})"
else
    echo "  DinoV2 feats  : skipped"
fi
if $NEED_SIGLIP; then
    SIGLIP_GPU_START=$((GPU_OFFSET + DINO_SHARDS))
    SIGLIP_GPU_END=$((SIGLIP_GPU_START + SIGLIP_SHARDS - 1))
    echo "  SigLIP feats  : $SIGLIP_FEATURE_DIR  (${SIGLIP_SHARDS} shards, GPU ${SIGLIP_GPU_START}-${SIGLIP_GPU_END})"
else
    echo "  SigLIP feats  : skipped"
fi
if [ $TOTAL_WORKERS -gt 1 ]; then
    echo "  Parallel      : $TOTAL_WORKERS workers across GPUs ${GPU_OFFSET}-$((GPU_OFFSET + NUM_GPUS - 1))"
fi
echo "============================================================"
echo ""

# ---- Step 1: Organize -------------------------------------------------------
N_STEPS=1
[ $TOTAL_WORKERS -gt 0 ] && N_STEPS=2

echo "Step 1/$N_STEPS - Organizing data ..."
echo ""

python workflow/organize_data.py \
    --source_dir "$SOURCE_DIR" \
    --target_dir "$TARGET_DIR" \
    --eval_count "$EVAL_COUNT" \
    $SYMLINK_FLAG

echo ""
echo "  Layout   : $TARGET_DIR/{0..13}/{task}_train_N/ + {task}_test/"
echo "  Metadata : $TARGET_DIR/instructions.json"

# ---- Step 2: Precompute features (all workers in parallel) ------------------

if [ $TOTAL_WORKERS -gt 0 ]; then
    echo ""
    echo "Step 2/$N_STEPS - Extracting features ($TOTAL_WORKERS parallel workers) ..."
    echo ""

    PIDS=()

    # Launch DinoV2 shards
    if $NEED_DINO; then
        for i in $(seq 0 $((DINO_SHARDS - 1))); do
            GPU_ID=$((GPU_OFFSET + i))
            # DinoV2 is lighter (224px, CLS only) - can use larger batches on A800
            echo "  [DinoV2 shard $i/$DINO_SHARDS]  GPU $GPU_ID  batch_size=128"
            python workflow/precompute_features.py \
                --data_dir    "$TARGET_DIR" \
                --feature_dir "$DINO_FEATURE_DIR" \
                --encoder     dinov2_vitl14 \
                --camera_indices 0 1 2 \
                --batch_size  128 \
                --gpu         "$GPU_ID" \
                --shard       "$i/$DINO_SHARDS" &
            PIDS+=($!)
        done
    fi

    # Launch SigLIP shards
    if $NEED_SIGLIP; then
        for i in $(seq 0 $((SIGLIP_SHARDS - 1))); do
            GPU_ID=$((GPU_OFFSET + DINO_SHARDS + i))
            # SigLIP is heavier (384px, 729 patch tokens) - fp16 allows batch_size=128 on 80GB
            echo "  [SigLIP shard $i/$SIGLIP_SHARDS]  GPU $GPU_ID  batch_size=128"
            python workflow/precompute_features.py \
                --data_dir    "$TARGET_DIR" \
                --feature_dir "$SIGLIP_FEATURE_DIR" \
                --encoder     siglip_so400m \
                --camera_indices 0 1 2 \
                --batch_size  128 \
                --gpu         "$GPU_ID" \
                --shard       "$i/$SIGLIP_SHARDS" &
            PIDS+=($!)
        done
    fi

    echo ""
    echo "  Waiting for $TOTAL_WORKERS workers to finish ..."

    # Wait for all workers; report any failures
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait $pid || { echo "ERROR: Worker PID $pid failed."; FAIL=1; }
    done
    [ $FAIL -ne 0 ] && exit 1

    echo ""
    $NEED_DINO   && echo "  DinoV2 layout : $DINO_FEATURE_DIR/{0..13}/{task}_train_N/ + {task}_test/"
    $NEED_SIGLIP && echo "  SigLIP layout : $SIGLIP_FEATURE_DIR/{0..13}/{task}_train_N/ + {task}_test/"
fi

# ---- Next steps -------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Preparation complete - next steps"
echo "============================================================"
echo ""
echo "  Train individual release models:"
echo "    # DP(DINO) (DinoV2 + Transformer, GPU 4)"
if [ -n "$DINO_FEATURE_DIR" ]; then
    echo "    bash scripts/train_dp.sh $TARGET_DIR checkpoints/dp_dino 4 $DINO_FEATURE_DIR"
else
    echo "    bash scripts/train_dp.sh $TARGET_DIR checkpoints/dp_dino 4"
fi
echo "    # DP_transformer_resnet (ResNet18 + Transformer, GPU 4)"
echo "    bash scripts/train_dp_transformer.sh $TARGET_DIR checkpoints/dp_transformer_resnet 4"
echo "    # DP_unet (ResNet18 + UNet, GPU 5)"
echo "    bash scripts/train_dp_unet.sh $TARGET_DIR checkpoints/dp_unet 5"
echo "    # ACT (ResNet18 + CVAE, GPU 5)"
echo "    bash scripts/train_act.sh $TARGET_DIR checkpoints/act 5"
echo "    # RDT_small (SigLIP + T5 + ACI, GPU 6)"
if [ -n "$SIGLIP_FEATURE_DIR" ]; then
    echo "    bash scripts/train_rdt_small.sh $TARGET_DIR checkpoints/rdt_small 6 $SIGLIP_FEATURE_DIR"
else
    echo "    bash scripts/train_rdt_small.sh $TARGET_DIR checkpoints/rdt_small 6"
fi
echo "    # RDT_FT requires the official RDT-1B checkpoint path as the 4th argument."
echo "    bash scripts/finetune_rdt.sh $TARGET_DIR checkpoints/rdt_ft 6 checkpoints/rdt-1b-pretrained ${SIGLIP_FEATURE_DIR:-}"
echo ""
