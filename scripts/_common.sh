#!/bin/bash
# Shared logic sourced by all train_*.sh scripts.
#
# CALLER must set these variables before sourcing:
#   DATA_BASE_PATH   - root data directory
#   FEATURE_DIR      - (optional) root feature directory
#   TRAIN_SUFFIX     - train dir suffix used when falling back (e.g. _train)
#   TEST_SUFFIX      - test  dir suffix used when falling back (e.g. _test)
#
# After sourcing, the following variables are set:
#   MULTITASK              - true / false
#   TRAIN_PATH             - semicolon-separated training dirs
#   VAL_PATH               - semicolon-separated validation dirs
#   FEATURE_TRAIN_PATHS    - (only when FEATURE_DIR set) semicolon-separated
#   FEATURE_VAL_PATHS      - (only when FEATURE_DIR set) semicolon-separated
#   TOTAL_TRAIN / TOTAL_VAL - file counts
#
# NOTE: This file is sourced, not executed. It intentionally does NOT use
# "set -e" - each calling script manages its own errexit setting so that
# short-circuit expressions like "[ cond ] && cmd" don't kill the parent.
# Explicit "|| exit 1" guards protect against genuine fatal errors.

# ---- Python environment ------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active venv: $VIRTUAL_ENV"
elif command -v conda &>/dev/null; then
    echo "Activating conda environment: texas"
    source "$(conda info --base)/etc/profile.d/conda.sh" || {
        echo "ERROR: Could not find conda. Is conda on your PATH?"
        exit 1
    }
    conda activate texas
    if [[ "$CONDA_DEFAULT_ENV" != "texas" ]]; then
        echo "ERROR: Could not activate the 'texas' conda environment."
        exit 1
    fi
    echo "Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "ERROR: No Python environment found. Activate a venv or install conda."
    exit 1
fi

# ---- Normalize paths (readlink -f to resolve symlinks consistently) ---------
DATA_BASE_PATH=$(readlink -f "$DATA_BASE_PATH" 2>/dev/null || echo "$DATA_BASE_PATH")
if [ -n "$FEATURE_DIR" ]; then
    FEATURE_DIR=$(readlink -f "$FEATURE_DIR" 2>/dev/null || echo "$FEATURE_DIR")
fi

if [ ! -d "$DATA_BASE_PATH" ]; then
    echo "ERROR: Data path not found: $DATA_BASE_PATH"
    exit 1
fi
if [ -n "$FEATURE_DIR" ] && [ ! -d "$FEATURE_DIR" ]; then
    echo "ERROR: Feature directory not found: $FEATURE_DIR"
    echo "       Omit the feature-dir argument to run the visual backbone on the fly,"
    echo "       or run workflow/precompute_features.py / scripts/prepare.sh first."
    exit 1
fi

# ---- Auto-detect single-task vs multi-task ----------------------------------
_find_train_val_pair() {
    local base="$1"
    local train_dir=""
    local test_dir=""

    for train_glob in \
        "pick_up_card_train_*|pick_up_card_test" \
        "move_chips_train_*|move_chips_test" \
        "task_train*|task_test"
    do
        local train_pat="${train_glob%%|*}"
        local test_name="${train_glob##*|}"
        for candidate in "$base"/$train_pat; do
            [ -d "$candidate" ] || continue
            test_dir="$base/$test_name"
            [ -d "$test_dir" ] || continue
            train_dir="$candidate"
            printf '%s\n%s\n' "$train_dir" "$test_dir"
            return 0
        done
    done
    return 1
}

MULTITASK=false
TRAIN_PATH=""
VAL_PATH=""
FEATURE_TRAIN_PATHS=""
FEATURE_VAL_PATHS=""

INSTRUCTION_SUBDIRS=()
while IFS= read -r d; do
    INSTRUCTION_SUBDIRS+=("$d")
done < <(find -L "$DATA_BASE_PATH" -maxdepth 1 -mindepth 1 -type d | sort)

if [ ${#INSTRUCTION_SUBDIRS[@]} -gt 0 ]; then
    FIRST="${INSTRUCTION_SUBDIRS[0]}"
    IS_INSTR_DIR=false

    if _find_train_val_pair "$FIRST" >/dev/null; then
        IS_INSTR_DIR=true
    fi
    # Subdirs that ARE the train/val dirs themselves -> not instruction dirs
    if [ -d "${FIRST}${TRAIN_SUFFIX}" ] || [ -d "${FIRST}${TEST_SUFFIX}" ]; then
        IS_INSTR_DIR=false
    fi

    if $IS_INSTR_DIR; then
        MULTITASK=true
        VALID_TRAIN=()
        VALID_VAL=()

        for subdir in "${INSTRUCTION_SUBDIRS[@]}"; do
            pair=$(_find_train_val_pair "$subdir") || continue
            t=$(printf '%s\n' "$pair" | sed -n '1p')
            v=$(printf '%s\n' "$pair" | sed -n '2p')
            VALID_TRAIN+=("$(readlink -f "$t")")
            VALID_VAL+=("$(readlink -f "$v")")
            echo "Found instruction $(basename "$subdir"):  $(basename "$t")  |  $(basename "$v")"
        done

        if [ ${#VALID_TRAIN[@]} -eq 0 ]; then
            echo "ERROR: No valid train/test pairs found under $DATA_BASE_PATH"
            exit 1
        fi

        TRAIN_PATH=$(IFS=';'; echo "${VALID_TRAIN[*]}")
        VAL_PATH=$(IFS=';';   echo "${VALID_VAL[*]}")
        echo "Multi-task mode: ${#VALID_TRAIN[@]} instructions"

        if [ -n "$FEATURE_DIR" ]; then
            for t in "${VALID_TRAIN[@]}"; do
                rel="${t#${DATA_BASE_PATH}/}"
                FEATURE_TRAIN_PATHS+="${FEATURE_DIR}/${rel};"
            done
            for v in "${VALID_VAL[@]}"; do
                rel="${v#${DATA_BASE_PATH}/}"
                FEATURE_VAL_PATHS+="${FEATURE_DIR}/${rel};"
            done
            FEATURE_TRAIN_PATHS="${FEATURE_TRAIN_PATHS%;}"
            FEATURE_VAL_PATHS="${FEATURE_VAL_PATHS%;}"
        fi
    fi
fi

if ! $MULTITASK; then
    FOUND=false
    pair=$(_find_train_val_pair "$DATA_BASE_PATH" 2>/dev/null || true)
    if [ -n "$pair" ]; then
        t=$(printf '%s\n' "$pair" | sed -n '1p')
        v=$(printf '%s\n' "$pair" | sed -n '2p')
        TRAIN_PATH="$(readlink -f "$t")"
        VAL_PATH="$(readlink -f "$v")"
        FOUND=true
    fi

    if ! $FOUND; then
        t="${DATA_BASE_PATH}${TRAIN_SUFFIX}"
        v="${DATA_BASE_PATH}${TEST_SUFFIX}"
        if [ -d "$t" ] && [ -d "$v" ]; then
            TRAIN_PATH="$(readlink -f "$t")"
            VAL_PATH="$(readlink -f "$v")"
            FOUND=true
        fi
    fi

    if [ -n "$FEATURE_DIR" ] && [ -n "$TRAIN_PATH" ]; then
        FEATURE_TRAIN_PATHS="${FEATURE_DIR}/${TRAIN_PATH#${DATA_BASE_PATH}/}"
        FEATURE_VAL_PATHS="${FEATURE_DIR}/${VAL_PATH#${DATA_BASE_PATH}/}"
    fi

    if ! $FOUND; then
        echo "ERROR: Could not find train/val directories under $DATA_BASE_PATH"
        echo "Expected one of:"
        echo "  $DATA_BASE_PATH/pick_up_card_train_* + $DATA_BASE_PATH/pick_up_card_test"
        echo "  $DATA_BASE_PATH/move_chips_train_* + $DATA_BASE_PATH/move_chips_test"
        echo "  $DATA_BASE_PATH/task_train* + $DATA_BASE_PATH/task_test"
        echo "  $DATA_BASE_PATH${TRAIN_SUFFIX}"
        exit 1
    fi
    echo "Single-task mode: $TRAIN_PATH  |  $VAL_PATH"
fi

# ---- File counts (supports both .npz files and .npy directories) -----------
_count_episodes() {
    # Count data* entries: .npz files OR directories containing .npy files
    echo "$1" | tr ';' '\n' | \
        xargs -I{} sh -c '
            npz=$(find -L "{}" -maxdepth 1 -name "data*.npz" 2>/dev/null | wc -l)
            npy=$(find -L "{}" -maxdepth 1 -type d -name "data*" 2>/dev/null | wc -l)
            echo $((npz + npy))
        ' | awk '{s+=$1}END{print s}'
}
TOTAL_TRAIN=$(_count_episodes "$TRAIN_PATH")
TOTAL_VAL=$(_count_episodes "$VAL_PATH")

if [ "${TOTAL_TRAIN:-0}" -eq 0 ] || [ "${TOTAL_VAL:-0}" -eq 0 ]; then
    echo "ERROR: No episodes found (train=${TOTAL_TRAIN:-0}, val=${TOTAL_VAL:-0})"
    echo "Expected data*.npz files or data*/ directories with .npy files"
    exit 1
fi
