#!/bin/bash

# Script to generate trajectory-sampling commands for HTCondor
# Translated from gen_commands_htcondor_sample.py

set -e

# Helper function to normalize paths (remove trailing slashes and double slashes)
normalize_path() {
    echo "$1" | sed 's:/*$::' | sed 's://*:/:g'
}

# Join a prefix onto a path, replicating pathlib's `/` operator semantics:
# if the second path is absolute, the prefix is discarded and the absolute
# path is returned unchanged; otherwise the two are concatenated. An empty
# prefix leaves the path unchanged. The result is normalized.
join_with_prefix() {
    local prefix="$1"
    local path="$2"
    if [[ "$path" == /* ]]; then
        # Absolute path: pathlib drops the prefix entirely.
        normalize_path "$path"
    elif [[ -z "$prefix" ]]; then
        normalize_path "$path"
    else
        normalize_path "${prefix}/${path}"
    fi
}

# Default values
CONDOR_DIR_PREFIX=""
NUM_EPISODES=10000
MAX_STEPS=256
TARGET_SAFE_RATIO=""
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --condor_dir_prefix)
            CONDOR_DIR_PREFIX="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --target_safe_ratio)
            TARGET_SAFE_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ROOT_DIR" ]]; then
    echo "Error: --root_dir is required"
    exit 1
fi

if [[ -z "$CONDOR_DIR_PREFIX" ]]; then
    echo "Error: --condor_dir_prefix is required"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: --output_file is required"
    exit 1
fi

# Normalize input paths (strip trailing slashes)
ROOT_DIR=$(normalize_path "$ROOT_DIR")
CONDOR_DIR_PREFIX=$(normalize_path "$CONDOR_DIR_PREFIX")

# Clear output file
> "$OUTPUT_FILE"

# Function to process a single benchmark directory (contains exactly 2 .jani files)
process_benchmark_dir() {
    local benchmark_dir=$(normalize_path "$1")

    # Expect exactly 2 .jani files in the benchmark directory
    local jani_files=("$benchmark_dir"/*.jani)
    if [[ ${#jani_files[@]} -ne 2 ]]; then
        echo "Error: Expected exactly 2 .jani files in $benchmark_dir, found ${#jani_files[@]}" >&2
        exit 1
    fi

    # Use fixed filenames for the model and property files. These are the
    # local (un-prefixed) paths used for existence checks, matching the Python
    # which asserts .exists() before applying the prefix.
    local model_file=$(normalize_path "${benchmark_dir}/model.jani")
    local property_file=$(normalize_path "${benchmark_dir}/pa_model_random_starts_100000.jani")

    if [[ ! -f "$model_file" ]]; then
        echo "Error: Expected model.jani in $benchmark_dir but it does not exist." >&2
        exit 1
    fi
    if [[ ! -f "$property_file" ]]; then
        echo "Error: Expected pa_model_random_starts_100000.jani in $benchmark_dir but it does not exist." >&2
        exit 1
    fi

    # Apply the condor prefix to every emitted path. join_with_prefix
    # replicates pathlib's `/` semantics (an absolute path discards the prefix).
    local model_path=$(join_with_prefix "$CONDOR_DIR_PREFIX" "$model_file")
    local property_path=$(join_with_prefix "$CONDOR_DIR_PREFIX" "$property_file")
    # Output directory for sampled trajectories: prefix / benchmark_dir / "sampled_trajectories"
    local save_trajs_dir=$(join_with_prefix "$CONDOR_DIR_PREFIX" "${benchmark_dir}/sampled_trajectories")

    # Build the command line.
    # Mirrors the Python ordering and the truthiness filtering: only emit
    # --target_safe_ratio when a value was provided, and --reduced_memory_mode
    # as a bare flag (it is always True in the Python config).
    local cmd="-m offline_rl.sample"
    cmd+=" --model_path ${model_path}"
    cmd+=" --property_path ${property_path}"
    cmd+=" --start_states ${property_path}"
    cmd+=" --num_episodes ${NUM_EPISODES}"
    cmd+=" --max_steps ${MAX_STEPS}"
    cmd+=" --output_dir ${save_trajs_dir}"
    if [[ -n "$TARGET_SAFE_RATIO" ]]; then
        cmd+=" --target_safe_ratio ${TARGET_SAFE_RATIO}"
    fi
    cmd+=" --seed ${SEED}"
    cmd+=" --reduced_memory_mode"

    echo "$cmd"
}

# Main: iterate root/domain/benchmark (two levels deep)
FIRST_LINE=true
for domain_dir in "$ROOT_DIR"/*/; do
    if [[ ! -d "$domain_dir" ]]; then
        continue
    fi
    for benchmark_dir in "$domain_dir"*/; do
        if [[ ! -d "$benchmark_dir" ]]; then
            continue
        fi
        while IFS= read -r line; do
            if [[ "$FIRST_LINE" == true ]]; then
                echo -n "$line" >> "$OUTPUT_FILE"
                FIRST_LINE=false
            else
                echo "" >> "$OUTPUT_FILE"
                echo -n "$line" >> "$OUTPUT_FILE"
            fi
        done < <(process_benchmark_dir "$benchmark_dir")
    done
done

echo "Generated commands written to $OUTPUT_FILE"
