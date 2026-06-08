#!/bin/bash

# Script to generate offline-RL training commands for HTCondor
# Translated from gen_commands_htcondor_offlinerl.py
#
# For each benchmark, emits two commands (vanilla + lower-bound) per experiment,
# for num_exps experiments with varying seeds.

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

# Drop the first path component, replicating Python's Path(*p.parts[1:]).
# pathlib treats the leading "/" of an absolute path as its own first part,
# so:
#   relative "root/a/b" -> "a/b"   (drop the first real component)
#   absolute "/x/a/b"   -> "x/a/b" (drop only the leading "/", keep "x")
# The result is always relative, and is normalized.
strip_first_component() {
    local path=$(normalize_path "$1")
    if [[ "$path" == /* ]]; then
        # Absolute: parts[0] is "/", so just drop the leading slash.
        path="${path#/}"
    elif [[ "$path" == */* ]]; then
        # Relative with >1 component: drop the first component.
        path="${path#*/}"
    else
        # Single relative component; Path(*parts[1:]) of one part is ".".
        path="."
    fi
    normalize_path "$path"
}

# Default values
CONDOR_DIR_PREFIX=""
SAFE_RATIO=""
MAX_STEPS=256
TOTAL_TIMESTEPS=50000
STEPS_PER_EPOCH=50
BATCH_SIZE=128
NUM_EXPS=20
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
        --safe_ratio)
            SAFE_RATIO="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --total_timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --steps_per_epoch)
            STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_exps)
            NUM_EXPS="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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

if [[ -z "$LOG_DIR" ]]; then
    echo "Error: --log_dir is required"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: --output_file is required"
    exit 1
fi

# Normalize input paths (strip trailing slashes)
ROOT_DIR=$(normalize_path "$ROOT_DIR")
CONDOR_DIR_PREFIX=$(normalize_path "$CONDOR_DIR_PREFIX")
LOG_DIR=$(normalize_path "$LOG_DIR")

# Clear output file
> "$OUTPUT_FILE"

# Function to process a single benchmark directory.
# Emits one command per line (vanilla and lower-bound, for each experiment).
process_benchmark_dir() {
    local benchmark_dir=$(normalize_path "$1")

    # Expect exactly 2 .jani files in the benchmark directory
    local jani_files=("$benchmark_dir"/*.jani)
    if [[ ${#jani_files[@]} -ne 2 ]]; then
        echo "Error: Expected exactly 2 .jani files in $benchmark_dir, found ${#jani_files[@]}" >&2
        exit 1
    fi

    # Fixed filenames for the model and property files (local, un-prefixed paths
    # used for existence checks, matching the Python which asserts before prefix).
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

    # Log directory: log_dir / benchmark_dir_with_first_component_stripped
    local benchmark_suffix=$(strip_first_component "$benchmark_dir")
    local benchmark_log_dir=$(normalize_path "${LOG_DIR}/${benchmark_suffix}")
    mkdir -p "$benchmark_log_dir"

    # Dataset path depends on safe_ratio
    local dataset_path
    if [[ -n "$SAFE_RATIO" ]]; then
        dataset_path=$(normalize_path "${benchmark_dir}/sampled_trajectories_${SAFE_RATIO}")
    else
        dataset_path=$(normalize_path "${benchmark_dir}/sampled_trajectories")
    fi
    if [[ ! -e "$dataset_path" ]]; then
        echo "Error: Expected sampled trajectories at $dataset_path but it does not exist." >&2
        exit 1
    fi

    # Apply the condor prefix to the paths shared across all experiments.
    local model_path=$(join_with_prefix "$CONDOR_DIR_PREFIX" "$model_file")
    local property_path=$(join_with_prefix "$CONDOR_DIR_PREFIX" "$property_file")
    local dataset_path_prefixed=$(join_with_prefix "$CONDOR_DIR_PREFIX" "$dataset_path")

    # Generate num_exps experiments, each with a varied seed and two configs.
    for (( exp_idx=0; exp_idx<NUM_EXPS; exp_idx++ )); do
        local exp_seed=$(( SEED + exp_idx ))
        local exp_log_dir=$(normalize_path "${benchmark_log_dir}/seed_${exp_seed}")
        mkdir -p "$exp_log_dir"

        local eval_vanilla=$(join_with_prefix "$CONDOR_DIR_PREFIX" "${exp_log_dir}/eval_results_vanilla.json")
        local eval_lower_bound=$(join_with_prefix "$CONDOR_DIR_PREFIX" "${exp_log_dir}/eval_results_lower_bound.json")

        # Vanilla config.
        # Note: --unsafe_reward is 0.0 in the Python config; the truthiness
        # filter there drops zero values, so it is intentionally NOT emitted.
        local cmd_vanilla="-m offline_rl.iql"
        cmd_vanilla+=" --model_path ${model_path}"
        cmd_vanilla+=" --property_path ${property_path}"
        cmd_vanilla+=" --start_states ${property_path}"
        cmd_vanilla+=" --dataset_path ${dataset_path_prefixed}"
        cmd_vanilla+=" --max_steps_per_episode ${MAX_STEPS}"
        cmd_vanilla+=" --total_timesteps ${TOTAL_TIMESTEPS}"
        cmd_vanilla+=" --steps_per_epoch ${STEPS_PER_EPOCH}"
        cmd_vanilla+=" --batch_size ${BATCH_SIZE}"
        cmd_vanilla+=" --write_eval_results ${eval_vanilla}"
        cmd_vanilla+=" --seed ${exp_seed}"
        echo "$cmd_vanilla"

        # Lower-bound config.
        local cmd_lower_bound="-m offline_rl.sample"
        cmd_lower_bound+=" --model_path ${model_path}"
        cmd_lower_bound+=" --property_path ${property_path}"
        cmd_lower_bound+=" --start_states ${property_path}"
        cmd_lower_bound+=" --dataset_path ${dataset_path_prefixed}"
        cmd_lower_bound+=" --max_steps_per_episode ${MAX_STEPS}"
        cmd_lower_bound+=" --total_timesteps ${TOTAL_TIMESTEPS}"
        cmd_lower_bound+=" --steps_per_epoch ${STEPS_PER_EPOCH}"
        cmd_lower_bound+=" --batch_size ${BATCH_SIZE}"
        cmd_lower_bound+=" --use_lower_bound"
        cmd_lower_bound+=" --lower_bound_type action_safe"
        cmd_lower_bound+=" --write_eval_results ${eval_lower_bound}"
        cmd_lower_bound+=" --seed ${exp_seed}"
        echo "$cmd_lower_bound"
    done
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
