#!/bin/bash

# Script to generate training commands for recovery RL policies
# Translated from gen_commands_htcondor_recrl.py

set -e

# Helper function to normalize paths (remove trailing slashes and double slashes)
normalize_path() {
    echo "$1" | sed 's:/*$::' | sed 's://*:/:g'
}

# Default values
LOG_DIRECTORY="./logs"
TOTAL_TIMESTEPS=35000
GOAL_REWARD=1.0
FAILURE_REWARD=-1.0
UNSAFE_REWARD=-0.01
MAX_STEPS=1024
DEVICE="cuda"
SEED=42
DISABLE_WANDB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --condor_dir_prefix)
            CONDOR_DIR_PREFIX="$2"
            shift 2
            ;;
        --log_directory)
            LOG_DIRECTORY="$2"
            shift 2
            ;;
        --total_timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --goal_reward)
            GOAL_REWARD="$2"
            shift 2
            ;;
        --failure_reward)
            FAILURE_REWARD="$2"
            shift 2
            ;;
        --unsafe_reward)
            UNSAFE_REWARD="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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
        --disable_wandb)
            DISABLE_WANDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ROOT" ]]; then
    echo "Error: --root is required"
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
ROOT=$(normalize_path "$ROOT")
CONDOR_DIR_PREFIX=$(normalize_path "$CONDOR_DIR_PREFIX")
LOG_DIRECTORY=$(normalize_path "$LOG_DIRECTORY")

# Clear output file
> "$OUTPUT_FILE"

# Function to process a single model file within a variant directory
process_model_file() {
    local model_file=$(normalize_path "$1")
    local variant_dir=$(normalize_path "$2")
    local domain_dir=$(normalize_path "$3")

    local variant_name=$(basename "$variant_dir")
    local domain_name=$(basename "$domain_dir")
    local jani_name=$(basename "$model_file" .jani)

    local recover_policy_path
    local q_risk_path
    local offline_buffer_path
    local model_save_dir
    local log_dir
    local property_dir

    if [[ "$variant_name" == "models" ]]; then
        property_dir="${domain_dir}/additional_properties"
        recover_policy_path="${domain_dir}/iql_policies/${jani_name}/iql_actor.pth"
        q_risk_path="${domain_dir}/iql_policies/${jani_name}/iql_q_net.pth"
        offline_buffer_path="${domain_dir}/iql_policies/${jani_name}/replay_buffer"
        model_save_dir="${CONDOR_DIR_PREFIX}/${domain_dir}/recovery_policies/${jani_name}"
        log_dir="${CONDOR_DIR_PREFIX}/${LOG_DIRECTORY}/${domain_name}/${jani_name}"
    else
        property_dir="${domain_dir}/additional_properties/${variant_name}"
        recover_policy_path="${domain_dir}/iql_policies/${variant_name}/${jani_name}/iql_actor.pth"
        q_risk_path="${domain_dir}/iql_policies/${variant_name}/${jani_name}/iql_q_net.pth"
        offline_buffer_path="${domain_dir}/iql_policies/${variant_name}/${jani_name}/replay_buffer"
        model_save_dir="${CONDOR_DIR_PREFIX}/${domain_dir}/recovery_policies/${variant_name}/${jani_name}"
        log_dir="${CONDOR_DIR_PREFIX}/${LOG_DIRECTORY}/${domain_name}/${variant_name}/${jani_name}"
    fi

    # Validate that required files exist
    if [[ ! -f "$recover_policy_path" ]]; then
        echo "Error: Recovery policy file $recover_policy_path does not exist." >&2
        exit 1
    fi
    if [[ ! -f "$q_risk_path" ]]; then
        echo "Error: Q risk file $q_risk_path does not exist." >&2
        exit 1
    fi
    if [[ ! -d "$offline_buffer_path" ]]; then
        echo "Error: Offline buffer directory $offline_buffer_path does not exist." >&2
        exit 1
    fi

    # Add condor prefix to paths
    local condor_recover_policy_path="${CONDOR_DIR_PREFIX}/${recover_policy_path}"
    local condor_q_risk_path="${CONDOR_DIR_PREFIX}/${q_risk_path}"
    local condor_offline_buffer_path="${CONDOR_DIR_PREFIX}/${offline_buffer_path}"

    # Locate training property file
    local training_property_dir="${property_dir}/random_starts_20000/${jani_name}"
    if [[ ! -d "$training_property_dir" ]]; then
        echo "Error: Training property directory $training_property_dir does not exist." >&2
        exit 1
    fi

    local training_property_files=("$training_property_dir"/*)
    if [[ ${#training_property_files[@]} -ne 1 ]]; then
        echo "Error: Expected one property file in $training_property_dir, found ${#training_property_files[@]}" >&2
        exit 1
    fi
    local training_property_file="${training_property_files[0]}"

    # Locate evaluation property file
    local eval_property_dir="${property_dir}/random_starts_5000/${jani_name}"
    if [[ ! -d "$eval_property_dir" ]]; then
        echo "Error: Evaluation property directory $eval_property_dir does not exist." >&2
        exit 1
    fi

    local eval_property_files=("$eval_property_dir"/*)
    if [[ ${#eval_property_files[@]} -ne 1 ]]; then
        echo "Error: Expected one property file in $eval_property_dir, found ${#eval_property_files[@]}" >&2
        exit 1
    fi
    local eval_property_file="${eval_property_files[0]}"

    # Build experiment name
    local experiment_name
    if [[ "$variant_name" == "models" ]]; then
        experiment_name="${jani_name}_recovery_rl"
    else
        experiment_name="${jani_name}_${variant_name}_recovery_rl"
    fi

    # Build the command line
    local cmd="-m recovery_rl.train"
    cmd+=" --jani_model ${CONDOR_DIR_PREFIX}/${model_file}"
    cmd+=" --jani_property ${CONDOR_DIR_PREFIX}/${training_property_file}"
    cmd+=" --start_states ${CONDOR_DIR_PREFIX}/${training_property_file}"
    cmd+=" --eval_start_states ${CONDOR_DIR_PREFIX}/${eval_property_file}"
    cmd+=" --goal_reward ${GOAL_REWARD}"
    cmd+=" --failure_reward ${FAILURE_REWARD}"
    cmd+=" --unsafe_reward ${UNSAFE_REWARD}"
    cmd+=" --recovery_policy_path ${condor_recover_policy_path}"
    cmd+=" --q_risk_path ${condor_q_risk_path}"
    cmd+=" --offline_buffer_path ${condor_offline_buffer_path}"
    cmd+=" --max_steps ${MAX_STEPS}"
    cmd+=" --total_timesteps ${TOTAL_TIMESTEPS}"
    cmd+=" --wandb_project ${jani_name}_clean"
    cmd+=" --experiment_name ${experiment_name}"
    cmd+=" --log_dir ${log_dir}"
    cmd+=" --model_save_dir ${model_save_dir}"
    cmd+=" --verbose 0"
    cmd+=" --device ${DEVICE}"
    cmd+=" --seed ${SEED}"

    if [[ "$DISABLE_WANDB" == true ]]; then
        cmd+=" --disable_wandb"
    fi

    echo "$cmd"
}

# Function to process a variant directory (contains .jani model files)
process_variant_dir() {
    local variant_dir=$(normalize_path "$1")
    local domain_dir=$(normalize_path "$2")

    for model_file in "$variant_dir"/*.jani; do
        if [[ -f "$model_file" ]]; then
            process_model_file "$model_file" "$variant_dir" "$domain_dir"
        fi
    done
}

# Function to process a domain directory
process_domain_dir() {
    local domain_dir=$(normalize_path "$1")
    local model_dir="${domain_dir}/models"

    if [[ ! -d "$model_dir" ]]; then
        return
    fi

    # Check if model_dir contains subdirectories (variants) or files directly
    local has_subdirs=false
    local has_files=false

    for entry in "$model_dir"/*; do
        if [[ -d "$entry" ]]; then
            has_subdirs=true
        elif [[ -f "$entry" ]]; then
            has_files=true
        fi
    done

    if [[ "$has_subdirs" == true && "$has_files" == true ]]; then
        echo "Error: Model directory $model_dir contains a mix of files and directories." >&2
        exit 1
    fi

    if [[ "$has_subdirs" == true ]]; then
        # Process each variant subdirectory
        for variant_dir in "$model_dir"/*/; do
            if [[ -d "$variant_dir" ]]; then
                process_variant_dir "$variant_dir" "$domain_dir"
            fi
        done
    elif [[ "$has_files" == true ]]; then
        # Process model_dir directly as the variant directory
        process_variant_dir "$model_dir" "$domain_dir"
    fi
}

# Main: iterate through all domain directories
FIRST_LINE=true
for domain_dir in "$ROOT"/*/; do
    if [[ -d "$domain_dir" ]]; then
        while IFS= read -r line; do
            if [[ "$FIRST_LINE" == true ]]; then
                echo -n "$line" >> "$OUTPUT_FILE"
                FIRST_LINE=false
            else
                echo "" >> "$OUTPUT_FILE"
                echo -n "$line" >> "$OUTPUT_FILE"
            fi
        done < <(process_domain_dir "$domain_dir")
    fi
done

echo "Generated commands written to $OUTPUT_FILE"
