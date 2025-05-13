#!/bin/bash

# List of thresholds you want to test
thresholds=(0.5 0.6 0.7 0.8 0.9)

# Base output directory
base_dir="experiments"

# Create base directory if it doesn't exist
mkdir -p "$base_dir"

# Loop over thresholds
for threshold in "${thresholds[@]}"; do
    # Format threshold for directory name (e.g., 0.5 -> 05, 0.9 -> 09)
    formatted_thresh=$(echo "$threshold" | sed 's/\.//g')

    # Define output directory and file
    output_dir="${base_dir}/cai_evaluation_${formatted_thresh}"
    output_file="${output_dir}/comparison_full.json"

    # Create output directory
    mkdir -p "$output_dir"

    echo "Running evaluation with confidence threshold $threshold..."
    python test_compound_ai_system.py \
        --router-model-path ./router_model \
        --small-model llama3 \
        --large-model-type claude \
        --num-samples 500 \
        --baseline \
        --confidence-threshold "$threshold" \
        --output-file "$output_file"

    echo "Finished evaluation for threshold $threshold. Results saved to $output_file"
    echo "--------------------------------------------------------"
done
