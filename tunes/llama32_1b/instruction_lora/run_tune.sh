#!/bin/bash

# Check if output_dir parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

# Get the output directory and add a timestamp
output_dir=$1
timestamp=$(date +%s)
output_dir_with_timestamp="/hkfs/work/workspace_haic/scratch/zn2950-llms/tunes/llama32_1b/instruction_lora/${output_dir%/}_${timestamp}"

mkdir -p "$output_dir_with_timestamp/logs"
# Play the tune!
tune run lora_finetune_single_device --config config.yaml output_dir="$output_dir_with_timestamp" 2>&1 | tee output.log
cp config.yaml "$output_dir_with_timestamp/"
mv output.log "$output_dir_with_timestamp/logs/"
