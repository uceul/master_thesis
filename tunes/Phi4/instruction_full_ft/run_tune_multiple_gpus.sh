#!/bin/bash

echo "ATTENTION: make sure to use torchtune >= 0.6.0 for Phi 4 to work! (Check Torchtune Github for nightly install guide)"

# Check if output_dir parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

# Get the output directory and add a timestamp
output_dir=$1
timestamp=$(date +%s)
output_dir_with_timestamp="/hkfs/work/workspace_haic/scratch/zn2950-llms/tunes/phi4/instruction_full_ft/${output_dir%/}_${timestamp}"

mkdir -p "$output_dir_with_timestamp/logs"
# Play the tune!
tune run --nproc_per_node 2 full_finetune_distributed --config config.yaml output_dir="$output_dir_with_timestamp" 2>&1 | tee output.log
cp config.yaml "$output_dir_with_timestamp/"
mv output.log "$output_dir_with_timestamp/logs/"

