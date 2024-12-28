#!/bin/bash

# Load necessary modules
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/11.8

# Activate conda environment
conda activate llm-extraction

# Export the PATH to include conda environment
export PATH="/home/kit/iti/zn2950/.conda/envs/llm-extraction/bin:$PATH"

# Start code-server with the correct environment
code-server "$@"
