#!/bin/bash
# Variables to set
SHORT_DESCRIPTION="llama_32_lora_run_1"
BASE_DIR="/home/iti/zn2950/home/haicore_ws/tunes/llama32_1b/instruction_lora/run_1_1736269137"
EPOCH_LIST=(0 1 2 3 4)  # Add the epochs you want to evaluate here

# Script execution loop
for EPOCH in "${EPOCH_LIST[@]}"; do
    FULL_SHORT_DESCRIPTION="${SHORT_DESCRIPTION}_epoch_${EPOCH}"
    MODEL_PATH="${BASE_DIR}/epoch_${EPOCH}"

    echo "Running evaluation and analysis for epoch ${EPOCH}..."

    bash evaluate_and_analyse.sh \
        --settings settings/settings_llama_32.yml \
        --short-description "${FULL_SHORT_DESCRIPTION}" \
        --only-model "LLaMa 3.2 1B Instruct" \
        --model-path "${MODEL_PATH}" \
        --prompt "Read the following Metal-Organic Framework (MOF) synthesis description and extract this information: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), one main solvent (no mixtures or ratios), one chemical additive ('None' if no additive present). Important: Do not use JSON or curly braces in your output, they are already provided for you and you do not need to generate them. Only output the extracted information and terminate strings with \\\"" \
        --temperature 0.0 \
        --evaluation-set

done

echo "All epochs processed."
