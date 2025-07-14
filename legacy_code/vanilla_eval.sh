#!/bin/bash
# Variables to set
SHORT_DESCRIPTION="phi_3_mini_vanilla"

echo "Running evaluation and analysis..."

bash evaluate_and_analyse.sh \
    --settings settings/settings_phi_3.yml \
    --short-description "${SHORT_DESCRIPTION}" \
    --only-model "Phi 3 Mini 4k Instruct" \
    --prompt "" \
    --temperature 0.1 \

echo "All epochs processed."
