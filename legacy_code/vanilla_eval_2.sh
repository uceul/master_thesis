#!/bin/bash
# Variables to set
SHORT_DESCRIPTION="phi_3_mini_vanilla_prompt"

echo "Running evaluation and analysis..."

bash evaluate_and_analyse.sh \
    --settings settings/settings_phi_3.yml \
    --short-description "${SHORT_DESCRIPTION}" \
    --only-model "Phi 3 Mini 4k Instruct" \
    --prompt "Read the following Metal-Organic Framework (MOF) synthesis description and extract this information: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), one main solvent (no mixtures or ratios), one chemical additive ('None' if no additive present). Important: Do not use JSON or curly braces in your output, they are already provided for you and you do not need to generate them. Only output the extracted information and terminate strings with \\\"" \
    --temperature 0.1 \

echo "All epochs processed."
