#!/bin/bash

PROMPTS=(
  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temperature, use 25°C if not specified), time (Longest duration maintained at highest temperature), choose one main solvent (no mixtures or ratios), choose one chemical additive (write 'None' if no additives present)"
  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: choose one chemical additive (write 'None' if no additives present), choose one main solvent (no mixtures or ratios), temperature (highest reaction temperature, use 25°C if not specified), time (Longest duration maintained at highest temperature)"
  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), solvent (choose one main solvent, no mixtures or ratios), and one chemical additive (write 'None' if no additive present)"
)

# Iterate through each prompt in the list
COUNTER=0
for PROMPT in "${PROMPTS[@]}"; do
    echo "Executing: bash evaluate_and_analyse.sh with prompt \"$COUNTER\": \"$PROMPT\""
    
    # Base command
    COMMAND="bash evaluate_and_analyse.sh --settings settings/settings_llama_31.yml \
                                          --short-description fourth_prompt_test_$COUNTER \
                                          --only-model \"LLaMa 3.1 8B Instruct\" \
                                          --temperature 0.0"
    
    if [ -n "$PROMPT" ]; then
        COMMAND="$COMMAND --prompt \"$PROMPT\""
    fi
    
    eval $COMMAND
    COUNTER=$((COUNTER + 1))
done