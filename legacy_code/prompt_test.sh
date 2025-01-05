#!/bin/bash

PROMPTS=(
  ""
  "Extract MOF synthesis conditions from the text. Use these rules: 
  - Temperature: Highest reaction temperature (25°C if unspecified) 
  - Time: Longest duration maintained at highest temperature 
  - Solvent: Choose one main solvent (no mixtures or ratios) 
  - Additive: Choose one additive, or write 'None' if absent"
  "Extract synthesis conditions from the text in the following format: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), one solvent (if multiple present, choose one), and one additive (write 'None' if no additive present)."
  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), solvent (choose one main solvent, no mixtures or ratios), and one additive (write 'None' if no additive present)"
  "Extract synthesis conditions from this Metal-Organic Framework (MOF) synthesis text:
  - Temperature: highest sustained reaction temperature (use 25 degrees celsius for room temperature or if not specified)
  - Time: longest duration at highest temperature
  - Solvent: primary reaction solvent (select one main solvent, not mixtures)
  - Additive: one chemical additive (write 'None' if no additives present)"
  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), solvent (choose one main solvent, no mixtures or ratios), and one chemical additive (write 'None' if no additive present)"
  "Extract synthesis conditions from this Metal-Organic Framework (MOF) synthesis text:
  - Temperature: highest sustained reaction temperature (use 25 degrees celsius for room temperature or if not specified)
  - Time: longest duration at highest temperature
  - Solvent: primary reaction solvent (select one main solvent, not mixtures)
  - Additive: one chemical additive (write 'None' if no additives present)"

  "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temperature, use 25°C if not specified), time (longest duration at highest temperature), choose one main solvent (no mixtures or ratios), choose one chemical additive (write 'None' if no additives present)"
  
  "Extract synthesis conditions from this Metal-Organic Framework (MOF) synthesis text:
  - Temperature: highest reaction temperature (use 25°C if not specified)
  - Time: longest duration at highest temperature
  - Solvent: Choose one main solvent (no mixtures or ratios)
  - Additive: one chemical additive (write 'None' if no additives present)"

  "Extract synthesis conditions from this Metal-Organic Framework (MOF) synthesis text:
  - Additive: one chemical additive (write 'None' if no additives present)
  - Solvent: primary reaction solvent (select one main solvent, not mixtures)
  - Temperature: highest sustained reaction temperature (use 25 degrees celsius for room temperature or if not specified)
  - Time: longest duration at highest temperature"
)

# Iterate through each prompt in the list
COUNTER=0
for PROMPT in "${PROMPTS[@]}"; do
    echo "Executing: bash evaluate_and_analyse.sh with prompt \"$COUNTER\": \"$PROMPT\""
    
    # Base command
    COMMAND="bash evaluate_and_analyse.sh --settings settings/settings_llama_31.yml \
                                          --short-description third_prompt_test_$COUNTER \
                                          --only-model \"LLaMa 3.1 8B Instruct\" \
                                          --temperature 0.0"
    
    if [ -n "$PROMPT" ]; then
        COMMAND="$COMMAND --prompt \"$PROMPT\""
    fi
    
    eval $COMMAND
    COUNTER=$((COUNTER + 1))
done