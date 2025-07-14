#!/bin/bash
# Variables to set
SHORT_DESCRIPTION="new_batch_base_llama_33"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${SHORT_DESCRIPTION}_${TIMESTAMP}"
LOG_DIR_NAME="$RUN_NAME"
STATS_PATH="stats_${RUN_NAME}.yml"
EVALUATE_OUT="${RUN_NAME}_evaluate_out.txt"
PROMPT="Read the following Metal-Organic Framework (MOF) synthesis description and extract this information: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), one main solvent (no mixtures or ratios), one chemical additive ('None' if no additive present). Important: Do not use JSON or curly braces in your output, they are already provided for you and you do not need to generate them. Only output the extracted information and terminate strings with \\\""

echo "Running evaluation and analysis..."

mkdir -p "runs/$RUN_NAME"

EVALUATE_CMD="LOG_LEVEL=DEBUG poetry run main evaluate \
    --settings settings/settings_llama_33_new_batch.yml \
    --log-dir \"$LOG_DIR_NAME\" \
    --stats-path \"$STATS_PATH\" \
    --only-model \"LLaMa 3.3 70B Instruct\" \
    --prompt  \"$PROMPT\" \
    --temperature 0.0 \
    --no-labels"

echo "===== EVALUATE COMMAND ====="
echo "$EVALUATE_CMD"
echo

eval "$EVALUATE_CMD" 2>&1 | tee "$EVALUATE_OUT"

if [[ -d "logs/"$LOG_DIR_NAME"_eval" ]]; then
    echo "Moving eval logs to runs/$RUN_NAME/logs"
    mkdir -p "runs/$RUN_NAME/logs/"
    mv "logs/$LOG_DIR_NAME"_eval "runs/$RUN_NAME/logs/eval"
fi

echo "All epochs processed."
