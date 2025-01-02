#!/usr/bin/env bash
#
# Automates:
#   1) main evaluate
#   2) main analyse
#   3) Collecting output into runs/<short_description>_<timestamp>/
#
# Example usage:
#   ./run_all.sh --settings settings/settings_llama_31.yml \ 
#       --only-model "LLaMa 3.1 8B Instruct" \
#       --model_path <path>  Model path passed to 'evaluate'
#       --prompt "test prompt" \
#       --temperature 0.2 \
#       --description "Some run description" \
#       --short_description "my_experiment"
#
# After finishing, you'll have:
#   runs/my_experiment_20230817_102015/
#       |- my_experiment_20230817_102015_evaluate_out.txt
#       |- my_experiment_20230817_102015_analyse_out.txt
#       |- stats_my_experiment_20230817_102015.yml
#       |- settings_llama_31.yml   (copy of your settings)
#       \- logs/                   (log directory)

set -e  # Exit on error
trap "popd > /dev/null" EXIT
pushd /gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/master_thesis/legacy_code > /dev/null

# Default parameter values
TEMPERATURE="0.1"
ONLY_MODEL=""
MODEL_PATH=""
PROMPT=""
DESCRIPTION=""
SHORT_DESCRIPTION=""

# Print usage
usage() {
  cat <<EOF
Usage: $0 --settings <path/to/settings.yml> [options]

Required:
  --settings <path>   Path to the settings YAML file
  --short_description <str>
                      Used as prefix for the run directory name

Optional:
  --only-model <str>  Only evaluate a specific model by name
  --prompt <str>      Prompt passed to 'evaluate'
  --temperature <num> Temperature passed to 'evaluate' (default: 0.1)
  --description <str> Description for both 'evaluate' and 'analyse'
EOF
  exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --settings)
      SETTINGS="$2"
      shift 2
      ;;
    --only-model)
      ONLY_MODEL="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --description)
      DESCRIPTION="$2"
      shift 2
      ;;
    --short_description)
      SHORT_DESCRIPTION="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown argument: $1"
      usage
      ;;
  esac
done

# Check required parameters
if [[ -z "$SETTINGS" ]]; then
  echo "Error: --settings is required."
  usage
fi

if [[ -z "$SHORT_DESCRIPTION" ]]; then
  echo "Error: --short_description is required."
  usage
fi

# Generate timestamp and construct names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${SHORT_DESCRIPTION}_${TIMESTAMP}"
LOG_DIR_NAME="$RUN_NAME"

# Create unique filenames for this run
STATS_PATH="stats_${RUN_NAME}.yml"
EVALUATE_OUT="${RUN_NAME}_evaluate_out.txt"
ANALYSE_OUT="${RUN_NAME}_analyse_out.txt"

# Build the 'evaluate' command
EVALUATE_CMD="LOG_LEVEL=DEBUG poetry run main evaluate \
  --settings \"$SETTINGS\" \
  --stats-path \"$STATS_PATH\" \
  --temperature \"$TEMPERATURE\" \
  --log-dir \"$LOG_DIR_NAME\""

if [[ -n "$ONLY_MODEL" ]]; then
  EVALUATE_CMD+=" --only-model \"$ONLY_MODEL\""
fi
if [[ -n "$MODEL_PATH" ]]; then
  EVALUATE_CMD+=" --model-path \"$MODEL_PATH\""
fi
if [[ -n "$PROMPT" ]]; then
  EVALUATE_CMD+=" --prompt \"$PROMPT\""
fi
if [[ -n "$DESCRIPTION" ]]; then
  EVALUATE_CMD+=" --description \"$DESCRIPTION\""
fi

# Build the 'analyse' command
ANALYSE_CMD="LOG_LEVEL=DEBUG poetry run main analyse \
  --settings \"$SETTINGS\" \
  --stats-path \"$STATS_PATH\" \
  --log-dir \"$LOG_DIR_NAME\""

if [[ -n "$DESCRIPTION" ]]; then
  ANALYSE_CMD+=" --description \"$DESCRIPTION\""
fi

# Create run directory and move files
echo "Creating directory runs/$RUN_NAME..."
mkdir -p "runs/$RUN_NAME"

# Print commands for user inspection
echo "===== EVALUATE COMMAND ====="
echo "$EVALUATE_CMD"
echo

mkdir -p "runs/$RUN_NAME/logs"
if [[ -d "logs/$LOG_DIR_NAME" ]]; then
    echo "Moving evaluation logs to runs/$RUN_NAME/logs"
    mv "logs/$LOG_DIR_NAME" "runs/$RUN_NAME/logs/eval"
fi


echo "===== ANALYSE COMMAND ====="
echo "$ANALYSE_CMD"
echo

# Run evaluation
echo "[1/2] Running EVALUATE..."
eval "$EVALUATE_CMD" 2>&1 | tee "$EVALUATE_OUT"

# Run analysis
echo "[2/2] Running ANALYSE..."
eval "$ANALYSE_CMD" 2>&1 | tee "$ANALYSE_OUT"



# Move all generated files
mv "$EVALUATE_OUT" "runs/$RUN_NAME/"
mv "$ANALYSE_OUT" "runs/$RUN_NAME/"
mv "$STATS_PATH" "runs/$RUN_NAME/"
cp "$SETTINGS" "runs/$RUN_NAME/"

# Move logs directory
if [[ -d "logs/$LOG_DIR_NAME" ]]; then
    echo "Moving analyse logs to runs/$RUN_NAME/logs"
    mv "logs/$LOG_DIR_NAME" "runs/$RUN_NAME/logs/analyse"
fi

echo
echo "All done. Results saved in runs/$RUN_NAME"