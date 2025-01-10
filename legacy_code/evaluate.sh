#!/usr/bin/env bash

set -e
trap "popd > /dev/null" EXIT
pushd /gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/master_thesis/legacy_code > /dev/null

# Default parameter values
TEMPERATURE="0.1"
ONLY_MODEL=""
MODEL_PATH=""
PROMPT=""
DESCRIPTION=""
SHORT_DESCRIPTION=""
EVALUATION_SET=false
EVALUATION_BOTH=false

# Print usage
usage() {
  cat <<EOF
Usage: $0 --settings <path/to/settings.yml> [options]

Required:
  --settings <path>   Path to the settings YAML file
  --short-description <str>
                      Used as prefix for the run directory name

Optional:
  --only-model <str>  Only evaluate a specific model by name
  --model-path <str>  Model path passed to 'evaluate'
  --prompt <str>      Prompt passed to 'evaluate'
  --temperature <num> Temperature passed to 'evaluate' (default: 0.1)
  --description <str> Description for 'evaluate'
  --evaluation-set <str>   Evaluation set path passed to 'evaluate'
EOF
  exit 1
}

# Parse arguments
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
    --model-path)
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
    --short-description)
      SHORT_DESCRIPTION="$2"
      shift 2
      ;;
    --evaluation-set)
      EVALUATION_SET=true
      shift 1
      ;;
    --evaluation-both)
      EVALUATION_BOTH=true
      shift 1
      ;;
    *)
      echo "Error: Unknown argument: $1"
      usage
      ;;
  esac
done

# Check required parameters
if [[ -z "$SETTINGS" || -z "$SHORT_DESCRIPTION" ]]; then
  usage
fi

if [[ "$EVALUATION_SET" == false ]]; then
  if [[ "$EVALUATION_BOTH" == true ]]; then
  echo "ERROR: --evaluation-set is required if --evaluation-both is set."
  usage
  fi
fi

# Generate timestamp and construct names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${SHORT_DESCRIPTION}_${TIMESTAMP}"
EVALUATE_OUT="${RUN_NAME}_evaluate_out.txt"
LOG_DIR_NAME="$RUN_NAME"

# Build and run the 'evaluate' command
EVALUATE_CMD="LOG_LEVEL=DEBUG poetry run main evaluate \
  --settings \"$SETTINGS\" \
  --stats-path \"stats_${RUN_NAME}.yml\" \
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
if [[ "$EVALUATION_SET" == true ]]; then
  if [[ "$EVALUATION_BOTH" == false ]]; then
    EVALUATE_CMD+=" --evaluation-set"
  fi
fi
echo "Running EVALUATE..."
echo "Command: $EVALUATE_CMD"
eval "$EVALUATE_CMD" 2>&1 | tee "$EVALUATE_OUT"

mkdir -p "runs/$RUN_NAME"
mv "$EVALUATE_OUT" "runs/$RUN_NAME/"
mv "stats_${RUN_NAME}.yml" "runs/$RUN_NAME/"
cp "$SETTINGS" "runs/$RUN_NAME/"
mkdir -p "runs/$RUN_NAME/logs"
if [[ -d "logs/$LOG_DIR_NAME" ]]; then
    echo "Moving evaluation logs (logs/$LOG_DIR_NAME) to runs/$RUN_NAME/logs"
    mkdir -p "runs/$RUN_NAME/logs/"
    mv "logs/$LOG_DIR_NAME" "runs/$RUN_NAME/logs/eval"
fi

if [[ "$EVALUATION_BOTH" == true ]]; then 
    EVALUATE_CMD+=" --evaluation-set"
    echo "Running EVALUATE..."
    echo "Command: $EVALUATE_CMD"
    eval "$EVALUATE_CMD" 2>&1 | tee "$EVALUATE_OUT"

    mkdir -p "runs/$RUN_NAME/eval_set"
    mv "$EVALUATE_OUT" "runs/$RUN_NAME/eval_set/"
    mv "stats_${RUN_NAME}.yml" "runs/$RUN_NAME/eval_set/"
    mkdir -p "runs/$RUN_NAME/eval_set/logs"
    if [[ -d "logs/$LOG_DIR_NAME" ]]; then
        echo "Moving evaluation logs (logs/$LOG_DIR_NAME) to runs/$RUN_NAME/eval_set/logs"
        mkdir -p "runs/$RUN_NAME/eval_set/logs/"
        mv "logs/$LOG_DIR_NAME" "runs/$RUN_NAME/eval_set/logs/eval"
    fi
fi

echo "Evaluation completed. Results saved in runs/$RUN_NAME"