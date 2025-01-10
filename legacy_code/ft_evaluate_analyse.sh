
set -e  # Exit on error

# Default parameter values
TEMPERATURE="0.1"
ONLY_MODEL=""
MODEL_PATH=""
PROMPT="Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), choose one main solvent (no mixtures or ratios), choose one chemical additive (write 'None' if no additive present)"
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
  --prompt <str>      Prompt passed to 'evaluate'
  --temperature <num> Temperature passed to 'evaluate' (default: 0.1)
  --description <str> Description for both 'evaluate' and 'analyse'
  --evaluation-set <str>   Evaluation set path passed to 'evaluate'
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
    --finetune-dir)
      FINETUNE_DIR="$2"
      shift 2
      ;;
    --finetune-out-dir)
      FINETUNE_OUT_DIR="$2"
      shift 2
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
if [[ -z "$SETTINGS" ]]; then
  echo "Error: --settings is required."
  usage
fi

if [[ -z "$SHORT_DESCRIPTION" ]]; then
  echo "Error: --short-description is required."
  usage
fi

if [[ -z "$FINETUNE_DIR" ]]; then
  echo "Error: --finetune-dir is required."
  usage
fi

if [[ -z "$FINETUNE_OUT_DIR" ]]; then
  echo "Error: --finetune-out-dir is required."
  usage
fi

trap "popd > /dev/null" EXIT
pushd $FINETUNE_DIR > /dev/null # TODO Chang edir
bash run_tune.sh "$(echo "$FINETUNE_OUT_DIR" | rev | cut -d'/' -f2 | rev)"

# Build the 'evaluate' command
EA_CMD="bash evaluate_and_analyse.sh \
  --settings \"$SETTINGS\" \
  --temperature \"$TEMPERATURE\" \
  --short-description \"$SHORT_DESCRIPTION\""

if [[ -n "$ONLY_MODEL" ]]; then
  EA_CMD+=" --only-model \"$ONLY_MODEL\""
fi
if [[ -n "$MODEL_PATH" ]]; then
  EA_CMD+=" --model-path \"$MODEL_PATH\""
fi
if [[ -n "$PROMPT" ]]; then
  EA_CMD+=" --prompt \"$PROMPT\""
fi
if [[ -n "$DESCRIPTION" ]]; then
  EA_CMD+=" --description \"$DESCRIPTION\""
fi
if [[ "$EVALUATION_SET" == true ]]; then
  if [[ "$EVALUATION_BOTH" == true ]]; then
    # do nothing
  else
    EA_CMD+=" --evaluation-set"
  fi
fi

popd > /dev/null
pushd /gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/master_thesis/legacy_code > /dev/null

echo "===== EVALUATE AND ANALYSE COMMAND ====="
echo "$EA_CMD"
echo

# Run evaluation
echo "STARTING EVALUATE AND ANALYSE"
eval "$EA_CMD" 

if [[ "$EVALUATION_BOTH" == true ]]; then
  EA_CMD+=" --evaluation-set"
  echo "===== SECOND EVALUATE AND ANALYSE COMMAND ====="
  echo "$EA_CMD"
  echo

  # Run evaluation
  echo "STARTING SECOND EVALUATE AND ANALYSE"
  eval "$EA_CMD" 
fi

