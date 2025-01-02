#!/usr/bin/env bash

set -e
trap "popd > /dev/null" EXIT
pushd /gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/master_thesis/legacy_code > /dev/null

# Default parameter values
SETTINGS=""
STATS_PATH=""
DESCRIPTION=""
RUN_DIR=""
SHORT_DESCRIPTION=""

# Print usage
usage() {
  cat <<EOF
Usage: $0 [options]

Required (one of):
  --run-dir <path>    Path to the directory containing results from the evaluate script
  AND
  --short_description <str>
                      Used to prefix the output file (if using --run-dir)
                      else: Used as prefix for the run directory name and output files (if not using --run-dir)
  OR
  --settings <path>   Path to the settings YAML file
  AND
  --stats-path <path> Path to the stats YAML file containing answers to analyze

Optional:
  --description <str> Description for 'analyse'
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
    --stats-path)
      STATS_PATH="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
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
if [[ -z "$RUN_DIR" && ( -z "$SETTINGS" || -z "$STATS_PATH" ) ]]; then
  echo "Error: Either --run-dir must be provided OR both --settings and --stats-path must be provided."
  usage
fi

if [[ -n "$RUN_DIR" ]]; then
  # Reuse files from a previous run
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: Directory $RUN_DIR does not exist."
    exit 1
  fi

  SETTINGS=$(find "$RUN_DIR" -maxdepth 1 -name "settings*.yml" | head -n1)
  STATS_PATH=$(find "$RUN_DIR" -maxdepth 1 -name "stats*.yml" | head -n1)
  ANALYSE_OUT="$RUN_DIR/$(basename "$RUN_DIR")_analyse_out.txt"

  if [[ -z "$SETTINGS" ]]; then
  echo "Error: No settings*.yml file found in $RUN_DIR."
  exit 1
  fi

  if [[ ! -f "$STATS_PATH" ]]; then
    echo "Error: Stats file not found in $RUN_DIR."
    exit 1
  fi
else
  # Generate new names if --run-dir is not used
  if [[ -z "$SHORT_DESCRIPTION" ]]; then
    echo "Error: --short_description is required when not using --run-dir."
    usage
  fi

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="${SHORT_DESCRIPTION}_${TIMESTAMP}"
  ANALYSE_OUT="${RUN_NAME}_analyse_out.txt"
fi

# Build the 'analyse' command
ANALYSE_CMD="LOG_LEVEL=DEBUG poetry run main analyse \
  --settings \"$SETTINGS\" \
  --stats-path \"$STATS_PATH\" \
  --log-dir \"${RUN_NAME:-$(basename "$RUN_DIR")}\""

if [[ -n "$DESCRIPTION" ]]; then
  ANALYSE_CMD+=" --description \"$DESCRIPTION\""
fi

# Run the analyse command
echo "Running ANALYSE..."
eval "$ANALYSE_CMD" 2>&1 | tee "$ANALYSE_OUT"

# If not reusing run_dir, organize new output
if [[ -z "$RUN_DIR" ]]; then
  echo "Creating directory runs/$RUN_NAME and moving files..."
  mkdir -p "runs/$RUN_NAME"
  cp "$STATS_PATH" "runs/$RUN_NAME/"
  cp "$SETTINGS" "runs/$RUN_NAME/"
fi

mkdir -p "runs/$RUN_NAME/logs"
if [[ -d "logs/$LOG_DIR_NAME" ]]; then
    echo "Moving analyse logs to runs/$RUN_NAME/logs"
    mv "logs/$LOG_DIR_NAME" "runs/$RUN_NAME/logs/analyse"
fi
mv "$ANALYSE_OUT" "runs/$RUN_NAME/"

echo "Analysis completed. Results saved in ${RUN_DIR:-runs/$RUN_NAME}"
