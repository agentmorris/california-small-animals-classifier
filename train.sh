#!/bin/bash
# Universal training launcher (path-agnostic).
#
# Run it FROM THE REPO ROOT, already in the correct conda environment. It reads the output base
# (OUT) from the --path-config JSON, resolves the run folder <OUT>/runs/<run-name>, refuses to
# reuse an existing run folder unless --resume is passed, tees the log, and runs src/train.py.
# All arguments are passed through to src/train.py; --run-name and --path-config are required.
#
# Example:
#   bash train.sh --path-config configs/this-machine.json --run-name eva02-llrd \
#       --devices 2 --batch-size 24 --workers 12 --layer-decay 0.75 --lr 1e-4 --warmup-steps 500 \
#       --epochs 8 --patience 3 --intermediate-checkpoints-per-epoch 8 \
#       --checkpoint-folder ~/data/checkpoints-eva02-llrd
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

args=("$@")
RUN_NAME=""
PATH_CONFIG=""
RESUME=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --run-name)            RUN_NAME="${args[$((i + 1))]}" ;;
    --run-name=*)          RUN_NAME="${args[$i]#--run-name=}" ;;
    --path-config)         PATH_CONFIG="${args[$((i + 1))]}" ;;
    --path-config=*)       PATH_CONFIG="${args[$i]#--path-config=}" ;;
    --resume | --resume=*) RESUME=1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "ERROR: --run-name is required" >&2
  exit 1
fi
if [ -z "$PATH_CONFIG" ]; then
  echo "ERROR: --path-config is required" >&2
  exit 1
fi

# Read OUT (output base) from the path-config JSON via the active env's python.
OUT=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['OUTPUT_FOLDER'])" "$PATH_CONFIG")
if [ -z "$OUT" ]; then
  echo "ERROR: could not read OUT from path-config: $PATH_CONFIG" >&2
  exit 1
fi

RUN_DIR="$OUT/runs/$RUN_NAME"
LOG="$RUN_DIR/train_${RUN_NAME}.log"

# Never reuse a run folder unless explicitly resuming it.
if [ -z "$RESUME" ] && [ -e "$RUN_DIR" ]; then
  echo "ERROR: run folder already exists: $RUN_DIR" >&2
  echo "Pick a new --run-name, or pass --resume to continue it." >&2
  exit 1
fi
mkdir -p "$RUN_DIR/checkpoints"

echo "run name: $RUN_NAME"
echo "run dir:  $RUN_DIR"
echo "log:      $LOG"
if [ -n "$RESUME" ]; then
  python -u src/train.py "${args[@]}" 2>&1 | tee -a "$LOG"
else
  python -u src/train.py "${args[@]}" 2>&1 | tee "$LOG"
fi
