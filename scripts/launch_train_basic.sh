#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

set -e

PYTHON_CMD="uv run python"
CONFIG="pytorch_impl/basic/config.yaml"
USE_WANDB=false
RUN_NAME=""

read -r -a PYTHON_CMD_ARR <<< "$PYTHON_CMD"

CMD=("${PYTHON_CMD_ARR[@]}" -m pytorch_impl.basic.train --config "$CONFIG")

if [ "$USE_WANDB" = true ]; then
    CMD+=(--wandb)
fi
if [ -n "$RUN_NAME" ]; then
    CMD+=(--run-name "$RUN_NAME")
fi

exec "${CMD[@]}"
