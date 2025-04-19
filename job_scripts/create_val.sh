#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N create_val
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -j y
#$ -o logs/create_val.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
VAL_IMAGES_DIR="/project01/cvrl/jporubci/ArcIris Dataset/val/images"
NUM_PAIRS=4096
BIN_PATH="val.bin"
SEED=42

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python utils/create_val_bin.py "$VAL_IMAGES_DIR" "$NUM_PAIRS" "$BIN_PATH" "$SEED"
