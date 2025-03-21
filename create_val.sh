#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N ArcIris2_create_val
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -j y
#$ -o create_val.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
VAL_DATASET="val"
BIN_FILE="val.bin"
NUM_PAIRS=4418

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python utils/create_val_bin.py --use_mxnet --val_dataset "$VAL_DATASET" --output "$BIN_FILE" --num_pairs "$NUM_PAIRS"
