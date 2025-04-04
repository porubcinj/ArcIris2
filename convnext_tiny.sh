#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N ArcIris2_convnext_tiny
#$ -q gpu
#$ -l gpu=2
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -j y
#$ -o convnext_tiny.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
CONFIG="convnext_tiny"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

stdbuf -oL python train_v2.py "configs/$CONFIG"
