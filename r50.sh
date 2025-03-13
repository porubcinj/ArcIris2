#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N ArcIris2_r50
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*|qa-rtx6k*"

set -e
set -o pipefail

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
CONFIG="r50"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python train_v2.py "configs/$CONFIG"
