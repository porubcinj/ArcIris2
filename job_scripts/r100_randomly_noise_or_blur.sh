#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N AI2r1rnob
#$ -q gpu
#$ -l gpu=2
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -j y
#$ -o logs/r100_randomly_noise_or_blur.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
CONFIG="r100_randomly_noise_or_blur"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

stdbuf -oL python train.py "configs/$CONFIG"
