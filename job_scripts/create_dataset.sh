#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N create_dataset
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -j y
#$ -o logs/create_dataset.log

set -e
set -o pipefail
fsync -d 60 "$SGE_STDOUT_PATH" &

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
IMAGES_DIR="/afs/crc/group/cvrl/czajka/gbir2/aczajka/BXGRID/iris_segmented_SegNet"
OUT_DIR="/project01/cvrl/jporubci/ArcIris Dataset"
VAL_SPLIT="0.2"
TEST_SPLIT="0.2"
IMG_UID_MAP="img_uid_map.json"
MAX_THREADS="32"
SEED=42

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python utils/create_dataset.py "$IMAGES_DIR" "$OUT_DIR" "$VAL_SPLIT" "$TEST_SPLIT" "$IMG_UID_MAP" "$MAX_THREADS" "$SEED"
