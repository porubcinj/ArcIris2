#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N ArcIris2
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*|qa-rtx6k*"

set -e
set -o pipefail

PROJECT_DIR="/afs/crc.nd.edu/user/j/jporubci/Private/ArcIris2"
IMAGES_DIR="/afs/crc/group/cvrl/czajka/gbir2/aczajka/BXGRID/iris_segmented_SegNet"
TRAIN_DIR="/project01/cvrl/jporubci/ArcIris Images/train"
TEST_DIR="/project01/cvrl/jporubci/ArcIris Images/test"
IMG_UID_MAP="img_uid_map.json"
TRAIN_TEST_SPLIT=0.8
SEED=42

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python utils/create_dataset.py "$IMAGES_DIR" "$TRAIN_DIR" "$TEST_DIR" "$IMG_UID_MAP" "$TRAIN_TEST_SPLIT"
python -m mxnet.tools.im2rec --list --recursive train "$TRAIN_DIR"
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train "$TRAIN_DIR"
python -m mxnet.tools.im2rec --list --recursive test "$TEST_DIR"
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 test "$TEST_DIR"
