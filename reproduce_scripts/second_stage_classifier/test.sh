#!/usr/bin/env bash
set -e
# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,3
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
WORKDIR="work_dirs/ssc-mulviews"
CKPT=$WORK_DIR/epoch_12.pth
mim test mmaction $CFG --checkpoint $CKPT --gpus 2 --fuse-conv-bn \
            --gpus 2 --launcher pytorch --eval top_k_accuracy