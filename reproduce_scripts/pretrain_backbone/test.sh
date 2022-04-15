#!/usr/bin/env bash
set -e
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG="configs/recognition/conNext_super_feat/aicity/convnext_noforgetting_tsp_333_224_k400.py"
WORK_DIR="work_dirs/convnext_noforgetting_tsp_pseudoA2_vidconv_V1"
CKPT=$WORK_DIR/epoch_6.pth
mim test mmaction $CFG --checkpoint $CKPT --fuse-conv-bn --gpus 4 --launcher pytorch --eval top_k_accuracy