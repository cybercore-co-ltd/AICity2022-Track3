#!/usr/bin/env bash
# set -e
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


CFG="configs/aicity/convnext_noforgetting_tsp_333_224_pseudoA2_v1.py"
WORK_DIR='work_dirs/work_dirs/convnext_noforgetting_tsp_round2'
mim train mmaction $CFG --work-dir $WORK_DIR --gpus 4 --launcher pytorch --validate 

# When want to resume training, use the following command:
# CKPT=$WORK_DIR/epoch_1.pth
# mim train mmaction $CFG --work-dir $WORK_DIR --gpus 4 --launcher pytorch --validate --resume-from $CKPT