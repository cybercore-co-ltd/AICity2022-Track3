export CUDA_VISIBLE_DEVICES=0
CONFIG=$1
python actionformer/train.py \
$CONFIG -c 5 -p 1
