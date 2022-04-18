export CUDA_VISIBLE_DEVICES=2
python actionformer/train.py \
configs/track3/track3_actionformer.yaml \
--output round1_pseudo -c 5 -p 1
#  --load_from /home/ccvn/Workspace/ngocnt/actionformer_release/rename_thumos14.pth.tar
