export CUDA_VISIBLE_DEVICES=0
#eval
python actionformer/eval.py \
configs/track3/track3_actionformer.yaml \
checkpoints/epoch_030_map_27.57.pth.tar -p 10 --output_file 'result_eval.json'
echo "Finish !!!"
