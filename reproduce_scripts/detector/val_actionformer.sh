export CUDA_VISIBLE_DEVICES=0
CFG=$1
CKPT=$2
OUTPUT=$3
#eval
python actionformer/eval.py \
$CFG \
$CKPT -p 10 --output_file $OUTPUT
echo "Finish !!!"
