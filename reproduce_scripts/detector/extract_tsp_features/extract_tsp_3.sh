export  CUDA_VISIBLE_DEVICES=2
CKPT=$1
FILE_LIST=$2
INTPUT_DIR=$3
OUTPUT_DIR=$4
python tools/detector/extract_feats.py \
--ckpt $CKPT \
--part 1 \
--filelist $FILE_LIST \
--in_dir $INTPUT_DIR \
--out_dir $OUTPUT_DIR
