export  CUDA_VISIBLE_DEVICES=1
CKPT=$1
python tools/detector/extract_feats.py \
--ckpt $CKPT \
--part 2 \
--filelist reproduce_scripts/detector/extract_tsp_features/file_list_A1.txt \
--in_dir data/raw_frames/full_video/A1/ \
--out_dir A1
