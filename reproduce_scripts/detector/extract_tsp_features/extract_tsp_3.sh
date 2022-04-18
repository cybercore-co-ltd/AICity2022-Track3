export  CUDA_VISIBLE_DEVICES=2
CKPT=$1
python tools/detector/extract_feats.py \
--ckpt $CKPT \
--part 1 \
--filelist reproduce_scripts/detector/extract_tsp_features/file_list_A2.txt \
--in_dir data/raw_frames/full_video/A2/ \
--out_dir A2
