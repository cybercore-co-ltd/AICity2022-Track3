export CUDA_VISIBLE_DEVICES=0
python tools/extract_feats.py \
--part 1 \
--filelist file_list_A1.txt \
--in_dir /ssd3/data/ai-city-2022/Track3/raw_frames/full_video/A1/ \
--out_dir A1
