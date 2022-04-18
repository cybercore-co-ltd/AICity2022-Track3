export  CUDA_VISIBLE_DEVICES=1
python tools/detector/extract_feats.py \
--part 2 \
--filelist reproduce_scripts/detector/extract_tsp_features/file_list_A1.txt \
--in_dir /ssd3/data/ai-city-2022/Track3/raw_frames/full_video/A1/ \
--out_dir A1
