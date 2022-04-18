export  CUDA_VISIBLE_DEVICES=3
python tools/detector/extract_feats.py \
--part 2 \
--filelist reproduce_scripts/detector/extract_tsp_features/file_list_A2.txt \
--in_dir /ssd3/data/ai-city-2022/Track3/raw_frames/full_video/A2/ \
--out_dir A2
