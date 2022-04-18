# Extract frames
python tools/pretrain_backbone/extract_frames_by_class.py \
        --src-dir data/raw_video/A2 \
        --out-dir data/raw_frames/A2

# Create labels
python tools/pretrain_backbone/build_file_list.py \
        --src-dir data/raw_frames/A2 \
        --out-file data/raw_frames/A2.txt
