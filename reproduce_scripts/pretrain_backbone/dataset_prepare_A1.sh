# Extract frames
python tools/pretrain_backbone/extract_frames_by_class.py \
        --src-dir data/raw_video/A1 \
        --out-dir data/raw_frames/A1

# Create labels
python tools/pretrain_backbone/build_file_list.py \
        --src-dir data/raw_frames/A1 \
        --out-file data/raw_frames/A1.txt
