VIDEO_DIR="./data/raw_video/A2"

#------------ A2
USER_ID='56306 72519 42271 65818 79336' # user-ids which are wanted to val


OUTDIR="./data/second_stage_classifier/val_trimmed_video"
LABEL_FILE='./data/second_stage_classifier/dashboard_val_without_bg_rawframes.csv' # file label csv for training/evaluation

python tools/ssc/trim_clip_ssc.py --video-dir $VIDEO_DIR --label-file $LABEL_FILE \
                    --user-id $USER_ID --outdir $OUTDIR

#---------- extract trimmed-clip to rawframes
OUTDIR_RAW="./data/second_stage_classifier/val_trimmed_rawframes"

python tools/ssc/extract_rawframe_data.py --video-dir $OUTDIR --label-file $LABEL_FILE --outdir $OUTDIR_RAW
