VIDEO_DIR="./data/raw_video/A1"

#------------ A1
USER_ID='24026 24491 35133 38058 49381' # user-ids which are wanted to train


OUTDIR="./data/second_stage_classifier/train_trimmed_video" 
LABEL_FILE='./data/second_stage_classifier/dashboard_train_without_bg_rawframes.csv' # file label csv for training/evaluation

python tools/ssc/trim_clip_ssc.py --video-dir $VIDEO_DIR --label-file $LABEL_FILE \
                    --user-id $USER_ID --outdir $OUTDIR

#---------- extract trimmed-clip to rawframes
OUTDIR_RAW="./data/second_stage_classifier/train_trimmed_rawframes"

python tools/ssc/extract_rawframe_data.py --video-dir $OUTDIR --label-file $LABEL_FILE --outdir $OUTDIR_RAW

rm -rf $OUTDIR