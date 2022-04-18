VIDEO_DIR=$1 #Folder contained video/csv-label the same format as organizer

#------------ A2
USER_ID='56306 72519 42271 65818 79336' # user-ids which are wanted to val


LABEL_FILE='dashboard_val_without_bg_rawframes.csv' # file label csv for training/evaluation
OUTDIR=$2 #example: '/ssd3/data/ai-city-2022/Track3/test_prepare_train_video_ssc'

python tools/ssc/trim_clip_ssc.py --video-dir $VIDEO_DIR --label-file $LABEL_FILE \
                    --user-id $USER_ID --outdir $OUTDIR

#---------- extract trimmed-clip to rawframes
OUTDIR_RAW=$3 #example: "/ssd3/data/ai-city-2022/Track3/test_prepare_train_rawframe_ssc"

python tools/ssc/extract_rawframe_data.py --video-dir $OUTDIR --label-file $LABEL_FILE --outdir $OUTDIR_RAW
