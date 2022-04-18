DATA_DIR='/ssd3/data/ai-city-2022/Track3/raw_video'
DATA_DIR_FULL_VIDEO='/ssd3/data/ai-city-2022/Track3/raw_frames/full_video'

docker run --rm -it --gpus all  --privileged=true --shm-size 200G\
    -v $DATA_DIR:/cctrack3/data/raw_video\
    -v $DATA_DIR_FULL_VIDEO:/cctrack3/data/full_video\
    cctrack3