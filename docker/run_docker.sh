DOCKER_NAME=$1

DATA_DIR='/ssd3/data/ai-city-2022/Track3/raw_video'

docker run --rm -it --gpus all  --privileged=true --shm-size 20G\
    -v $DATA_DIR:/cctrack3/data/raw_video \
    --name $DOCKER_NAME cctrack3