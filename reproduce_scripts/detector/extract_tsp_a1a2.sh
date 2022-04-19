CKPT=$1
INTPUT_DIR=$2
OUTPUT_DIR=$3
mkdir A1 A2
reproduce_scripts/detector/extract_tsp_features/extract_tsp_1.sh $CKPT reproduce_scripts/detector/extract_tsp_features/file_list_A1.txt data/raw_frames/full_video/A1 A1 &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_2.sh $CKPT reproduce_scripts/detector/extract_tsp_features/file_list_A1.txt data/raw_frames/full_video/A1 A1 &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_3.sh $CKPT reproduce_scripts/detector/extract_tsp_features/file_list_A2.txt data/raw_frames/full_video/A2 A2 &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_4.sh $CKPT reproduce_scripts/detector/extract_tsp_features/file_list_A2.txt data/raw_frames/full_video/A2 A2

mkdir -p tsp_features/round1
python tools/detector/concat_multiview_feats.py A1/ $OUTPUT_DIR 
python tools/detector/concat_multiview_feats.py A2/ $OUTPUT_DIR
rm A1/* A2/*