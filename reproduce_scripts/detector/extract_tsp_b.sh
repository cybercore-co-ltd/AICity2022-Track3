CKPT=$1
INPUT_VIDEO=$2
OUTPUT_VIDEO=$3
mkdir B
ls $INPUT_VIDEO > file_list_B.txt
reproduce_scripts/detector/extract_tsp_features/extract_tsp_1.sh $CKPT file_list_B.txt $INPUT_VIDEO $OUTPUT_VIDEO &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_2.sh $CKPT file_list_B.txt $INPUT_VIDEO $OUTPUT_VIDEO &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_3.sh $CKPT file_list_B.txt $INPUT_VIDEO $OUTPUT_VIDEO &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_4.sh $CKPT file_list_B.txt $INPUT_VIDEO $OUTPUT_VIDEO

mkdir -p tsp_features/round_b
python tools/detector/concat_multiview_feats.py B/ $OUTPUT_VIDEO 