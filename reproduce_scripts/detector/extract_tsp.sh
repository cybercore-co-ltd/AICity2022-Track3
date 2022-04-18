CKPT=$1
OUTPUT=$2
# mkdir A1 A2
# reproduce_scripts/detector/extract_tsp_features/extract_tsp_1.sh $CKPT &
# reproduce_scripts/detector/extract_tsp_features/extract_tsp_2.sh $CKPT &
# reproduce_scripts/detector/extract_tsp_features/extract_tsp_3.sh $CKPT &
# reproduce_scripts/detector/extract_tsp_features/extract_tsp_4.sh $CKPT

mkdir -p tsp_features/round1
python tools/detector/concat_multiview_feats.py A1/ $OUTPUT 
python tools/detector/concat_multiview_feats.py A2/ $OUTPUT
rm A1/* A2/*
# tsp_features/round1/
# 118.69.233.170:60001/open/AICity/track3/detector/ckpt/round1_tsp_62.5_student.pth
# http://118.69.233.170:60001/open/AICity/track3/ssc/best_multiview_ckpt_e12.pth