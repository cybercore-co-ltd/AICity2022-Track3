CKPT=$1
OUTPUT=$2
reproduce_scripts/detector/extract_tsp_features/extract_tsp_1.sh $CKPT &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_2.sh $CKPT &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_3.sh $CKPT &
reproduce_scripts/detector/extract_tsp_features/extract_tsp_4.sh $CKPT &
mkdir A1 A2
mkdir tsp_features/outputs
python tools/detector/concat_multiview_feats.py A1/ tsp_features/outputs/
python tools/detector/concat_multiview_feats.py A2/ tsp_features/outputs/
