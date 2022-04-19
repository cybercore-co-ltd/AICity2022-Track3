set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG_MODEL="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
CKPT_MODEL="http://118.69.233.170:60001/open/AICity/track3/ssc/best_multiview_ckpt_e12.pth"

DEVICE="cuda:0"

VIDEO_DIR=$1
PROPOSAL_FILE=$2
OUTPUT=$3
OUT_DIR="ssc_json_folder"

#------------ inference
python tools/ssc/inference_ssc.py --config $CFG_MODEL --checkpoint $CKPT_MODEL \
                                  --proposal-thr 0.3 \
                                  --device $DEVICE \
                                  --video-dir $VIDEO_DIR --proposal $PROPOSAL_FILE --outdir $OUT_DIR


#------------ combine all video json files for post-process
python tools/ssc/combine_json.py --proposal $PROPOSAL_FILE --outdir $OUT_DIR --output $OUTPUT