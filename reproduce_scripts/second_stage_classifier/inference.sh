set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG_MODEL1="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
CKPT_MODEL1="http://118.69.233.170:60001/open/AICity/track3/ssc/best_multiview_ckpt_e12.pth"

CFG_MODEL2="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_S1k.py"
CKPT_MODEL2="http://118.69.233.170:60001/open/AICity/track3/ssc/convnext_s1k_e12_78.57.pth"

DEVICE="cuda:0"

VIDEO_DIR=$1
PROPOSAL_FILE=$2
OUT_DIR="ssc_json_folder"

#------------ inference
python tools/ssc/inference_ssc.py --config-t1k $CFG_MODEL1 --checkpoint-t1k $CKPT_MODEL1 \
                                  --config-s1k $CFG_MODEL2 --checkpoint-s1k $CKPT_MODEL2 \
                                  --proposal-thr 0.3 \
                                  --device $DEVICE \
                                  --video-dir $VIDEO_DIR --proposal $PROPOSAL_FILE --outdir $OUT_DIR


#------------ combine all video json files for post-process
python tools/ssc/combine_json.py --proposal $PROPOSAL_FILE --outdir $OUT_DIR # output will be: actionformer_mulview_ssc.json