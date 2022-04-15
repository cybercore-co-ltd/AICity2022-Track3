set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
CKPT="http://118.69.233.170:60001/open/AICity/track3/vidconv_classifier/best_multiview_ckpt_e12.pth"
VIDEO_DIR="/raid/data/ai-city-2022/Track3/raw_video/A2/" # folder which store video to test
PROPOSAL_FILE="actionformer_mulviews_thebest_0804.json" #proposal from action-former
OUT_DIR="ssc_json_folder"

#------------ inference
python ccaction/tools/ssc/inference_ssc.py --config $CFG --checkpoint $CKPT --proposal-thr 0.3 \
                                        --video-dir $VIDEO_DIR --proposal $PROPOSAL_FILE --outdir $OUT_DIR


#------------ combine all video json files for post-process
python ccaction/tools/ssc/combine_json.py --proposal $PROPOSAL_FILE --outdir $OUT_DIR # output will be: actionformer_mulview_ssc.json