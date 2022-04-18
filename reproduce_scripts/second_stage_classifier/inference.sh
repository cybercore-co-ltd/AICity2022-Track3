set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG_T1K="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
CKPT_T1K="http://118.69.233.170:60001/open/AICity/track3/ssc/best_multiview_ckpt_e12.pth"

CFG_S1K="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_S1k.py"
CKPT_S1K="http://118.69.233.170:60001/open/AICity/track3/ssc/convnext_s1k_e12_78.57.pth"

DEVICE="cuda:0"

VIDEO_DIR="/ssd3/data/ai-city-2022/Track3/raw_video/A2/" # folder which store video to test
PROPOSAL_FILE="actionformer_ssc_1.json" #proposal from action-former
OUT_DIR="ssc_json_folder"

#------------ inference
python tools/ssc/inference_ssc.py --config-t1k $CFG_T1K --checkpoint-t1k $CKPT_T1K \
                                  --config-s1k $CFG_S1K --checkpoint-s1k $CKPT_S1K \
                                  --proposal-thr 0.3 \
                                  --device $DEVICE \
                                  --video-dir $VIDEO_DIR --proposal $PROPOSAL_FILE --outdir $OUT_DIR


#------------ combine all video json files for post-process
python tools/ssc/combine_json.py --proposal $PROPOSAL_FILE --outdir $OUT_DIR # output will be: actionformer_mulview_ssc.json