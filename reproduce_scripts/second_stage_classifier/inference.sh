set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG="configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py"
CKPT="http://118.69.233.170:60001/open/AICity/track3/vidconv_classifier/best_multiview_ckpt_e12.pth"
VIDEO_DIR="/ssd3/data/ai-city-2022/Track3/raw_video/A2/"
PROPOSAL_FILE="actionformer_mulviews_thebest_0804.json"

python ccaction/tools/ssc/inference_ssc.py --config $CFG --checkpoint $CKPT --proposal-thr 0.3 \
                                        --video-dir $VIDEO_DIR --proposal $PROPOSAL_FILE