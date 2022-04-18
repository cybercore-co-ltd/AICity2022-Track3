# /ssd3/data/ai-city-2022/Track3/raw_frames/combine_A1A2Pseudo_vidconv_round1_bg/$i/
JSON_OUTPUT=$1
INPUT_PSEUDO=$2
OUTPUT_PSEUDO_DATA=$3
OUTPUT_PSEUDO_LABEL=$4
mkdir -p $OUTPUT_PSEUDO_DATA
cp data/raw_frames/full_video/raw_frames/A1 $OUTPUT_PSEUDO_DATA -r
python tools/detector/get_rawframes_a2pseudo.py $JSON_OUTPUT $INPUT_PSEUDO $OUTPUT_PSEUDO_DATA
python tools/detector/build_file_list.py $OUTPUT_PSEUDO_DATA $OUTPUT_PSEUDO_LABEL
echo "Finish !"
