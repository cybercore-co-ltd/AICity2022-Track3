PREDICTION=$1
OUTPUT="submit.json"
rm submit.json
rm submit.txt
python tools/detector/f1_eval.py $PREDICTION $OUTPUT
python tools/detector/convert_json2txt.py 