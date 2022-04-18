PREDICTION=$1
rm submit.json
rm submit.txt
python tools/detector/f1_eval.py $PREDICTION
python tools/detector/convert_json2txt.py 'submit.json'