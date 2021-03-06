# 2022 AI-CITY CHALLENGE - TRACK3: Naturalistic Driving Action Recognition
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3. See details in our [paper](25.pdf).
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2) and [action-former](https://github.com/happyharrycn/actionformer_release)

# Installation

Please follow the steps in [docs/Env_setup.md](docs/Env_setup.md). We provide both Docker and Conda Env.
# Training 

## 1. Pretrain the Backbone to extract feature:
Please follow the steps in [docs/Pretrain_backbone.md](docs/Pretrain_backbone.md)
## 2. Train the Second Stage Classifier:
Please follow the steps in [docs/second_stage_classifier.md](docs/second_stage_classifier.md)

## 3. Train the Detector and Generate Pseudo Labels on A2. 
Please follow the steps in [docs/Proposal_Generation.md](docs/Proposal_Generation.md)

# Inference on testing dataset 

### Step 1. Run the detector to create proposals 
#### 1.1 Extract rawframes on new videos-B:
```
IN_DIR='path/to/videos'
OUT_DIR="data/raw_frames/full_video/B"
reproduce_scripts/detector/extract_rawframes_fullvideo.sh $IN_DIR $OUT_DIR
```
where:
+ `IN_DIR` is the directory of testing videos . For A2 test set as an example, set `IN_DIR=data/raw_video/A2`. For B test set, set `IN_DIR=data/raw_video/B`.
+ The extracted frames for each video is saved at `OUT_DIR` folder, and procecced in the step 1.2 


#### 1.2 Extract features on new dataset-B
```
IN_DIR='path/to/raw_frames/full_video/B'
OUT_DIR="tsp_features/round_b/"
CKPT="http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_tsp_67.5.pth"
./reproduce_scripts/detector/extract_tsp_b.sh  $CKPT $IN_DIR $OUT_DIR
```
where:
+ `IN_DIR` is the directory of testing raw frames. For A2 test set as an example, set `IN_DIR=data/raw_frames/full_video/A2`. For B test set, set `IN_DIR=data/raw_frames/full_video/B`.
+ `CKPT` is our pretrained checkpoint.
+ The extracted feature for each video is saved at `OUT_DIR` folder, and procecced in the step 1.3


#### 1.3 Generate proposals


Using the feature extracted in step 1.1, run the following command to create proposals:
```
CONFIG="configs/aicity/actionformer/track3_actionformer_B.yaml"
CKPT="http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_map_31.55.pth.tar"
PROPOSAL_RESULT="proposals.json"

./reproduce_scripts/detector/val_actionformer.sh $CONFIG $CKPT $PROPOSAL_RESULT 
```
where:
+ `CONFIG` is the model's config file. We assume the extracted feature in Step 1.1. is saved as `tsp_features/round_b/`, which is specified in the config yaml. If you change the directory or testing video, please adjust the config file accordingly.
+ `CKPT` is the model's checkpoint.
+ `PROPOSAL_RESULT` is the output file, which is used in the step 2.
### Step 2. Inference classification from action-former proposal
```bash
TEST_VIDEO_DIR=<path/to/test_video>
PROPOSAL_RESULT="proposals.json"
OUTPUT="result_submission.json"
./reproduce_scripts/second_stage_classifier/inference.sh $TEST_VIDEO_DIR $PROPOSAL_RESULT $OUTPUT
```
For example:
```
./reproduce_scripts/second_stage_classifier/inference.sh \
    ./data/raw_video/A2 \ 
    proposals.json \
    result_submission.json
```
After running this script: we have result file: `result_submission.json` which is used for post-processing in the next step.

### Step 3: post-processing and generate submission file on Server. 
```
INPUT_FILE=result_submission.json
./reproduce_scripts/gen_submit.sh $INPUT_FILE
```
where:
+ `INPUT_FILE` is the output from step 2.
+ This will generate the file `submit.txt` for submission. 
+ NOTE: For submission, we need the `video_ids` file, such as one in `tools/detector/video_ids_A2clean.csv`. To run inference on B dataset (unknown at this time), we need a similar file. See input of `tools/detector/convert_json2txt.py`.

# Credits:
We thank [mmaction2](https://github.com/open-mmlab/mmaction2), [action-former](https://github.com/happyharrycn/actionformer_release), and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) code base. Please cite their works should you find our code is helpful.
