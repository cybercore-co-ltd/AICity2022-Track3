# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
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
#### 1.1 Extract features on new dataset
```
OUT_DIR="tsp_features/new_dataset/"
CKPT="http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_tsp_67.5.pth"
./reproduce_scripts/detector/extract_tsp.sh  $CKPT $OUT_DIR
```
where:
+ `CKPT` is our pretrained checkpoint.
+ The extracted feature for each video is saved at `OUT_DIR` folder, and procecced in the step 1.2 

#### 1.2 Generate proposals
Using the feature extracted in step 1.1, run the following command to create proposals:
```
CONFIG="configs/aicity/actionformer/track3_actionformer_new.yaml"
CKPT="http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_map_31.55.pth.tar"
PROPOSAL_RESULT="proposals.json"
./reproduce_scripts/detector/val_actionformer.sh $CONFIG $CKPT $PROPOSAL_RESULT 
```
where:
+ `CONFIG` is the model's config file.
+ `CKPT` is the model's checkpoint.
+ `PROPOSAL_RESULT` is the output file, which is used in the step 2.
### Step 2. Inference classification from action-former proposal
```bash
TEST_VIDEO_DIR=<path/to/test_video>
PROPOSAL_RESULT="proposals.json"
./reproduce_scripts/second_stage_classifier/inference.sh $TEST_VIDEO_DIR $PROPOSAL_RESULT
```
For example:
```
./reproduce_scripts/second_stage_classifier/inference.sh ./data/raw_video/A2 proposals.json result_submission.json
```
After running this script: we have result file: ./actionformer_mulview_ssc.json which is used for post-processing in the next step.

### Step 3: post-processing and generate submission file on Server. 
```
./reproduce_scripts/gen_submit.sh result_submission.json
```


# Credits:
We thank [mmaction2](https://github.com/open-mmlab/mmaction2) and [action-former](https://github.com/happyharrycn/actionformer_release) for the code base. Please cite their work if you found this code is helpful.