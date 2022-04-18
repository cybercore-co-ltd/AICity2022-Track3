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

# Inference on new dataset

### Step 1. Run the detector to create proposals 
#### 1.1 Extract tsp features on new dataset
```
./reproduce_scripts/detector/extract_tsp.sh http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_tsp_67.5.pth tsp_features/new_dataset/
```
**Note:** The tsp features are saved at: *tsp_features/new_dataset/*

#### 1.2 Generate proposals
```
./reproduce_scripts/detector/val_actionformer.sh configs/aicity/actionformer/track3_actionformer_new.yaml http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_map_31.55.pth.tar  proposals.json
```

### Step 2. Inference classification from action-former proposal
```bash
./reproduce_scripts/second_stage_classifier/inference.sh $TEST_VIDEO_DIR $PROPOSAL_JSON
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