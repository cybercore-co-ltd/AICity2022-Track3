# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2)

# Installation

Please follow the steps in [docs/Env_setup.md](docs/Env_setup.md). We provide both Docker and Conda Env.
# Training 

## 1. Pretrain the Backbone to extract feature:
Please follow the steps in [docs/Pretrain_backbone.md](docs/Pretrain_backbone.md)
## 2. Train the Second Stage Classifier:
Please follow the steps in [docs/second_stage_classifier.md](docs/second_stage_classifier.md)

# Inference on new dataset

### Step 1. Run the detector to create proposals 
TBD 

### Step 2. Inference classification from action-former proposal
```bash
./reproduce_scripts/second_stage_classifier/inference.sh $TEST_VIDEO_DIR $PROPOSAL_JSON
```
For example:
```
./reproduce_scripts/second_stage_classifier/inference.sh ./data/raw_video/A2 ./actionformer.json
```
After running this script: we have result file: ./actionformer_mulview_ssc.json which is used for post-processing.