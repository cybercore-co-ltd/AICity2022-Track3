# Second-Stage Convnext Classification with multi-view cameras

## Prepare Datasets:

+ prepare for A1:
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A1.sh 
```
After running this scripts: raw-frames dataset usded for training is placed at folder: **./data/second_stage_classifier/train_trimmed_rawframes** with label file: **./data/second_stage_classifier/dashboard_train_without_bg_rawframes.csv**

Note: A1 folder which is the same structure video A1 folder of ai-city-track3

+ prepare for A2:
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A2.sh
```
After running this scripts: raw-frames dataset usded for testing is placed at folder: **./data/second_stage_classifier/val_trimmed_rawframes** with label file: **./data/second_stage_classifier/dashboard_val_without_bg_rawframes.csv**

Note: please put each reproduce_scripts/A2_local_labels to relevant user-id folders


## Quick Run: 
### 1. Inference classification from action-former proposal
```bash
./reproduce_scripts/second_stage_classifier/inference.sh $TEST_VIDEO_DIR $PROPOSAL

Example: ./reproduce_scripts/second_stage_classifier/inference.sh ./data/A2 ./actionformer.json
```
After running this script: we have result file: ./actionformer_mulview_ssc.json which is used for post-processing.

### 2. Training
```bash
./reproduce_scripts/second_stage_classifier/train.sh
```
### 3. Testing
```bash
./reproduce_scripts/second_stage_classifier/test.sh
```