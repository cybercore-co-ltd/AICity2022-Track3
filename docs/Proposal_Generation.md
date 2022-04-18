# QUICK RUN: ACTIONFORMER-PROPOSAL GENERATION

## ROUND 1
### 1. Train TSP
save as: *round1_tsp_62.5_student.pth*

### 2. Extract TSP Features Round 1:
  
```
./reproduce_scripts/detector/extract_tsp.sh http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round1_tsp_62.5_student.pth tsp_features/round1/
```
**Note:** The tsp features are saved at: *tsp_features/round1/*


### 3. Train Round 1: Using A1 dataset
#### 3.1 Install:
```
cd actionformer/utils
python setup.py install --user
cd ../..
```
#### 3.2 Download annotations:
Download [round1 annotaions]('http://118.69.233.170:60001/open/AICity/track3/detector/annotations/a1a2_anns.json') and save at *annotaions/*: 
  
```
./reproduce_scripts/detector/train_actionformer.sh configs/aicity/actionformer/track3_actionformer_round1.yaml
```
**Note:** The best ckpt is saved at: *ckpt/track3_actionformer_round1/max_map_epoch.pth.tar*

### 4. Evaluate on A2 dataset:
```
./reproduce_scripts/detector/val_actionformer.sh configs/aicity/actionformer/track3_actionformer_round1.yaml ckpt/track3_actionformer_round1/max_map_epoch.pth.tar  result_round1.json
```
**Note:** The output is saved at: *result_round1.json*

### 4. Inference classification from action-former proposal:
```
./reproduce_scripts/second_stage_classifier/inference.sh data/raw_video/A2 result_round1.json
```
**Note:** The output is saved at: *result_round1_ssc.json*

## ROUND 2
### 1. Generate Pseudo-A2-v1:
  
```
./reproduce_scripts/detector/gen_pseudo.sh result_round1_ssc.json data/raw_frames/full_video/A2 data/raw_frames/full_video/raw_frames/combine_A1A2Pseudo_vidconv_round1_bg/ data/raw_frames/full_video/raw_frames/combine_A1A2Pseudo_vidconv_round1_bg.txt
```

**Note:** The pseudo-A2 data are saved at: 

+ raw_frames: *data/raw_frames/combine_A1A2Pseudo_vidconv_round1_bg/*

+ label: *data/raw_frames/combine_A1A2Pseudo_vidconv_round1_bg.txt*

### 3. Train TSP

### 4. Extract TSP Features Round 2:
```
./reproduce_scripts/detector/extract_tsp.sh http://118.69.233.170:60001/open/AICity/track3/detector/ckpt/round2_tsp_67.5.pth tsp_features/round2/
```
**Note:** The tsp features are saved at: *tsp_features/round2/*

### 5. Train Round 2: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v1):
Download [round2 annotaions]('http://118.69.233.170:60001/open/AICity/track3/detector/annotations/round1_anns.json') and save at annotaions/: 

```
./reproduce_scripts/detector/train_actionformer.sh configs/aicity/actionformer/track3_actionformer_round2.yaml
```
**Note:** The best ckpt is saved at: *ckpt/track3_actionformer_round2/max_map_epoch.pth.tar*

### 6. Evaluate on A2 dataset:
```
./reproduce_scripts/detector/val_actionformer.sh configs/aicity/actionformer/track3_actionformer_round2.yaml ckpt/track3_actionformer_round2/max_map_epoch.pth.tar  result_round2.json
```
**Note:** The output is saved at: *result_round2.json*

### 7. Inference classification from action-former proposal:
```
./reproduce_scripts/second_stage_classifier/inference.sh data/raw_video/A2 result_round2.json
```
**Note:** We use this result for the submission: *result_submission.json*


### 8. Generate submission file:
```
./reproduce_scripts/gen_submit.sh result_submission.json
```
