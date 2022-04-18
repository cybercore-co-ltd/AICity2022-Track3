## 1. Pretrain backbone with TSP and Learning without Forgetting: 

### 1.1. Train Round 1: Using A1 dataset, run:

Prepare dataset:

```
python 
```
```
./reproduce_scripts/pretrain_backbone/train_round1.sh
```
**Note:** Due to Random Drop-out and small dataset, the Top-1 Accuracy can be in range (60.1 - 63.5). 

### 1.2. Train Round 2: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v1), run:
```
./reproduce_scripts/pretrain_backbone/train_round2.sh
```
**Note:** We use result of Round 2 for the submission.

### 1.3. Train Round 3: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v2): run:
```
./reproduce_scripts/pretrain_backbone/train_round2.sh
```
**Note:** Round 3 is overfited to pseudo-label and not used for Public test. 