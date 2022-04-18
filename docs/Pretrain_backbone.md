# Pretrain backbone with TSP and Learning without Forgetting: 


## 1. Train Round 1: Using A1 dataset

### Prepare dataset:
+ Extract raw frames for dataset A1: 
```
./reproduce_scripts/pretrain_backbone/dataset_prepare_A1.sh
```

+ Extract raw frames for dataset A2:
For reproducing purpose and quick evaluation, we also provide our Manual Labels for the A2 set in:
```
reproduce_scripts/A2_local_labels
```        
Please copy the corresponding labels file for each user to the folder as in A1 folder:
```
data/raw_video/A2
                ├── user_id_42271
                │   ├── Dashboard_user_id_42271_NoAudio_3.MP4
                │   ├── Dashboard_user_id_42271_NoAudio_4.MP4
                │   ├── Rear_view_user_id_42271_NoAudio_3.MP4
                │   ├── Rear_view_user_id_42271_NoAudio_4.MP4
                │   ├── Right_side_window_user_id_42271_NoAudio_3.MP4
                │   ├── Right_side_window_user_id_42271_NoAudio_4.MP4
                │   └── user_id_42271.csv
                ├── user_id_56306
                │   ├── Dashboard_user_id_56306_NoAudio_2.MP4
                │   ├── Dashboard_user_id_56306_NoAudio_3.MP4
                │   ├── Rear_view_user_id_56306_NoAudio_2.MP4
                │   ├── Rear_view_user_id_56306_NoAudio_3.MP4
                │   ├── Rightside_window_user_id_56306_NoAudio_2.MP4
                │   ├── Rightside_window_user_id_56306_NoAudio_3.MP4
                │   └── user_id_56306.csv
                ...
```
then run command:
```
./reproduce_scripts/pretrain_backbone/dataset_prepare_A1.sh
```

### Train model
Run command:
```
./reproduce_scripts/pretrain_backbone/train_round1.sh
```
**Note:** Due to Random Drop-out and small dataset, the Top-1 Accuracy can be in range (60.1 - 63.5). 

### 2. Train Round 2: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v1), run:

```
./reproduce_scripts/pretrain_backbone/train_round2.sh
```
**Note:** We use result of Round 2 for the submission.
