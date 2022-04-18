# Second-Stage Convnext Classification with multi-view cameras

## Prepare Datasets:

+ prepare for A1:
We expect the data folder has following structure:

```
data/raw_video/A2
                ├── user_id_42271
                │   ├── Dashboard_user_id_42271_NoAudio_3.MP4
                │   ├── Dashboard_user_id_42271_NoAudio_4.MP4
                │   ├── Rear_view_user_id_42271_NoAudio_3.MP4
                │   ├── Rear_view_user_id_42271_NoAudio_4.MP4
                │   ├── Right_side_window_user_id_42271_NoAudio_3.MP4
                │   ├── Right_side_window_user_id_42271_NoAudio_4.MP4
                │   └── user_id_42271.csv
                ├── user_id_56306
                │   ├── Dashboard_user_id_56306_NoAudio_2.MP4
                │   ├── Dashboard_user_id_56306_NoAudio_3.MP4
                │   ├── Rear_view_user_id_56306_NoAudio_2.MP4
                │   ├── Rear_view_user_id_56306_NoAudio_3.MP4
                │   ├── Rightside_window_user_id_56306_NoAudio_2.MP4
                │   ├── Rightside_window_user_id_56306_NoAudio_3.MP4
                │   └── user_id_56306.csv
                ...
```
Please run the following commands to prepare data for training.
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A1.sh 
```
This script will extract:
+ raw-frames at the folder: `data/second_stage_classifier/train_trimmed_rawframes` 
+ and the label csv file: `data/second_stage_classifier/dashboard_train_without_bg_rawframes.csv`

+ prepare for A2:
For reproducing purpose and quick evaluation, we also provide our Manual Labels for the A2 set in folder:
```
reproduce_scripts/A2_local_labels
```        
Please copy the corresponding labels file for each user to the folder A2 similar to A1 folder:
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
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A2.sh
```
This script will extract:
+ raw-frames at the folder: `data/second_stage_classifier/val_trimmed_rawframes` 
+ and the label csv file: `data/second_stage_classifier/dashboard_val_without_bg_rawframes.csv`
  
## Training and Testing scripts

### 1. Training
```bash
./reproduce_scripts/second_stage_classifier/train.sh
```
### 2. Evaluate on A2 dataset 
```bash
./reproduce_scripts/second_stage_classifier/test.sh
```