# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2)

# Installation


## Data Download and extract frame:
We assume that the dataset is downloaded and placed (or have symbolic-link) in the following structure :
```
├── ccaction/
├── configs/
├── README.md
├── reproduce_scripts/
├── setup.py
├── tools/
├── data/ 
    └──raw_video/ -> <download_folder>/ai-city-2022_track3
        ├── A1/
        ├── A2/
        ├── Distracted_Activity_Class_definition.txt
        ├── README.txt
        └── video_ids.csv
```

[Optional] For reproducing purpose and quick evaluation, we also provide our Manual Labels for the A2 set in:
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
# Training 

## 1. Pretrain the Backbone to extract feature:
Please follow the steps in [docs/Pretrain_backbone.md](docs/Pretrain_backbone.md)
## 2. Train the Second Stage Classifier:
Please follow the steps in [docs/second_stage_classifier.md](docs/second_stage_classifier.md)
