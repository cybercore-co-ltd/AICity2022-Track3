# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2)

# Installation
## Environment setup with Dockerfile
We recommend to use DockerFile for easy reproduction.
1. First, check if you have docker and Nvidia-cuda setup.If not, follow the official website to setup docker:
  ```
  sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
  ```
2. Build Docker file: 
  ```
  docker build -t cctrack3 docker/ 
  ```
3. Run it with:
  ```
  docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data cctrack3
  ```
  where `DATA_DIR=<path-to-AI-CITY-2022-Track3-Downloaded folder>`. We also provide the script to run docker file.
  ```
  ./docker/run_docker.sh 
  ```


## Environment Setup with CONDA
1. Create a conda virtual environment and activate it.

```shell
conda create -n aicity_t3 python=3.7 -y
conda activate aicity_t3
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```
Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.5,
you need to install the prebuilt PyTorch with CUDA 10.2.

```shell
conda install pytorch cudatoolkit=10.2 torchvision -c pytorch
```
3. Install MMAction2
Install MMAction2 with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install openmim
mim install mmaction2==0.17.0
```
MIM can automatically install OpenMMLab projects and their requirements.

4. Install CCAction package
```shell
python setup.py develop
```

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
A2
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



## 2. Train the Second Stage Classifier:

### 2.1 Prepare Datasets:

+ prepare for A1:
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A1.sh A1_video_dir trimmed_video_dir trimmed_rawframe_dir

Example:./reproduce_scripts/second_stage_classifier/dataset_prepare_A1.sh A1 /out/prepair_train_ssc_video /out/prepair_train_ssc_rawframe
```
Output: rawframe trimmed-clip folder + csv-label-file (example: dashboard_train_without_bg_rawframes.csv)

Note: A1 folder which is the same structure video A1 folder of ai-city-track3

+ prepare for A2:
```bash
./reproduce_scripts/second_stage_classifier/dataset_prepare_A2.sh A2_video_dir trimmed_video_dir trimmed_rawframe_dir

Example:./reproduce_scripts/second_stage_classifier/dataset_prepare_A2.sh A2 /out/prepair_val_ssc_video /out/prepair_val_ssc_rawframe
```
Output: rawframe trimmed-clip folder + csv-label-file (example: dashboard_val_without_bg_rawframes.csv)

Note: A2 folder which is the same structure video A1 folder of ai-city-track3 (please put each user-label-csv-file in each user-id-folder as A1 structure)
### 2.2 Train/Test/Inference commands:
Please change the path to rawframe_train/rawframe_val folder + relevant csv-label-file prepared at Section 2.1 to configs/_base_/datasets/aicity_multi_views_9rgb_224_rawframe.py

+ Inference classification from action-former proposal
```bash
./reproduce_scripts/second_stage_classifier/inference.sh
```
+ Training
```bash
./reproduce_scripts/second_stage_classifier/train.sh
```
+ Testing
```bash
./reproduce_scripts/second_stage_classifier/test.sh
```