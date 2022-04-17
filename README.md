# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2)

# Installation
## Environment Setup
a. Create a conda virtual environment and activate it.

```shell
conda create -n aicity_t3 python=3.7 -y
conda activate aicity_t3
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

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
## Install MMAction2
Install MMAction2 with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install openmim
mim install mmaction2==0.17.0
```
MIM can automatically install OpenMMLab projects and their requirements.

## Install CCAction package
```shell
python setup.py develop
```

# Training 


## 1. Pretrain backbone with TSP and Learning without Forgetting: 

+ Train Round 1: Using A1 dataset, run:
  
```
./reproduce_scripts/pretrain_backbone/train_round1.sh
```
**Note:** Due to Random Drop-out and small dataset, the Top-1 Accuracy can be in range (60.1 - 63.5). 

+ Train Round 2: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v1), run:
```
./reproduce_scripts/pretrain_backbone/train_round2.sh
```
**Note:** We use result of Round 2 for the submission.

+ Train Round 3: Using A1 dataset and Pseudo Label on A2 (Pseudo-A2-v2): run:
```
./reproduce_scripts/pretrain_backbone/train_round2.sh
```
**Note:** Round 3 is overfited to pseudo-label and not used for Public test. 

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