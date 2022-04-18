# Environment Setup 

We can use either Docker or Conda for building environment.
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
  docker run --gpus all --shm-size=200g -it -v {DATA_DIR}:/mmdetection/data cctrack3
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