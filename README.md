# ccaction
This project is developed by Cybercore AI for AI City Challenge 2022 Track 3.
The project is based on the open source [mmaction2](https://github.com/open-mmlab/mmaction2)

# Installation
## Environment Setup
a. Create a conda virtual environment and activate it.

```shell
conda create -n user_tra python=3.7 -y
conda activate user_ccaction
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
a. We recommend you to install MMAction2 with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install openmim
mim install mmaction2==0.17.0
```
MIM can automatically install OpenMMLab projects and their requirements.

b. Required packages 
 - [decord](https://github.com/dmlc/decord) (optional, 0.4.1+): Install CPU version by `pip install decord==0.4.1` and install GPU version from source
 - [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (optional): `pip install PyTurboJPEG`
 - [denseflow](https://github.com/open-mmlab/denseflow) (optional): See [here](https://github.com/innerlee/setup) for simple install scripts.
 - [moviepy](https://zulko.github.io/moviepy/) (optional): `pip install moviepy`. See [here](https://zulko.github.io/moviepy/install.html) for official installation. **Note**(according to [this issue](https://github.com/Zulko/moviepy/issues/693)) that:

    For Linux users, there is a need to modify the /etc/ImageMagick-6/policy.xml file by commenting out ```<policy domain="path" rights="none" pattern="@*" /> to <!-- <policy domain="path" rights="none" pattern="@*" /> -->```, if ImageMagick is not detected by moviepy or you face with any bug related to moviepy.
## Optinal packages
a. MMdetection
```shell
mim install mmdet
```
b. MMPose
```shell
# We need to manual install this package, (not supported with mim yet)
pip install git+https://github.com/open-mmlab/mmtracking.git 
pip install git+https://github.com/svenkreiss/poseval.git
mim install mmpose
```
c. Timm 
```shell
pip install timm
```

d. Apex for inference
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Install CCAction package
```shell
python setup.py develop
```
After running setup, the config folder should look like this:
```
configs
    |-mmaction/
    |-mmdet/
    |-mmpose/
    |-{cc-developed-cfg}
```

## Features
### Tools
- [x] [Train Overfitting](docs/train_overfit.md)
- [x] [Measure FPS and FLOPS](docs/fps_flops.md)
### Networks
- [x] [Swin Transformer Backbone](configs/recognition/swin/README.md)
- [x] [MoViNet Backbone](configs/recognition/movinet/README.md)