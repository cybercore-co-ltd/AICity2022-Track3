ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install packages
RUN conda clean --all
RUN pip install openmim 
RUN mim install mmaction2==0.21.0
RUN pip install timm imgaug moviepy


# Install AICity2022-Track3
RUN git clone https://github.com/cybercore-co-ltd/AICity2022-Track3.git /cctrack3
WORKDIR /cctrack3
RUN mkdir -p /cctrack3/data/raw_video
ENV FORCE_CUDA="1"
RUN python setup.py develop
